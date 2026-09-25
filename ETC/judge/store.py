"""SQLite-backed identity, immutable submissions and an atomic local queue."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import secrets
import sqlite3
import time
import uuid
import unicodedata

from .catalog import CHALLENGES, ROOT, fingerprint, rules
from .expressions import SubmissionError


class BusyError(ValueError):
    pass


class NicknameConflict(ValueError):
    pass


class NicknameRequired(ValueError):
    pass


def clean_nickname(value):
    if not isinstance(value, str) or not value.isprintable():
        raise ValueError("Use a nickname with 1 to 40 visible characters.")
    name = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(name) <= 40:
        raise ValueError("Use a nickname with 1 to 40 visible characters.")
    return name


def require_available_nickname(db, nickname, participant=None):
    key = nickname.casefold()
    for row in db.execute("SELECT id,nickname FROM participants WHERE nickname IS NOT NULL"):
        if row["id"] != participant and clean_nickname(row["nickname"]).casefold() == key:
            raise NicknameConflict("That nickname is already in use. Choose another nickname.")


def challenge_board(payload, challenge):
    """Project only one Challenge's scores, ranks and queue counts for the screen."""
    if challenge not in CHALLENGES:
        raise ValueError("Choose Challenge 1, 2, 3 or 4.")
    participants = [
        {"nickname": row["nickname"], "scores": {challenge: row["scores"][challenge]},
         "challenge_ranks": {challenge: row["challenge_ranks"][challenge]}}
        for row in payload["participants"]
    ]
    participants.sort(key=lambda row: (
        row["challenge_ranks"][challenge] if row["challenge_ranks"][challenge] is not None else float("inf"),
        row["nickname"],
    ))
    return {
        "challenge": challenge,
        "participants": participants,
        "queue": payload["queue_by_challenge"].get(challenge, {}),
        "rules": {key: payload["rules"][key] for key in ("status", "rubric", "challenge_max")},
        "fingerprint": payload["fingerprint"],
        "challenges": {challenge: payload["challenges"][challenge]},
    }


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.path = self.directory / "judge.sqlite3"
        if not self.path.is_file():
            raise ValueError("Initialize a new private judge state directory first.")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def initialize(cls, directory, steps=200, device="cpu"):
        settings = rules(steps, device)
        revision = fingerprint(settings)
        directory = Path(directory).expanduser().resolve()
        if directory.is_relative_to(ROOT):
            raise ValueError("Keep private judge state outside the teaching repository.")
        if directory.exists() and directory.stat().st_mode & 0o077:
            raise ValueError("Use a new private directory, or an existing directory accessible only by its owner.")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / "judge.sqlite3"
        # Exclusive creation must never overwrite an existing event.
        with path.open("xb"):
            pass
        path.chmod(0o600)
        instance = cls(directory)
        with instance.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE settings (value TEXT NOT NULL, fingerprint TEXT NOT NULL);
                CREATE TABLE participants (id TEXT PRIMARY KEY, nickname TEXT UNIQUE, token_hash TEXT UNIQUE NOT NULL);
                CREATE TABLE submissions (
                    id TEXT PRIMARY KEY, participant TEXT NOT NULL, challenge TEXT NOT NULL,
                    source_hash TEXT NOT NULL, sources TEXT NOT NULL, status TEXT NOT NULL,
                    created REAL NOT NULL, started REAL, finished REAL, lease TEXT,
                    score REAL, result TEXT, error TEXT, fingerprint TEXT NOT NULL);
                CREATE INDEX queue ON submissions(status, created);
                CREATE INDEX history ON submissions(participant, challenge, created);
            """)
            db.execute("INSERT INTO settings VALUES (?, ?)", (json.dumps(settings), revision))
        return instance

    def settings(self):
        with self.connect() as db:
            row = db.execute("SELECT * FROM settings").fetchone()
        return json.loads(row["value"]), row["fingerprint"]

    def require_current_version(self):
        settings, revision = self.settings()
        if fingerprint(settings) != revision:
            raise ValueError("Judge code or configuration changed. Start a new pilot state; do not mix scoring versions.")
        return settings, revision

    def add_participant(self, nickname=None):
        # The private account is provisioned first; its owner chooses a public name in Jupyter.
        nickname = clean_nickname(nickname) if nickname is not None else None
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if nickname is not None:
                require_available_nickname(db, nickname)
            db.execute("INSERT INTO participants VALUES (?, ?, ?)",
                       (uuid.uuid4().hex, nickname, hashlib.sha256(token.encode()).hexdigest()))
        return token

    def set_nickname(self, participant, nickname):
        nickname = clean_nickname(nickname)
        self.require_current_version()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT id FROM participants WHERE id=?", (participant,)).fetchone():
                raise ValueError("Unknown participant")
            require_available_nickname(db, nickname, participant)
            db.execute("UPDATE participants SET nickname=? WHERE id=?", (nickname, participant))
        return nickname

    def authenticate(self, token):
        if not isinstance(token, str) or not 20 <= len(token) <= 200:
            return None
        with self.connect() as db:
            row = db.execute("SELECT id, nickname FROM participants WHERE token_hash=?",
                             (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        return dict(row) if row else None

    def submit(self, participant, challenge, sources, cooldown=15):
        challenge = str(challenge)
        if challenge not in CHALLENGES or not isinstance(sources, dict) or not sources:
            raise SubmissionError("Choose Challenge 1, 2, 3 or 4 and at least one .py file.")
        if not set(sources) <= set(CHALLENGES[challenge]["files"]):
            raise SubmissionError("Use this Challenge's original .py filenames only.")
        for filename, source in sources.items():
            # Syntax only. Never execute an upload in the web process.
            if challenge == "4":
                from .operators import source_nodes
                source_nodes(source, int(filename.removesuffix(".py").rsplit("_l", 1)[1]))
            else:
                from .contracts import source_nodes
                source_nodes(source, challenge)
        encoded = json.dumps(sources, sort_keys=True)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        _, revision = self.require_current_version()
        now, identifier = time.time(), uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            person = db.execute("SELECT id,nickname FROM participants WHERE id=?", (participant,)).fetchone()
            if not person:
                raise ValueError("Unknown participant")
            if person["nickname"] is None:
                raise NicknameRequired("Register your nickname in the notebook before submitting.")
            duplicate = db.execute("SELECT id FROM submissions WHERE participant=? AND challenge=? AND source_hash=? AND status IN ('queued','running','completed')",
                                   (participant, challenge, digest)).fetchone()
            if duplicate:
                return duplicate["id"]
            if db.execute("SELECT id FROM submissions WHERE participant=? AND status IN ('queued','running')", (participant,)).fetchone():
                raise BusyError("Wait for your current submission to finish before submitting again.")
            recent = db.execute("SELECT MAX(created), COUNT(*) FROM submissions WHERE participant=? AND challenge=?", (participant, challenge)).fetchone()
            if recent[0] is not None and now - recent[0] < cooldown:
                raise BusyError("Please wait 15 seconds between submissions.")
            if recent[1] >= 30:
                raise BusyError("Pilot submission limit reached (30 per Challenge). Contact the instructor.")
            db.execute("INSERT INTO submissions (id,participant,challenge,source_hash,sources,status,created,fingerprint) VALUES (?,?,?,?,?,'queued',?,?)",
                       (identifier, participant, challenge, digest, encoded, now, revision))
        return identifier

    def claim(self):
        settings, revision = self.require_current_version()
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # An interrupted worker cannot leave submissions running forever.
            db.execute("UPDATE submissions SET status='system_error',error='Worker stopped or lease expired; resubmit this code.',finished=? WHERE status='running' AND started<?",
                       (now, now - settings["timeout_seconds"] - 60))
            row = db.execute("SELECT * FROM submissions WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            lease = uuid.uuid4().hex
            db.execute("UPDATE submissions SET status='running',started=?,lease=? WHERE id=?", (now, lease, row["id"]))
        return {**dict(row), "lease": lease, "settings": settings, "sources": json.loads(row["sources"])}

    def finish(self, job, result=None, error=None, status="completed"):
        if status not in {"completed", "system_error", "time_limit"}:
            raise ValueError("Invalid terminal status")
        score = result.get("score") if result is not None else None
        if status == "completed":
            if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100
                    or result.get("challenge") != job["challenge"] or result.get("kind") != "pilot_not_official"
                    or result.get("rubric") != job["settings"]["rubric"]):
                raise ValueError("Invalid trusted evaluation result")
        with self.connect() as db:
            db.execute("UPDATE submissions SET status=?,score=?,result=?,error=?,finished=? WHERE id=? AND status='running' AND lease=?",
                       (status, score, json.dumps(result, allow_nan=False) if result else None, error,
                        time.time(), job["id"], job["lease"]))

    def history(self, participant):
        with self.connect() as db:
            rows = db.execute("SELECT id,challenge,source_hash,status,created,score,result,error FROM submissions WHERE participant=? ORDER BY created DESC LIMIT 100", (participant,)).fetchall()
        return [{**dict(row), "result": json.loads(row["result"]) if row["result"] else None} for row in rows]

    def board(self, challenge=None):
        if challenge is not None and challenge not in CHALLENGES:
            raise ValueError("Choose Challenge 1, 2, 3 or 4.")
        settings, revision = self.settings()
        with self.connect() as db:
            participants = db.execute("SELECT id,nickname FROM participants WHERE nickname IS NOT NULL").fetchall()
            rows = db.execute("SELECT participant,challenge,MAX(score) AS score FROM submissions WHERE status='completed' GROUP BY participant,challenge").fetchall()
            counts = {row[0]: row[1] for row in db.execute("SELECT status,COUNT(*) FROM submissions GROUP BY status")}
            queues = {key: {} for key in CHALLENGES}
            for key, status, count in db.execute("SELECT challenge,status,COUNT(*) FROM submissions GROUP BY challenge,status"):
                queues[key][status] = count
        best = {(row["participant"], row["challenge"]): row["score"] for row in rows}
        board = []
        for person in participants:
            scores = {key: best.get((person["id"], key)) for key in CHALLENGES}
            board.append({"nickname": person["nickname"], "scores": scores,
                          "total": round(sum(v for v in scores.values() if v is not None), 2),
                          "rank": None, "challenge_ranks": {key: None for key in CHALLENGES}})
        board.sort(key=lambda r: (-r["total"], r["nickname"]))
        for key in (None, *CHALLENGES):
            eligible = [r for r in board if (any(v is not None for v in r["scores"].values()) if key is None else r["scores"][key] is not None)]
            value = lambda r: r["total"] if key is None else r["scores"][key]
            eligible.sort(key=lambda r: -value(r))
            previous, rank = None, None
            for index, row in enumerate(eligible, 1):
                if value(row) != previous:
                    rank = index
                if key is None:
                    row["rank"] = rank
                else:
                    row["challenge_ranks"][key] = rank
                previous = value(row)
        payload = {"rules": settings, "fingerprint": revision, "participants": board, "queue": counts,
                   "queue_by_challenge": queues, "challenges": CHALLENGES}
        return challenge_board(payload, challenge) if challenge is not None else payload
