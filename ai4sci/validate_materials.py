"""Static integrity checks for the PhysicsNeMo 2.2.2 course.

Runtime tests are separate: python ai4sci/run_validation.py --suite all ...
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "9cae27f8303268cdaf7528fe963ce12ba439377f"
FENCED = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
OLD_API = re.compile(r"(?:from|import)\s+physicsnemo\.sym\.(?:solver|domain|key|models|geometry|hydra)(?:[.\s]|$)")
EXCLUDE = {".git", ".venv", "__pycache__", ".pytest_cache", "validation-runs", "runs", "outputs"}


def is_active(path):
    return not any(part in EXCLUDE or part.startswith("._") for part in path.relative_to(ROOT).parts)


def markdown_text(path):
    if path.suffix == ".ipynb":
        d = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(source_text(c) for c in d["cells"] if c["cell_type"] == "markdown")
    return path.read_text(encoding="utf-8")


def fragments(path):
    text = FENCED.sub("", markdown_text(path))
    found = set(re.findall(r"\bid=[\"']([^\"']+)[\"']", text))
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.MULTILINE):
        clean = re.sub(r"<[^>]*>", "", heading).strip()
        found.add(clean.replace(" ", "-"))
        clean = re.sub(r"[`*_~]", "", clean)
        slug = re.sub(r"[^\w\- ]", "", clean.lower()).replace(" ", "-")
        number = counts.get(slug, 0)
        found.add(slug + (f"-{number}" if number else ""))
        counts[slug] = number + 1
    return found



def source_text(cell):
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise ValueError("cell source must be a string or list of strings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "static source, full course coverage, notebooks, local links, schedule and assets",
              "physicsnemo": "2.2.2", "upstream_commit": UPSTREAM,
              "python": [], "notebooks": [], "links": [], "assets": [], "errors": [],
              "runtime_training_tested": False}
    def check(condition, message):
        if not condition:
            report["errors"].append(message)
    manifest = json.loads((ROOT / "ai4sci/course_manifest.json").read_text())
    check(manifest["physicsnemo"] == "2.2.2", "manifest version mismatch")
    check(len(manifest["course"]) == 9, "nine original course notebooks required")
    check(len({c["id"] for c in manifest["course"]}) == 9, "duplicate course IDs")
    check(len(manifest["runs"]) == 18, "18 executable lesson modes required")
    for course in manifest["course"]:
        check((ROOT / course["notebook"]).is_file(), "missing course: " + course["notebook"])
    for course, levels in {"wave": 3, "fluid": 3, "climate": 2, "operators": 3}.items():
        actual = sorted(r["level"] for r in manifest["runs"] if r["course"] == course)
        check(actual == list(range(1, levels + 1)), "missing/duplicate challenge level: " + course)
    for run in manifest["runs"]:
        check((ROOT / run["script"]).is_file(), "missing executable: " + run["script"])
    report["coverage"] = {"course_notebooks": len(manifest["course"]), "executable_modes": len(manifest["runs"]), "challenge_levels": 11}
    # Original data and figures are retained byte-for-byte. Code/config changes are authorized.
    tree = subprocess.run(["git", "ls-tree", "-r", UPSTREAM, "--", "tutorial", "challenge"],
                          cwd=ROOT, capture_output=True, text=True)
    if tree.returncode == 0:
        for entry in tree.stdout.splitlines():
            metadata, relative = entry.split("\t", 1)
            if Path(relative).suffix.lower() in {".py", ".yaml", ".yml", ".md", ".ipynb"}:
                continue
            path = ROOT / relative
            expected = metadata.split()[2]
            actual = None
            if path.is_file():
                data = path.read_bytes()
                actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            check(expected == actual, "original asset modified/missing: " + relative)
            report["assets"].append({"path": relative, "upstream_git_blob": expected, "unchanged": expected == actual})
        report["asset_check"] = "upstream Git blob comparison"
    else:
        report["asset_check"] = "unavailable: source snapshot has no upstream Git object"
    language_files = []
    hangul = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and is_active(p)):
        if path.suffix.lower() not in {".py", ".md", ".json", ".ipynb", ".yaml", ".yml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".json", ".ipynb"}:
            text = json.dumps(json.loads(text), ensure_ascii=False)
        relative = str(path.relative_to(ROOT))
        check(not hangul.search(text) and not hangul.search(relative), "English-only course: Hangul found in " + relative)
        language_files.append(relative)
    report["english_only_scan"] = {"files": len(language_files), "scope": "text, decoded JSON and filenames; mathematical symbols are allowed"}
    documents = []
    for path in sorted(p for p in ROOT.rglob("*.py") if is_active(p)):
        relative = str(path.relative_to(ROOT))
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=relative)
            # This validator's regex mentions paths as data, not imports.
            check(not OLD_API.search(source), "removed legacy import: " + relative)
            report["python"].append({"path": relative, "ast": "pass"})
        except (SyntaxError, UnicodeError) as exc:
            check(False, f"{relative}: {exc}")
    transform = TransformerManager().transform_cell
    for path in sorted(p for p in ROOT.rglob("*.ipynb") if is_active(p)):
        relative = str(path.relative_to(ROOT))
        info = {"path": relative, "code_cells_compiled": 0}
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(copy.deepcopy(notebook))
            for number, cell in enumerate(notebook.cells, 1):
                source = cell.source
                location = f"{relative} cell {number}"
                if cell.cell_type == "markdown":
                    documents.append((path, location, source))
                    # Historical migration notes belong in MIGRATION.md, not executable lesson examples.
                    check(not OLD_API.search(source), "removed legacy API in lesson example: " + location)
                elif cell.cell_type == "code":
                    compile(transform(source), location, "exec")
                    check(not OLD_API.search(source), "removed legacy import: " + location)
                    check(not cell.get("outputs"), "clear stale notebook outputs: " + location)
                    info["code_cells_compiled"] += 1
            info["schema"] = "pass"
        except Exception as exc:
            check(False, f"{relative}: {exc}")
        report["notebooks"].append(info)
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".md" and is_active(p)):
        documents.append((path, str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")))
    fragment_cache = {}
    for path, location, source in documents:
        source = FENCED.sub("", source)
        targets = re.findall(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)", source)
        targets += [m[1] for m in re.findall(r"(?:href|src)=([\"'])(.*?)\1", source)]
        for raw_target in targets:
            target = raw_target.strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            resolved = (path.parent / unquote(url.path)).resolve() if url.path else path.resolve()
            exists = resolved.exists()
            inside = resolved.is_relative_to(ROOT)
            check(inside and exists, f"{location}: missing or outside-repo relative link {target}")
            item = {"source": location, "target": target, "exists": exists}
            if url.fragment and exists and resolved.suffix.lower() in {".md", ".ipynb"}:
                if resolved not in fragment_cache:
                    fragment_cache[resolved] = fragments(resolved)
                match = unquote(url.fragment) in fragment_cache[resolved]
                check(match, f"{location}: missing heading fragment {target}")
            report["links"].append(item)
    check("nvidia-physicsnemo[sym]==2.2.2" in (ROOT / "requirements.txt").read_text(), "PhysicsNeMo version must be pinned")
    course = (ROOT / "ai4sci/course-plan.md").read_text(encoding="utf-8")
    active, previous_end = False, None
    schedule = {"minutes": 0, "education": 0, "lunch": 0, "break": 0, "rows": 0, "categories": {}}
    for line in course.splitlines():
        if line.startswith("## "):
            active = bool(re.match(r"## 7-hour", line))
        if not active:
            continue
        row = re.match(r"\|\s*(\d\d):(\d\d)[–—-](\d\d):(\d\d)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|", line)
        if row:
            sh, sm, eh, em, minutes = map(int, row.groups()[:5])
            start, end = sh * 60 + sm, eh * 60 + em
            category = row.group(6).strip()
            check(end - start == minutes and minutes > 0, f"7h schedule: interval/duration mismatch: {line}")
            check(previous_end is None or previous_end == start, f"7h schedule: gap/overlap before {line}")
            previous_end = end
            schedule["minutes"] += minutes
            schedule["rows"] += 1
            schedule["categories"][category] = schedule["categories"].get(category, 0) + minutes
            schedule[{"Lunch":"lunch", "Break":"break"}.get(category, "education")] += minutes
        if "**Total**" in line:
            total = re.search(r"\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*", line)
            check(total is not None and int(total.group(1)) == schedule["minutes"], "7h summary total mismatch")
            for category, field in (("Teaching", "education"), ("Lunch", "lunch"), ("Break", "break")):
                declared = re.search(category + r"\s+(\d+)", line)
                check(declared is not None and int(declared.group(1)) == schedule[field], f"7h summary category mismatch: {category}")
    check(schedule["minutes"] == 420, "Shared 7h schedule must total 420 minutes")
    check((schedule["education"], schedule["lunch"], schedule["break"]) == (320, 60, 40), "Shared 7h schedule must contain education 320, lunch 60, break 40 minutes")
    report["schedules"] = {"7": schedule, "6": {"status": "pending_time_allocation", "arithmetic_validation": "not_applicable_no_schedule_proposed"}}

    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workdir = re.search(r"^WORKDIR\s+(\S+)\s*$", docker, re.MULTILINE)
    copy_match = re.search(r"^COPY\s+\.\s+(\S+)\s*$", docker, re.MULTILINE)
    check(workdir is not None and copy_match is not None and workdir.group(1).rstrip("/") == copy_match.group(1).rstrip("/"), "Docker COPY . destination must match WORKDIR")
    command = next((line[4:].strip() for line in docker.splitlines() if line.startswith("CMD ")), "[]")
    try:
        command = json.loads(command)
        landing = next((arg.split("=/lab/tree/", 1)[1] for arg in command if arg.startswith("--LabApp.default_url=/lab/tree/")), None)
        check(landing is not None and (ROOT / landing).is_file(), "Docker landing document missing")
        report["docker"] = {"copy_to_workdir": bool(workdir and copy_match and workdir.group(1).rstrip("/") == copy_match.group(1).rstrip("/")), "landing": landing, "runtime_tested": False}
    except (ValueError, TypeError):
        check(False, "Docker CMD must be a JSON command array")

    report["status"] = "pass" if not report["errors"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "python": len(report["python"]), "notebooks": len(report["notebooks"]), "links": len(report["links"]), "coverage": report["coverage"], "errors": report["errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
