"""Collect saved exercise functions and submit from the teaching notebook."""
import ast
from html import escape
import json
import os
from pathlib import Path
from uuid import uuid4

from ETC.judge.catalog import CHALLENGES
from ETC.judge.expressions import SubmissionError
from ETC.judge.contracts import function_names, source_nodes as problem_source_nodes
from .judge_client import validate_url


def submission_html(challenge, reference, judge_url=""):
    spec = CHALLENGES[str(challenge)]
    exercises = ("build_datasets, build_model and PINO's ReactionDiffusionPDE" if str(challenge) == "4"
                 else ", ".join(function_names(challenge)))
    mode = ("Instructor demonstration: submission is disabled. Switch USE_REFERENCE to False and rerun the submission cell."
            if reference else "Student practice: graphs and local errors are feedback, not a submitted score.")
    link = "If the judge connection is not configured, you can still practice. The submission controls below check the private workspace configuration."
    if judge_url:
        validate_url(judge_url)
        link = f'Judge API: <code>{escape(judge_url)}</code>. Submit and view your results here in the notebook.'
    return (f'<section><h3>Challenge {escape(str(challenge))}: submission</h3><p><strong>{mode}</strong></p>'
            f'<p>Files: {escape(", ".join(spec["files"]))}</p>'
            f'<p>Register your nickname below before your first submission. The same name appears in all four Challenges and on the scoreboard. Save your {exercises} implementation, choose the Levels, then click Submit code. '
            'Running a cell does not submit. The server recalculates results; it does not read your local metrics.json.</p>'
            '<p>Current server scores are a pilot, not official event points. All four Challenges are included (400 points total).</p>'
            f'<p>{link}</p></section>')


def show_submission_panel(challenge, *, reference, judge_url=""):
    from IPython.display import HTML, display
    display(HTML(submission_html(challenge, reference, judge_url)))


def collect_submission(challenge, lesson_dir, *, levels=(1,), reference=False):
    if type(reference) is not bool or reference:
        raise SubmissionError("Reference demonstrations cannot be submitted. Use student mode and save your equations.")
    challenge = str(challenge)
    if challenge not in CHALLENGES or not levels:
        raise SubmissionError("Choose a supported Challenge and at least one Level.")
    filenames = CHALLENGES[challenge]["files"]
    sources = {}
    for level in levels:
        if type(level) is not int or not 1 <= level <= len(filenames):
            raise SubmissionError("Invalid Level number.")
        name = filenames[level - 1]
        source = (Path(lesson_dir) / name).read_text(encoding="utf-8")
        if challenge == "4":
            from ETC.judge.operators import source_nodes
            nodes = source_nodes(source, level)
        else:
            nodes = problem_source_nodes(source, challenge)
        # Validation guards may raise ValueError; unfinished exercise markers
        # must never be mistaken for a completed implementation.
        for function in nodes:
            for node in ast.walk(function):
                unfinished = (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                              and isinstance(node.exc.func, ast.Name)
                              and node.exc.func.id in {"NotImplementedError", "UnfinishedExerciseError"})
                if isinstance(node, ast.Pass) or unfinished:
                    raise SubmissionError(f"{name}: finish the required exercise functions before submitting this Level.")
        sources[name] = "\n\n".join(ast.get_source_segment(source, node) for node in nodes) + "\n"
    return {"challenge": challenge, "sources": sources}


def export_submission(challenge, lesson_dir, *, levels=(1,), reference=False):
    """Optional instructor/test backup, not the student submission path."""
    payload = collect_submission(challenge, lesson_dir, levels=levels, reference=reference)
    destination = Path(lesson_dir) / "outputs" / "submissions"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"challenge-{challenge}-{uuid4().hex}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return output


def show_export(path):
    from IPython.display import FileLink, display
    print("Local backup only. Use Submit code in the notebook to request evaluation.")
    display(FileLink(os.path.relpath(path, Path.cwd())))


def show_submission_controls(challenge, lesson_dir, *, levels=(1,), reference=False, judge_url=""):
    from IPython.display import display
    from .submission_widgets import SubmissionPanel
    key = (str(challenge), str(Path(lesson_dir).resolve()))
    old = _panels.pop(key, None)
    if old is not None:
        old.close()
    panel = SubmissionPanel(challenge, lesson_dir, levels=levels, reference=reference, judge_url=judge_url)
    _panels[key] = panel
    display(panel.widget)


_panels = {}
