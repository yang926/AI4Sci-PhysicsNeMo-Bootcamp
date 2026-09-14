"""Validate the complete original course and AI4Sci editorial changes.

Run with a Python environment containing NumPy for the CPU reference tests:
    python ai4sci/validate_materials.py --output /tmp/ai4sci-validation.json

Compare original content with upstream commit 9cae27f; validate all notebooks
and repository Markdown. Exercise completion and GPU/container execution are
not performed. Supplementary wave CPU checks are reported separately.
"""

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "9cae27f8303268cdaf7528fe963ce12ba439377f"
FENCED = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$(?!\$).*?(?<!\\)\$|\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL)
GENERATED = {
    "tutorial/projectile/outputs/projectile/constraints/IC.vtp",
    "tutorial/projectile/outputs/projectile/constraints/interior.vtp",
    "tutorial/projectile/outputs/projectile/inferencers/inferencer_data.vtp",
    "tutorial/projectile/outputs/projectile/validators/validator.vtp",
}


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, capture_output=True).stdout


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
    report = {
        "scope": "nine original content notebooks, Start_Here, two AI4Sci notebooks and all repository Markdown",
        "upstream_commit": UPSTREAM,
        "preservation": {"files": [], "notebooks": []},
        "excluded": "original challenge .py execution and exercise completion; GPU training; container build; external links and downloads",
        "python": [], "notebooks": [], "links": [], "schedules": {},
        "errors": [], "limitations": [],
    }

    def check(condition, message):
        if not condition:
            report["errors"].append(message)

    originals = []
    protected = []
    for entry in git("ls-tree", "-r", UPSTREAM, "--", "tutorial", "challenge").decode().splitlines():
        metadata, relative = entry.split("\t", 1)
        if relative.endswith(".ipynb"):
            originals.append(ROOT / relative)
        elif relative != "tutorial/readme.md":
            protected.append((relative, metadata.split()[2]))
    check(len(originals) == 9, "Expected nine original content notebooks")
    for relative, expected in protected:
        actual = git("hash-object", "--no-filters", "--", relative).decode().strip() if (ROOT / relative).is_file() else None
        equal = expected == actual
        check(equal, f"Original file bytes changed or missing: {relative}")
        report["preservation"]["files"].append({"path": relative, "upstream_git_blob": expected, "current_git_blob": actual, "bytes_equal": equal})
    report["preservation"]["protected_file_count"] = len(protected)
    def extract(d, pattern):
        return [v for c in d["cells"] if c["cell_type"] == "markdown" for v in pattern.findall(source_text(c))]
    for path in originals:
        relative = str(path.relative_to(ROOT))
        base = json.loads(git("show", f"{UPSTREAM}:{relative}"))
        current = json.loads(path.read_text(encoding="utf-8"))
        old = base["cells"]
        new = [c for c in current["cells"] if "ai4sci-navigation" not in c.get("metadata", {}).get("tags", [])]
        findings = {
            "code_cells_outputs_and_metadata_equal": [c for c in old if c["cell_type"] != "markdown"] == [c for c in new if c["cell_type"] != "markdown"],
            "existing_cell_types_and_metadata_equal": [(c["cell_type"], c.get("metadata")) for c in old] == [(c["cell_type"], c.get("metadata")) for c in new],
            "notebook_metadata_equal": {k:v for k,v in base.items() if k != "cells"} == {k:v for k,v in current.items() if k != "cells"},
            "fenced_code_examples_equal": extract(base, FENCED) == extract(current, FENCED),
            "all_math_equal": extract(base, MATH) == extract(current, MATH),
        }
        for key, equal in findings.items():
            check(equal, f"{relative}: preservation check failed: {key}")
        report["preservation"]["notebooks"].append({"path": relative, **findings})

    documents = []
    for path in sorted((ROOT / "ai4sci").rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            report["python"].append({"path": relative, "ast": "pass"})
        except (SyntaxError, UnicodeError) as exc:
            check(False, f"{relative}: {exc}")

    try:
        from IPython.core.inputtransformer2 import TransformerManager
        transform = TransformerManager().transform_cell
    except ImportError:
        transform = None
        report["limitations"].append(
            "IPython unavailable: standalone !/% lines are replaced by pass for Python syntax checks; shell/magic semantics are not validated."
        )

    try:
        import nbformat
    except ImportError:
        nbformat = None
        report["limitations"].append("nbformat unavailable: notebook JSON structure, required cell fields and Python syntax are checked directly; full nbformat schema validation is not performed.")
    notebook_paths = originals + [ROOT / "Start_Here.ipynb"] + sorted((ROOT / "ai4sci").rglob("*.ipynb"))
    check(len(notebook_paths) == 12, "Expected 12 notebooks")
    for path in notebook_paths:
        relative = str(path.relative_to(ROOT))
        info = {"path": relative, "code_cells_compiled": 0, "magic_lines_skipped": 0}
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if nbformat is not None:
                nbformat.validate(copy.deepcopy(notebook))
            info["schema"] = "nbformat pass" if nbformat is not None else "direct required-field checks"
            check(notebook.get("nbformat") == 4, f"{relative}: nbformat must be 4")
            check(isinstance(notebook.get("nbformat_minor"), int), f"{relative}: missing nbformat_minor")
            check(isinstance(notebook.get("metadata"), dict), f"{relative}: invalid metadata")
            cells = notebook.get("cells")
            if not isinstance(cells, list):
                raise ValueError("cells must be a list")
            ids = set()
            for number, cell in enumerate(cells, 1):
                location = f"{relative} cell {number}"
                identifier = cell.get("id")
                if identifier is not None or notebook["nbformat_minor"] >= 5:
                    check(isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier), f"{location}: invalid cell id")
                    check(identifier not in ids, f"{location}: duplicate cell id")
                    ids.add(identifier)
                kind = cell.get("cell_type")
                check(kind in {"code", "markdown", "raw"}, f"{location}: invalid cell type")
                check(isinstance(cell.get("metadata"), dict), f"{location}: invalid metadata")
                source = source_text(cell)
                if kind == "markdown":
                    documents.append((path, location, source))
                elif kind == "code":
                    check(isinstance(cell.get("outputs"), list), f"{location}: invalid outputs")
                    check("execution_count" in cell and (cell["execution_count"] is None or isinstance(cell["execution_count"], int)), f"{location}: invalid execution_count")
                    if transform:
                        compiled_source = transform(source)
                    else:
                        lines = []
                        for line in source.splitlines():
                            if line.lstrip().startswith("%%"):
                                raise ValueError(f"{location}: cell magic requires IPython to validate")
                            if line.lstrip().startswith(("!", "%")):
                                line = line[:len(line) - len(line.lstrip())] + "pass # IPython line omitted"
                                info["magic_lines_skipped"] += 1
                            lines.append(line)
                        compiled_source = "\n".join(lines)
                    compile(compiled_source, location, "exec")
                    info["code_cells_compiled"] += 1
            info["cells"] = len(cells)
        except (ValueError, TypeError, SyntaxError, KeyError, AttributeError) as exc:
            check(False, f"{relative}: {exc}")
        report["notebooks"].append(info)

    markdown_paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".md" and ".git" not in p.parts)
    for path in markdown_paths:
        documents.append((path, str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")))
    fragment_cache = {}
    for path, location, source in documents:
        source = FENCED.sub("", source)
        targets = re.findall(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)", source)
        targets += [match[1] for match in re.findall(r"(?:href|src)=([\"'])(.*?)\1", source)]
        for raw_target in targets:
            target = raw_target.strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            resolved = (path.parent / unquote(url.path)).resolve() if url.path else path.resolve()
            exists = resolved.exists()
            inside = resolved.is_relative_to(ROOT)
            relative = str(resolved.relative_to(ROOT)) if inside else str(resolved)
            generated = relative in GENERATED
            check(inside and (exists or generated), f"{location}: missing or outside-repo relative link {target}")
            item = {"source": location, "target": target, "exists": exists, "inside_repo": inside, "generated_after_training": generated}
            if generated:
                check("output links below become available after training" in markdown_text(path), f"{location}: generated output requires after-training explanation")
            if url.fragment and exists and resolved.suffix.lower() in {".md", ".ipynb"}:
                if resolved not in fragment_cache:
                    fragment_cache[resolved] = fragments(resolved)
                match = unquote(url.fragment) in fragment_cache[resolved]
                item["fragment_exists"] = match
                check(match, f"{location}: missing heading fragment {target}")
            report["links"].append(item)

    course = (ROOT / "ai4sci/course-plan.md").read_text(encoding="utf-8")
    active, previous_end = False, None
    schedule = {"minutes": 0, "education": 0, "lunch": 0, "break": 0, "rows": 0, "categories": {}}
    for line in course.splitlines():
        if line.startswith("## "):
            active = bool(re.match(r"## 7시간", line))
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
            schedule[{"점심":"lunch", "휴식":"break"}.get(category, "education")] += minutes
        if "**합계**" in line:
            total = re.search(r"\*\*합계\*\*\s*\|\s*\*\*(\d+)\*\*", line)
            check(total is not None and int(total.group(1)) == schedule["minutes"], "7h summary total mismatch")
            for category, field in (("교육", "education"), ("점심", "lunch"), ("휴식", "break")):
                declared = re.search(category + r"\s+(\d+)", line)
                check(declared is not None and int(declared.group(1)) == schedule[field], f"7h summary category mismatch: {category}")
    check(schedule["minutes"] == 420, "Shared 7h schedule must total 420 minutes")
    check((schedule["education"], schedule["lunch"], schedule["break"]) == (320, 60, 40), "Shared 7h schedule must contain education 320, lunch 60, break 40 minutes")
    report["schedules"] = {"7": schedule, "6": {"status": "pending_time_allocation", "arithmetic_validation": "not_applicable_no_schedule_proposed"}}

    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workdir = re.search(r"^WORKDIR\s+(\S+)\s*$", docker, re.MULTILINE)
    copy = re.search(r"^COPY\s+\.\s+(\S+)\s*$", docker, re.MULTILINE)
    check(workdir is not None and copy is not None and workdir.group(1).rstrip("/") == copy.group(1).rstrip("/"), "Docker COPY . destination must match WORKDIR")
    command = next((line[4:].strip() for line in docker.splitlines() if line.startswith("CMD ")), "[]")
    try:
        command = json.loads(command)
        landing = next((arg.split("=/lab/tree/", 1)[1] for arg in command if arg.startswith("--LabApp.default_url=/lab/tree/")), None)
        check(landing is not None and (ROOT / landing).is_file(), "Docker landing document missing")
        report["docker"] = {"copy_to_workdir": bool(workdir and copy and workdir.group(1).rstrip("/") == copy.group(1).rstrip("/")), "landing": landing, "runtime_tested": False}
    except (ValueError, TypeError):
        check(False, "Docker CMD must be a JSON command array")

    with tempfile.TemporaryDirectory(prefix="ai4sci-validate-") as temporary:
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": temporary}
        preflight_code = '''import json, pathlib, sys
notebook = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
namespace = {}
for number, cell in enumerate(notebook["cells"], 1):
    if cell["cell_type"] == "code":
        source = cell["source"]
        exec(compile("".join(source) if isinstance(source, list) else source, f"preflight cell {number}", "exec"), namespace)
state = {key: namespace.get(key) for key in ("CHECK_EVENT_GPU", "missing_files", "package_issues", "interface_issues", "cuda_available", "environment_issues")}
print("PREFLIGHT_STATE=" + json.dumps(state))
'''
        preflight = subprocess.run([sys.executable, "-c", preflight_code, str(ROOT / "ai4sci/00_환경확인.ipynb")], cwd=ROOT / "ai4sci", env=environment, text=True, capture_output=True, timeout=60)
        report["preflight"] = {"scope": "Read-only local environment/file checks; no GPU training", "exit_code": preflight.returncode, "stdout": preflight.stdout, "stderr": preflight.stderr}
        check(preflight.returncode == 0, "preflight notebook execution failed")
        states = [line.split("=", 1)[1] for line in preflight.stdout.splitlines() if line.startswith("PREFLIGHT_STATE=")]
        if states:
            state = json.loads(states[-1])
            report["preflight"]["state"] = state
            check(state.get("missing_files") == [], "preflight could not locate required files")
            report["preflight"]["event_gpu_basic_checks_passed"] = bool(state.get("CHECK_EVENT_GPU") and state.get("cuda_available") and not state.get("environment_issues"))
        else:
            check(False, "preflight did not report its final state")
        reference = subprocess.run([sys.executable, "test_reference.py"], cwd=ROOT / "ai4sci/wave", env=environment, text=True, capture_output=True, timeout=60)
        report["supplementary_wave_cpu_checks"] = {"scope": "Added ai4sci/wave reference only; not original Wave Challenge or GPU validation", "exit_code": reference.returncode, "stdout": reference.stdout, "stderr": reference.stderr}
        check(reference.returncode == 0, "Supplementary wave CPU reference checks failed")

    report["status"] = "pass" if not report["errors"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "python": len(report["python"]), "notebooks": len(report["notebooks"]), "links": len(report["links"]), "schedules": report["schedules"], "errors": report["errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
