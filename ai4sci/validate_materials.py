"""Validate the authored AI4Sci route without training or installing packages.

Run with a Python environment containing NumPy for the CPU reference tests:
    python ai4sci/validate_materials.py --output /tmp/ai4sci-validation.json

Original tutorial/challenge content is linked but is outside the authored-code
checks. A successful run does not certify the GPU/container environment.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


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
        "scope": "authored ai4sci route plus root entry/deployment documents",
        "excluded": "original tutorial/challenge exercise internals; GPU training; container build; external links",
        "python": [], "notebooks": [], "links": [], "schedules": {},
        "errors": [], "limitations": [],
    }

    def check(condition, message):
        if not condition:
            report["errors"].append(message)

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

    for path in sorted((ROOT / "ai4sci").rglob("*.ipynb")):
        relative = str(path.relative_to(ROOT))
        info = {"path": relative, "code_cells_compiled": 0, "magic_lines_skipped": 0}
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            check(notebook.get("nbformat") == 4, f"{relative}: nbformat must be 4")
            check(isinstance(notebook.get("nbformat_minor"), int), f"{relative}: missing nbformat_minor")
            check(isinstance(notebook.get("metadata"), dict), f"{relative}: invalid metadata")
            cells = notebook.get("cells")
            if not isinstance(cells, list):
                raise ValueError("cells must be a list")
            ids = set()
            for number, cell in enumerate(cells, 1):
                location = f"{relative} cell {number}"
                identifier = cell.get("id", "")
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

    markdown_paths = sorted((ROOT / "ai4sci").rglob("*.md")) + [ROOT / "README.md", ROOT / "Deployment_Guide.MD"]
    for path in markdown_paths:
        documents.append((path, str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")))
    for path, location, source in documents:
        source = re.sub(r"^```.*?^```\s*$", "", source, flags=re.MULTILINE | re.DOTALL)
        targets = re.findall(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)", source)
        targets += [match[1] for match in re.findall(r"(?:href|src)=([\"'])(.*?)\1", source)]
        for raw_target in targets:
            target = raw_target.strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            resolved = (path.parent / unquote(url.path)).resolve()
            exists = resolved.exists()
            inside = resolved.is_relative_to(ROOT)
            check(inside and exists, f"{location}: missing or outside-repo relative link {target}")
            report["links"].append({"source": location, "target": target, "exists": exists, "inside_repo": inside})

    course = (ROOT / "ai4sci/course-plan.md").read_text(encoding="utf-8")
    active, previous_end = None, None
    for line in course.splitlines():
        heading = re.match(r"## ([67])시간안", line)
        if heading:
            active, previous_end = heading[1], None
            report["schedules"][active] = {"minutes": 0, "categories": {}, "rows": 0}
        elif line.startswith("## "):
            active = None
        if not active:
            continue
        row = re.match(r"\|\s*(\d\d):(\d\d)[–—-](\d\d):(\d\d)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|", line)
        if row:
            sh, sm, eh, em, minutes = map(int, row.groups()[:5])
            start, end = sh * 60 + sm, eh * 60 + em
            category = row.group(6).strip()
            check(end - start == minutes and minutes > 0, f"{active}h schedule: interval/duration mismatch: {line}")
            check(previous_end is None or previous_end == start, f"{active}h schedule: gap/overlap before {line}")
            previous_end = end
            schedule = report["schedules"][active]
            schedule["minutes"] += minutes
            schedule["rows"] += 1
            schedule["categories"][category] = schedule["categories"].get(category, 0) + minutes
        if "**합계**" in line:
            total = re.search(r"\*\*합계\*\*\s*\|\s*\*\*(\d+)\*\*", line)
            check(total is not None and int(total.group(1)) == report["schedules"][active]["minutes"], f"{active}h summary total mismatch")
            for category in ("교육", "점심", "휴식"):
                declared = re.search(category + r"\s+(\d+)", line)
                check(declared is not None and int(declared.group(1)) == report["schedules"][active]["categories"].get(category), f"{active}h summary category mismatch: {category}")
    for hours in ("6", "7"):
        schedule = report["schedules"].get(hours, {})
        check(schedule.get("minutes") == int(hours) * 60, f"{hours}h schedule total must be {int(hours) * 60}")
    participant_guide = (ROOT / "ai4sci/README.md").read_text(encoding="utf-8")
    participant_minutes = sum(map(int, re.findall(r"^\|\s*\d+\.[^|]+\|\s*(\d+)분", participant_guide, flags=re.MULTILINE)))
    report["participant_guide_education_minutes"] = participant_minutes
    check(participant_minutes == report["schedules"].get("7", {}).get("categories", {}).get("교육"), "participant guide duration differs from 7h teaching total")

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
        report["preflight"] = {"exit_code": preflight.returncode, "stdout": preflight.stdout, "stderr": preflight.stderr}
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
        report["reference_tests"] = {"exit_code": reference.returncode, "stdout": reference.stdout, "stderr": reference.stderr}
        check(reference.returncode == 0, "CPU wave reference tests failed")

    report["status"] = "pass" if not report["errors"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "python": len(report["python"]), "notebooks": len(report["notebooks"]), "links": len(report["links"]), "schedules": report["schedules"], "errors": report["errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
