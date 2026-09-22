"""Keep the learner's entry points aligned with the executable course."""

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ETC.course_materials.validate_materials import FENCED, fragments, markdown_text, source_text


FIRST_LAB = "01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb"
CONCEPT_INTRO = "01_Introduction.ipynb"
COURSE_PATHS = (
    ("intro", CONCEPT_INTRO),
    ("lab1", FIRST_LAB),
    ("lab2", "01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb"),
    ("lab3", "01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb"),
    ("lab4", "01_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb"),
    ("wave", "02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb"),
    ("fluid", "02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb"),
    ("climate", "02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb"),
    ("operators", "02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb"),
)
ENTRY_POINTS = (
    "Start_Here.ipynb",
    "ETC/course_materials/README.md",
    "00_Setup.ipynb",
)


def notebook(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def start_here_cell(cell_id):
    return source_text(next(cell for cell in notebook("Start_Here.ipynb")["cells"] if cell["id"] == cell_id))


def link_targets(source):
    """Read inline Markdown links/images and HTML links outside code fences."""
    source = FENCED.sub("", source)
    targets = re.findall(r"\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)", source)
    targets += [match[1] for match in re.findall(r"(?:href|src)=([\"'])(.*?)\1", source)]
    return [target.strip("<>") for target in targets]


def local_links(relative):
    document = ROOT / relative
    for target in link_targets(markdown_text(document)):
        url = urlsplit(target)
        if not url.scheme and not url.netloc:
            resolved = (document.parent / unquote(url.path)).resolve() if url.path else document.resolve()
            yield target, resolved, unquote(url.fragment)


def test_start_here_course_overview_matches_manifest_order():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text(encoding="utf-8"))
    expected = [lesson["notebook"] for lesson in manifest["course"]]
    assert len(expected) == 9
    listed = [unquote(urlsplit(target).path) for target in link_targets(start_here_cell("603bdfa8"))]
    assert [path for path in listed if path in expected] == expected


def test_manifest_preserves_numbered_lesson_layout_and_training_paths():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text(encoding="utf-8"))
    assert [(lesson["id"], lesson["notebook"]) for lesson in manifest["course"]] == list(COURSE_PATHS)
    for _, path in COURSE_PATHS:
        assert (ROOT / path).is_file(), path
    lesson_folders = {identifier: Path(path).parent for identifier, path in COURSE_PATHS}
    for run in manifest["runs"]:
        script = ROOT / run["script"]
        assert script.is_file(), run["script"]
        assert script.is_relative_to(ROOT / lesson_folders[run["course"]])
    for chapter in ("01_labs", "02_challenges"):
        folders = sorted(path.name for path in (ROOT / chapter).iterdir()
                         if path.is_dir() and re.match(r"\d{2}_", path.name))
        assert [name.split("_", 1)[0] for name in folders] == ["01", "02", "03", "04"]


def test_first_problem_link_belongs_to_course_overview_not_header():
    header = start_here_cell("cd4f1533")
    header_lines = [line for line in header.splitlines() if line.strip()]
    assert header_lines[0].startswith("# ")
    assert not any(urlsplit(target).path.endswith(".ipynb") for target in link_targets(header)), (
        "Lesson navigation belongs in the Complete course table; header links identify the event"
    )
    overview_links = [urlsplit(target) for target in link_targets(start_here_cell("603bdfa8"))]
    first_problem_links = [url for url in overview_links if unquote(url.path) == FIRST_LAB]
    assert len(first_problem_links) == 1
    assert unquote(first_problem_links[0].fragment) == "first-problem"


def test_start_here_identifies_the_event_and_uses_local_original_logos():
    import hashlib

    header = start_here_cell("cd4f1533")
    for text in ("NVIDIA PhysicsNeMo Tutorial", "AI4Science Korea 2026", "30 September 2026",
                 "Seoul Dragon City", "Mingyu Yang", "Hyungon Ryu", "Open Hackathons"):
        assert text in header
    assert "https://ai4scikorea.org/" in link_targets(header)
    expected = {
        "ETC/assets/branding/nvidia-logo-horz.svg":
            "4325ea0c4078941bce84b2b1c6f045c3daf36de44cfc4c89ac5c49186f4874e8",
        "ETC/assets/branding/open-hackathons.jpg":
            "634cf9fdd5c6fd5861fd5b08a7b2761282436ffda496d1134a7762cf34ce171d",
    }
    images = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"[^>]*\balt="([^"]+)"', header)
    assert {src for src, _ in images} == set(expected)
    assert all(alt for _, alt in images)
    for path, digest in expected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("relative", ["Start_Here.ipynb", "README.md", "ETC/course_materials/course-plan.md"])
def test_course_entry_points_do_not_link_to_private_planning_documents(relative):
    source = markdown_text(ROOT / relative)
    assert not any(urlsplit(target).hostname in {"docs.google.com", "drive.google.com"}
                   for target in link_targets(source))


def test_complete_course_combines_schedule_with_notebook_navigation():
    agenda = start_here_cell("603bdfa8")
    assert "| Time (KST) | Session / Topic | Notebook | Instructor |" in agenda
    schedule = (ROOT / "ETC/course_materials/course-plan.md").read_text(encoding="utf-8")
    schedule = schedule.split("## 8-hour", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in agenda.splitlines() if re.match(r"\| \d{2}:\d{2}", line)]
    assert len(rows) == 12
    for line in rows:
        time, topic, links, lecturer = [field.strip() for field in line.split("|")[1:5]]
        row = next(row for row in schedule.splitlines() if row.startswith(f"| {time} |"))
        assert topic in row
        assert lecturer in row
        if "Break" in topic or "Q&A" in topic:
            assert not link_targets(links)
        else:
            assert link_targets(links)
    labs = next(row for row in rows if row.startswith("| 10:30–11:30 |"))
    assert [urlsplit(target).path for target in link_targets(labs)] == [
        path for identifier, path in COURSE_PATHS if identifier.startswith("lab")
    ]


def test_start_here_exports_one_visible_course_table():
    from nbconvert import HTMLExporter

    body, _ = HTMLExporter().from_filename(str(ROOT / "Start_Here.ipynb"))
    tables = re.findall(r"<table>(.*?)</table>", body, flags=re.DOTALL)
    assert len(tables) == 1
    assert "<details>" not in body
    assert len(re.findall(r"<td>\d{2}:\d{2}", tables[0])) == 12
    assert len(re.findall(r'href="[^"#]+\.ipynb(?:#[^"]*)?"', tables[0])) == 10


def test_lesson_filenames_identify_number_and_topic_in_jupyter_tabs():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text())
    filenames = []
    for folder, prefix in (("01_labs", "Lab"), ("02_challenges", "Challenge")):
        lessons = [Path(course["notebook"]) for course in manifest["course"]
                   if Path(course["notebook"]).parts[0] == folder]
        assert len(lessons) == 4
        for number, lesson in enumerate(lessons, start=1):
            assert re.fullmatch(rf"{prefix}_{number}_[A-Za-z]+(?:_[A-Za-z]+)*\.ipynb", lesson.name), lesson
            assert (ROOT / lesson).is_file()
            assert not (ROOT / lesson.parent / f"{prefix}.ipynb").exists()
            filenames.append(lesson.name)
    assert len(set(filenames)) == 8


def test_course_overview_starts_with_environment_check_and_lists_each_lesson_once():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text(encoding="utf-8"))
    expected = ["00_Setup.ipynb"] + [lesson["notebook"] for lesson in manifest["course"]]
    paths = [unquote(urlsplit(target).path) for target in link_targets(start_here_cell("603bdfa8"))]
    assert [path for path in paths if path.endswith(".ipynb")] == expected


def test_folder_guide_covers_lesson_and_support_directories():
    guide = start_here_cell("5ac39fa0")
    for folder in ("01_labs/", "02_challenges/", "ETC/"):
        assert folder in guide, folder
        assert (ROOT / folder).is_dir()
    assert "ETC/legacy/tutorial/" not in guide
    assert "ETC/legacy/challenge/" not in guide


def test_first_lab_has_an_unambiguous_title():
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text(encoding="utf-8"))
    first_lab = next(lesson for lesson in manifest["course"] if lesson["id"] == "lab1")
    assert first_lab["notebook"] == FIRST_LAB
    title = next(line for line in markdown_text(ROOT / FIRST_LAB).splitlines() if line.startswith("# "))
    assert "Lab 1" in title and "PINN" in title


@pytest.mark.parametrize("anchor", ["first-problem", "forward-pinn-execution"])
def test_first_lab_has_stable_explicit_anchors(anchor):
    source = markdown_text(ROOT / FIRST_LAB)
    assert re.search(r"\bid=[\"']" + re.escape(anchor) + r"[\"']", source)
    assert anchor in fragments(ROOT / FIRST_LAB)


@pytest.mark.parametrize("relative", ENTRY_POINTS)
def test_entry_points_link_directly_to_the_first_problem(relative):
    links = {(resolved, fragment) for _, resolved, fragment in local_links(relative)}
    assert ((ROOT / FIRST_LAB).resolve(), "first-problem") in links


@pytest.mark.parametrize("relative", (*ENTRY_POINTS, CONCEPT_INTRO, FIRST_LAB))
def test_local_navigation_targets_and_fragments_exist(relative):
    links = list(local_links(relative))
    assert links, f"No local navigation links in {relative}"
    for target, resolved, fragment in links:
        assert resolved.is_relative_to(ROOT), f"{relative}: outside-repository link {target}"
        assert resolved.exists(), f"{relative}: missing link target {target}"
        if fragment and resolved.suffix.lower() in {".md", ".ipynb"}:
            assert fragment in fragments(resolved), f"{relative}: missing fragment {target}"


def test_concept_introduction_is_reading_only():
    cells = notebook(CONCEPT_INTRO)["cells"]
    assert cells
    assert all(cell["cell_type"] == "markdown" for cell in cells)


def test_course_prose_does_not_use_em_dashes():
    paths = {ROOT / path for _, path in COURSE_PATHS}
    paths.update(ROOT / path for path in ("README.md", "Start_Here.ipynb", "00_Setup.ipynb"))
    for folder in ("ETC/course_materials", "ETC/environment", "01_labs", "02_challenges"):
        paths.update((ROOT / folder).rglob("*.md"))
    paths.add(ROOT / "ETC/course_materials/01_Wave_PINN.ipynb")
    violations = []
    for path in sorted(paths):
        for line_number, line in enumerate(markdown_text(path).splitlines(), start=1):
            if "\u2014" in line or "&mdash;" in line or "&#8212;" in line:
                violations.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert not violations, "Use a sentence break, colon or parentheses instead: " + ", ".join(violations)


def test_active_course_materials_do_not_advertise_unimplemented_topics():
    """Keep source-history comparisons in MIGRATION.md, outside the teaching path."""
    paths = {ROOT / relative for relative in (
        "README.md", "Start_Here.ipynb", "00_Setup.ipynb", CONCEPT_INTRO,
        "ETC/course_materials/README.md", "ETC/course_materials/INSTRUCTOR.md",
        "ETC/course_materials/course-plan.md",
    )}
    for chapter in ("01_labs", "02_challenges"):
        paths.update(path for path in (ROOT / chapter).rglob("*")
                     if path.suffix in {".ipynb", ".py"}
                     and ".ipynb_checkpoints" not in path.parts)
    unsupported = re.compile(
        r"\b(?:fourcastnet|forecastnet|darcy|magnetohydrodynamics|mhd)\b", re.IGNORECASE
    )
    violations = []
    for path in sorted(paths):
        if path.suffix == ".ipynb":
            cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
            source = "\n".join(source_text(cell) for cell in cells)
        else:
            source = path.read_text(encoding="utf-8")
        matches = sorted(set(unsupported.findall(source)))
        if matches:
            violations.append(f"{path.relative_to(ROOT)}: {', '.join(matches)}")
    assert not violations, "Unimplemented topics in active course material:\n" + "\n".join(violations)


def test_repository_readme_has_one_course_start_and_optional_installation_link():
    source = markdown_text(ROOT / "README.md")
    links = list(local_links("README.md"))
    notebook_links = [(resolved, fragment) for _, resolved, fragment in links
                      if resolved.suffix == ".ipynb"]
    assert notebook_links == [((ROOT / "Start_Here.ipynb").resolve(), "")]
    assert (ROOT / "ETC/environment/SETUP.md").resolve() in {resolved for _, resolved, _ in links}
    assert not any(line.lstrip().startswith("|") for line in source.splitlines()), (
        "Keep the lesson table in Start_Here.ipynb, not the GitHub introduction"
    )


@pytest.mark.parametrize("relative", ["ETC/course_materials/README.md", "ETC/course_materials/course-plan.md"])
def test_course_guides_cover_manifest_order_and_current_topics(relative):
    manifest = json.loads((ROOT / "ETC/course_materials/course_manifest.json").read_text(encoding="utf-8"))
    source = markdown_text(ROOT / relative)
    expected = [(ROOT / item["notebook"]).resolve() for item in manifest["course"]]
    listed = [resolved for _, resolved, _ in local_links(relative) if resolved in expected]
    # Some guides also introduce Lab 1 before the table. Check the table itself.
    table = "\n".join(line for line in source.splitlines() if line.startswith("|"))
    resolved_table = [(ROOT / relative).parent.joinpath(unquote(urlsplit(target).path)).resolve()
                      for target in link_targets(table)]
    assert [path for path in resolved_table if path in expected] == expected
    assert all(path in listed for path in expected)
    operator_row = next(line for line in table.splitlines() if "04_neural_operators" in line)
    assert all(method in operator_row for method in ("FNO", "AFNO", "PINO"))
    heat_row = next(line for line in table.splitlines() if "03_heat_conduction" in line)
    assert re.search(r"heat|conduction", heat_row, re.IGNORECASE)


def test_local_schedule_uses_instructor_agenda_and_actual_challenge_topics():
    source = (ROOT / "ETC/course_materials/course-plan.md").read_text(encoding="utf-8")
    assert "Private planning links and internal coordination notes are not distributed" in source
    assert "instructor-provided 09:30–17:30 agenda" in source
    assert "110 individual participants" in source
    schedule = source.split("## 8-hour", 1)[1].split("\n## ", 1)[0]
    rows = "\n".join(line for line in schedule.splitlines() if line.startswith("|"))
    assert re.search(r"flow|fluid", rows, re.IGNORECASE)
    assert re.search(r"climate|temperature|atmosphere", rows, re.IGNORECASE)
    assert all(method in rows for method in ("FNO", "AFNO", "PINO"))
    assert not any(term in rows for term in ("FourCastNet", "Darcy", "Magnetohydrodynamics"))


@pytest.mark.parametrize("relative", ["README.md", "ETC/course_materials/README.md", "ETC/course_materials/course-plan.md", "ETC/course_materials/INSTRUCTOR.md", "ETC/course_materials/ASSESSMENT.md", "ETC/environment/SETUP.md"])
def test_guide_links_and_anchors_are_valid(relative):
    for target, resolved, fragment in local_links(relative):
        assert resolved.is_relative_to(ROOT) and resolved.exists(), (relative, target)
        if fragment and resolved.suffix.lower() in {".md", ".ipynb"}:
            assert fragment in fragments(resolved), (relative, target)
