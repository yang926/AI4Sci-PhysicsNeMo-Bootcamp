"""Student checks intentionally contain no mathematical answer fixtures."""
from pathlib import Path
import tempfile
import unittest

from ETC.judge.expressions import SubmissionError
from ETC.runtime.submission_preflight import check_saved_submission
from ETC.runtime.submission_widgets import preflight_html
from ETC.runtime.notebook import result_html


# Deliberately incorrect dummy results: format acceptance is not correctness.
DUMMY = '''
def student_equations(x, y, t, u, c):
    return {"dummy": 0}
def student_conditions():
    return {"dummy": 0}
def student_speed(x, y):
    return {"dummy": 0}
'''


class StudentPreflightTests(unittest.TestCase):
    def payload(self, source=DUMMY):
        return {"challenge": "1", "sources": {"wave_l1.py": source}}

    def test_does_not_award_points_or_claim_correctness(self):
        result = check_saved_submission(self.payload())
        self.assertFalse(result["correctness_checked"])
        self.assertEqual(result["scope"], "format_only")
        self.assertEqual(result["levels"]["wave_l1.py"]["status"], "format_checked")
        self.assertEqual(result["levels"]["wave_l2.py"]["status"], "not_submitted")
        self.assertNotIn("score", result)
        rendered = preflight_html(result)
        self.assertIn("Mathematical correctness and points are checked by the judge", rendered)
        self.assertNotIn("checks passed", rendered)

    def test_placeholder_after_return_is_unfinished(self):
        result = check_saved_submission(self.payload(DUMMY.replace(
            'return {"dummy": 0}', 'return {"dummy": 0}\n    raise NotImplementedError()', 1)))
        self.assertEqual(result["levels"]["wave_l1.py"]["status"], "invalid")

    def test_missing_definition_and_syntax(self):
        for source in ("", "def student_equations(:", "x = 1", DUMMY.replace("student_speed", "other_name")):
            with self.subTest(source=source):
                result = check_saved_submission(self.payload(source))
                self.assertEqual(result["levels"]["wave_l1.py"]["status"], "invalid")

    def test_no_code_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "must-not-exist"
            source = f"open({str(marker)!r}, 'w').write('no')\n" + DUMMY
            check_saved_submission(self.payload(source))
            self.assertFalse(marker.exists())

    def test_size_and_request_bounds(self):
        result = check_saved_submission(self.payload("#" * 65537))
        self.assertEqual(result["levels"]["wave_l1.py"]["status"], "invalid")
        for payload in (None, {}, {"challenge": "9", "sources": {}},
                        {"challenge": "1", "sources": {"unknown.py": DUMMY}}):
            with self.assertRaises(SubmissionError):
                check_saved_submission(payload)
        for timeout in (0, 61, float("inf"), True):
            with self.assertRaises(ValueError):
                check_saved_submission(self.payload(), timeout=timeout)

    def test_student_comparison_is_not_called_ground_truth(self):
        metrics = {"steps": 1, "reference_implementation": False,
                   "comparison_source": "student_solution", "reference_over_time": {
                       "rmse": 1.0, "per_time": [{"time": 0.0, "rmse": 1.0, "relative_l2": 0.5}]}}
        rendered = result_html(metrics, Path("unused"))
        self.assertIn("not independently checked", rendered)
        self.assertNotIn("Analytical comparison across", rendered)


if __name__ == "__main__":
    unittest.main()
