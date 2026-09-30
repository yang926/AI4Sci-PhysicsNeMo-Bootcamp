"""Offline checks, partial-credit submissions and honest feedback status."""
import pytest

from ETC.runtime.submission import collect_submission
from ETC.runtime.submission_preflight import check_saved_submission
from ETC.runtime.submission_widgets import SubmissionPanel, history_html
from ETC.tests.test_judge import dummy_source
from ETC.tests.test_notebook_submission import FakeClient


def checked(status="format_checked"):
    return {"scope": "format_only", "correctness_checked": False,
            "levels": {"wave_l1.py": {"status": status,
                                     "messages": ["Save all selected Levels."]}}}


def test_real_offline_check_does_not_train(tmp_path):
    (tmp_path / "wave_l1.py").write_text(dummy_source("1", "wave_l1.py"))
    result = check_saved_submission(collect_submission("1", tmp_path))
    assert result["scope"] == "format_only"
    assert result["correctness_checked"] is False and "score" not in result
    assert result["levels"]["wave_l1.py"]["status"] == "format_checked"
    assert not (tmp_path / "outputs").exists()


def test_offline_check_button_needs_no_identity_or_server(monkeypatch, tmp_path):
    monkeypatch.setenv("AI4SCI_JUDGE_CONFIG", str(tmp_path / "missing"))
    monkeypatch.delenv("AI4SCI_JUDGE_TOKEN", raising=False)
    monkeypatch.delenv("AI4SCI_JUDGE_URL", raising=False)
    monkeypatch.setattr("ETC.runtime.submission_widgets.check_saved_submission", lambda _: checked())
    (tmp_path / "wave_l1.py").write_text(dummy_source("1", "wave_l1.py"))
    panel = SubmissionPanel("1", tmp_path)
    try:
        assert not panel.check_button.disabled and panel.submit_button.disabled
        panel.check_button.click()
        assert "Nothing was submitted" in panel.status.value
        assert "Format checked" in panel.check_result.value
        assert "correctness and points are checked by the judge" in panel.check_result.value
        assert "checks passed" not in panel.check_result.value
    finally:
        panel.close()


@pytest.mark.parametrize("status,sent", [("invalid", 0), ("format_checked", 1)])
def test_preflight_blocks_invalid_format_without_claiming_mathematical_correctness(monkeypatch, tmp_path, status, sent):
    (tmp_path / "wave_l1.py").write_text(dummy_source("1", "wave_l1.py"))
    calls = []
    def check(payload):
        calls.append(payload)
        return checked(status)
    monkeypatch.setattr("ETC.runtime.submission_widgets.check_saved_submission", check)
    client = FakeClient()
    panel = SubmissionPanel("1", tmp_path, client=client)
    try:
        assert not calls and not client.sent
        panel.submit_button.click()
        assert len(calls) == 1 and len(client.sent) == sent
        assert panel.connected
        assert "Save all selected Levels" in panel.check_result.value
        panel.refresh_button.click()
        assert len(calls) == 1 and len(client.sent) == sent
    finally:
        panel.close()


@pytest.mark.parametrize("status", ["failed", "time_limit"])
def test_history_separates_credit_from_training_failure(status):
    html = history_html({"submissions": [{"id": "test", "challenge": "1", "status": "completed",
                         "score": 100, "result": {"feedback_status": status}}]}, "1")
    assert "100.00 / 100" in html and "Implementation score saved" in html
    assert "timed out" in html if status == "time_limit" else "feedback failed" in html
