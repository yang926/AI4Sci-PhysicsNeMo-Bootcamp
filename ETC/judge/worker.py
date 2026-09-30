"""One worker per GPU (or one CPU worker for local development)."""
import json
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import tempfile
import time

from .catalog import ROOT
from .result_checkpoint import bounded_json, recover_checkpoint, validate_result


def work_once(store, device="cpu", gpu=None):
    if device == "cuda" and (gpu is None or not str(gpu).isdigit()):
        raise ValueError("Assign one GPU explicitly with --gpu 0, --gpu 1, etc.")
    if store.settings()[0]["device"] != device:
        raise ValueError("Worker device differs from the frozen scoring session. Create a separate state for CPU/GPU trials.")
    job = store.claim()
    if job is None:
        return False
    runs = store.directory / "runs"
    runs.mkdir(mode=0o700, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=job["id"] + "-", dir=runs))
    request, result_path = directory / "request.json", directory / "result.json"
    checkpoint = directory / "implementation-checkpoint.json"
    payload = {key: job[key] for key in ("challenge", "sources", "settings")}
    payload["run_nonce"] = secrets.token_hex(32)
    request.write_text(json.dumps(payload))
    environment = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT") if key in os.environ}
    environment.update(MPLBACKEND="Agg", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu) if device == "cuda" else "")
    command = [sys.executable, "-m", "ETC.judge.evaluate", "--input", str(request),
               "--output", str(result_path), "--checkpoint", str(checkpoint),
               "--runs", str(directory / "artifacts"), "--device", device]
    process = None
    evaluation_started = time.monotonic()
    def finish_failure(reason, status, message):
        try:
            recovered = recover_checkpoint(checkpoint, payload, reason)
        except (ValueError, OSError, TypeError, KeyError):
            store.finish(job, status=status, error=message)
        else:
            # "completed" means every submitted implementation was assessed;
            # numerical feedback explicitly retains its independent failure.
            recovered["elapsed_seconds"] = round(time.monotonic() - evaluation_started, 6)
            store.finish(job, result=validate_result(recovered, payload))
    try:
        with (directory / "runner.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log, start_new_session=True)
            try:
                process.wait(timeout=job["settings"]["timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                finish_failure("time_limit", "time_limit", "Submission exceeded the pilot time limit before implementation checks completed. Previous best scores are unchanged.")
                return True
        if process.returncode != 0 or not result_path.is_file() or result_path.stat().st_size > 1024 * 1024:
            raise RuntimeError("Evaluator failed")
        result = validate_result(bounded_json(result_path), payload)
        # Includes child startup and implementation checks; excludes queue wait.
        result["elapsed_seconds"] = round(time.monotonic() - evaluation_started, 6)
        store.finish(job, result=validate_result(result, payload))
    except Exception:
        finish_failure("failed", "system_error", "Evaluation failed before implementation checks completed. Ask the instructor to inspect this submission's private runner log; then resubmit.")
    except BaseException:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        finish_failure("failed", "system_error", "Worker interrupted before implementation checks completed; resubmit this code.")
        raise
    return True
