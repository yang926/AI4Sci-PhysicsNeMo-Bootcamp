"""One worker per GPU (or one CPU worker for local development)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .catalog import ROOT


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
    request.write_text(json.dumps({key: job[key] for key in ("challenge", "sources", "settings")}))
    environment = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT") if key in os.environ}
    environment.update(MPLBACKEND="Agg", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu) if device == "cuda" else "")
    command = [sys.executable, "-m", "ETC.judge.evaluate", "--input", str(request),
               "--output", str(result_path), "--runs", str(directory / "artifacts"), "--device", device]
    process = None
    try:
        with (directory / "runner.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log, start_new_session=True)
            try:
                process.wait(timeout=job["settings"]["timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                store.finish(job, status="time_limit", error="Submission exceeded the pilot time limit. Previous best scores are unchanged.")
                return True
        if process.returncode != 0 or not result_path.is_file() or result_path.stat().st_size > 1024 * 1024:
            raise RuntimeError("Evaluator failed")
        store.finish(job, result=json.loads(result_path.read_text()))
    except Exception:
        store.finish(job, status="system_error", error="Evaluation failed. Ask the instructor to inspect this submission's private runner log; then resubmit.")
    except BaseException:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        store.finish(job, status="system_error", error="Worker interrupted; resubmit this code.")
        raise
    return True
