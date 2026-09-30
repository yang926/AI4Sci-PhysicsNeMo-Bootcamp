"""Pretrained NVIDIA FourCastNet AFNO, initial-state-only 6-hour inference.

Uses the official 26-channel checkpoint and normalization contract documented by
NVIDIA Earth2Studio (Apache-2.0):
https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/fcn.py
No future verification file is accepted or loaded by this program.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from physicsnemo.models.afno import AFNO

from download_model import FILES, REVISION


VARIABLES = ["u10m", "v10m", "t2m", "sp", "msl", "t850", "u1000", "v1000", "z1000",
             "u850", "v850", "z850", "u500", "v500", "z500", "t500", "z50", "r500",
             "r850", "tcwv", "u100m", "v100m", "u250", "v250", "z250", "t250"]
OUTPUT_VARIABLES = ["u10m", "v10m", "msl", "t2m"]
OUTPUT_INDICES = [VARIABLES.index(name) for name in OUTPUT_VARIABLES]
LATITUDE = np.arange(90., -90., -.25)
LONGITUDE = np.arange(0., 360., .25)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_initial(path):
    with np.load(path, allow_pickle=False) as saved:
        state = saved["state"].copy()
        variables = saved["variables"].tolist()
        latitude, longitude = saved["latitude"].copy(), saved["longitude"].copy()
        initial_time = saved["time"].copy()
    if state.shape != (26, 720, 1440) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("Expected finite FP32 initial state [26,720,1440]")
    if variables != VARIABLES:
        raise ValueError("Initial variables must match the official checkpoint order exactly")
    if not np.array_equal(latitude, LATITUDE) or not np.array_equal(longitude, LONGITUDE):
        raise ValueError("Expected north-to-south, south-pole-excluding 0.25-degree global grid")
    if initial_time.shape != () or initial_time.dtype.kind != "M" or np.isnat(initial_time):
        raise ValueError("Initial time must be a scalar valid datetime64")
    hourly = initial_time.astype("datetime64[h]")
    if initial_time != hourly or int(hourly.astype(np.int64)) % 6:
        raise ValueError("Initial time must be an exact UTC 00/06/12/18-hour analysis cycle")
    return state, hourly


def load_model(folder, device):
    for name, (size, sha) in FILES.items():
        path = folder / name
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError(f"Pinned artifact mismatch: {name}")
    model = AFNO.from_checkpoint(str(folder / "fcn.mdlus"), strict=True).to(device).eval()
    mean = torch.tensor(np.load(folder / "global_means.npy", allow_pickle=False), dtype=torch.float32, device=device)
    std = torch.tensor(np.load(folder / "global_stds.npy", allow_pickle=False), dtype=torch.float32, device=device)
    if mean.shape != (1,26,1,1) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid official normalization")
    return model, mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).with_name("model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="One artificial constant-state pass for memory/API check, not a weather forecast")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if not args.smoke and args.initial is None:
        parser.error("Supply --initial or use --smoke only for API/memory testing")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    program_started = time.perf_counter()
    model, mean, std = load_model(args.model_dir, args.device)
    if args.smoke:
        current = mean.expand(1,26,720,1440).clone()
        initial_time, outputs = None, []
        steps = 1
    else:
        physical_state, initial_time = load_initial(args.initial)
        outputs = [physical_state[OUTPUT_INDICES].copy()]
        current = torch.from_numpy(physical_state[None]).to(args.device)
        steps = args.steps
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    normalized_initial = (current - mean) / std
    initial_normalized_statistics = {
        name: {"mean":float(normalized_initial[0,i].mean()),
               "std":float(normalized_initial[0,i].std()),
               "min":float(normalized_initial[0,i].min()),
               "max":float(normalized_initial[0,i].max())}
        for i,name in enumerate(VARIABLES)
    }
    del normalized_initial
    step_seconds, channel_ranges = [], []
    with torch.inference_mode():
        for step in range(1, steps+1):
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            # Official mapping predicts a complete next normalized state, NOT an increment.
            current = model((current - mean) / std) * std + mean
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            if not torch.isfinite(current).all():
                raise FloatingPointError(f"Nonfinite forecast at lead {6*step}h")
            step_seconds.append(seconds)
            channel_ranges.append({name: {"min": float(current[0,i].min()), "max": float(current[0,i].max()),
                                          "mean": float(current[0,i].mean())} for i,name in enumerate(VARIABLES)})
            if not args.smoke:
                outputs.append(current[0, OUTPUT_INDICES].cpu().numpy().copy())
            print(json.dumps({"lead_hours":6*step, "inference_seconds":seconds,
                              "cuda_peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None,
                              "smoke_only":args.smoke}), flush=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint_revision=REVISION, checkpoint_sha256=FILES["fcn.mdlus"][1],
                  initial_sha256=digest(args.initial) if not args.smoke else None,
                  source_sha256=digest(__file__), device=args.device,
                  hardware=torch.cuda.get_device_name() if args.device=="cuda" else "CPU",
                  pytorch_version=torch.__version__, precision="FP32, TF32 disabled, no autocast",
                  steps=steps, step_seconds=step_seconds, inference_seconds=sum(step_seconds),
                  wall_seconds_before_output_io=time.perf_counter()-program_started,
                  cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else None,
                  future_verification_data_used_as_input=False, smoke_only=args.smoke,
                  variables=VARIABLES, output_variables=OUTPUT_VARIABLES, channel_ranges=channel_ranges,
                  initial_normalized_statistics=initial_normalized_statistics,
                  normalization="unchanged pinned official mean/std for all26 channels",
                  limitations="Runtime describes only the recorded hardware; one retrospective case is not an operational weather benchmark")
    args.output.mkdir(parents=True)
    if not args.smoke:
        lead_hours = np.arange(steps+1)*6
        np.savez_compressed(args.output/"forecast.npz", prediction=np.stack(outputs),
                            variables=np.array(OUTPUT_VARIABLES), latitude=LATITUDE, longitude=LONGITUDE,
                            times=initial_time+lead_hours.astype("timedelta64[h]"), lead_hours=lead_hours)
    (args.output/"runtime.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {args.output}; numerical forecast verification is separate.", flush=True)


if __name__ == "__main__":
    main()
