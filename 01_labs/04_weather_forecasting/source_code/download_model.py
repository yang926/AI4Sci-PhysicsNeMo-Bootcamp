"""Download only the pinned public NVIDIA model artifacts; verify full SHA256."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


REVISION = "c67a63995f6c8e0e557eb3d791f32f437e9b02d5"
BASE = f"https://huggingface.co/nvidia/fourcastnet1/resolve/{REVISION}"
FILES = {
    "fcn.mdlus": (301168640, "995cdfdc3b64330caade5518aff09e0ce8f941b4262f3f8eb792b6fea8b6423a"),
    "global_means.npy": (336, "dd207b78084ad28387c1dc1110ac6db3bc85b25182433ce854ddb8968f219921"),
    "global_stds.npy": (336, "d2adff90952d7980e1951e3b7688451dbb11b63eb04c1b43991e30b2d65e3dcb"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        os.environ.get("AI4SCI_WEATHER_CACHE", "~/.cache/ai4sci/weather")
    ).expanduser() / "model")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, (expected_size, expected_hash) in FILES.items():
        destination = output / name
        if destination.exists():
            if destination.stat().st_size != expected_size or digest(destination) != expected_hash:
                raise ValueError(f"Existing artifact failed verification: {name}")
            print(f"Verified existing {name}", flush=True)
            continue
        partial = None
        try:
            with tempfile.NamedTemporaryFile(dir=output, prefix=name+".", suffix=".partial", delete=False) as target:
                partial = Path(target.name)
                count = 0
                with urllib.request.urlopen(f"{BASE}/{name}", timeout=120) as response:
                    while chunk := response.read(1024*1024):
                        count += len(chunk)
                        if count > expected_size:
                            raise ValueError(f"Unexpected download size: {name}")
                        target.write(chunk)
            if count != expected_size or digest(partial) != expected_hash:
                raise ValueError(f"Downloaded artifact failed verification: {name}")
            partial.rename(destination)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}: {count} bytes", flush=True)


if __name__ == "__main__":
    main()
