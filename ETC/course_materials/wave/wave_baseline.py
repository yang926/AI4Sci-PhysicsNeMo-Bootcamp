"""Instructor entry point for the current full-course Wave Level 1.

Uses the same c=1, [0, pi]^2, t in [0, 2*pi] setup as the challenge.
python ETC/course_materials/wave/wave_baseline.py --steps 1000 --device cpu --output-dir runs/wave-reference
"""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[3] / "02_challenges/01_wave/wave_l1.py"
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), "--reference", *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")
