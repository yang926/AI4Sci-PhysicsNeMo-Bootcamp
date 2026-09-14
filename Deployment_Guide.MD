# Installation and validation

Use **PhysicsNeMo 2.2.2**, Python **3.11–3.14**, and PyTorch **2.10 or later**. An isolated Python 3.12 virtual environment is recommended. See the [validation record](ai4sci/VALIDATION.md) for measured environments and results.

## Linux CPU environment

Run these commands from the repository root. Install CPU PyTorch first.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.10' 'torchvision>=0.25' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name python3 --display-name 'AI4Sci PhysicsNeMo 2.2.2'
jupyter lab
```

For exact reproduction on Linux x86_64/Python 3.12, install [requirements-linux-cpu.lock.txt](requirements-linux-cpu.lock.txt) in a new virtual environment instead of the two package-installation commands above.

On macOS, omit the `--index-url` argument and install PyTorch from PyPI. The tested macOS arm64/Python 3.12 package set is [requirements-macos.lock.txt](requirements-macos.lock.txt). Do not use that platform-specific lock file on Linux. Use `--device cpu` for CPU runs. AMD GPU and Apple MPS execution have not been validated here.

Open the [course guide](ai4sci/README.md) in JupyterLab's Markdown Preview and run the [environment check](ai4sci/00_environment_check.ipynb). Confirm that the kernel uses the intended virtual environment.

## NVIDIA GPU container

The Dockerfile starts from the [official 26.08 container](https://docs.nvidia.com/physicsnemo/26.08/release-notes/index.html) and installs the course's pinned 2.2.2 package. Check the container release notes for host-driver compatibility. Actual build and GPU execution status are recorded separately in the validation record.

```bash
docker build -t ai4sci-physicsnemo:2.2.2 .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 -it --rm ai4sci-physicsnemo:2.2.2
```

Connect using the token printed in the log. For a remote event, use the authenticated access route provided by the organizers. Work stored only inside this `--rm` container disappears when it exits; use a dedicated workspace volume for participant work that must persist.

## Reproduce the checks

Run from the repository root and choose a fresh output directory each time.

```bash
python ai4sci/validate_materials.py --output validation-runs/static.json
python ai4sci/run_validation.py --suite unit --device cpu --output-dir validation-runs/unit
python ai4sci/run_validation.py --suite smoke --device cpu --output-dir validation-runs/smoke
python ai4sci/run_validation.py --suite convergence --device cpu --output-dir validation-runs/convergence
python ai4sci/run_notebooks.py --device cpu --steps 2 --output-dir validation-runs/notebooks
```

Use `--device cuda` to require a GPU; missing CUDA fails rather than silently passing a CPU run. Smoke checks execute all eighteen lesson modes and validate their artifacts. The separate improvement suite checks held-out or analytical error reduction; it does not certify full convergence or weather-forecast accuracy. Notebook checks execute all twelve notebooks and require embedded plots for each practical lesson.

[Complete course](Start_Here.ipynb) · [Instructor guide](ai4sci/INSTRUCTOR.md)
