# Install with uv and open the course in JupyterLab

The course uses **Python 3.12**, **PhysicsNeMo 2.2.2** with its integrated `sym` extra, and a separate virtual environment. The CUDA recipe below pins the PyTorch versions used by the local RTX 3080 checks. A Brev **NVIDIA L40** is the planned next target, not a tested environment; confirm the actual GPU with `nvidia-smi` before using the CUDA path. See the [validation record](../course_materials/VALIDATION.md) for the scope of each result.

For the student Brev Launchable, use the [managed Jupyter setup](../launchable/README.md). The manual server instructions below are for standalone machines, not an additional server to start beside Brev-managed Jupyter.

Choose the section that matches your situation:

- JupyterLab already opens: go to [Start Here](../../Start_Here.ipynb).
- A working environment exists, but Jupyter is stopped: [start JupyterLab](#start-jupyterlab-on-the-remote-machine).
- The Mac browser lost its connection: [reconnect the tunnel](#connect-from-the-mac-and-reconnect-after-moving).
- A new remote machine needs Python packages: keep reading below.

## Keep an existing workspace

If the repository is already open, use that working copy. Run `pwd` and `git status --short`, save notebook and Python edits, and retain existing datasets and outputs. Do not replace the folder or recreate an existing environment to follow this guide.

For a new machine with no working copy, clone into an unused directory:

```bash
git clone https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git
cd AI4Sci-PhysicsNeMo-Bootcamp
```

Run installation commands in the **remote Linux terminal** for Brev, or in **WSL** for the Windows desktop. The Mac terminal is used later for the connection tunnel. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if `uv --version` is unavailable. Command options below were checked against uv 0.8.17; `uv venv --python 3.12.11` can download that interpreter if it is missing. These are commands to run when preparing the machine, not evidence that a new Brev environment has already been installed.

If this checkout already has a working `.venv`, activate it and continue at **Check the environment**. An environment linked to a directory on another machine does not transfer with a Git clone.

## New Linux NVIDIA GPU environment

From the repository root, inspect the GPU and driver first:

```bash
nvidia-smi
uv --version
```

Create a new environment at the path below. The guard preserves an existing directory or symlink; if the name is already in use, choose a different path or reuse the existing environment after checking it.

```bash
course_env="$HOME/.venvs/ai4sci-brev-cuda"
(
  set -eu
  if test -e "$course_env" || test -L "$course_env"; then
    printf 'Environment already exists: %s. Reuse it or choose a new path.\n' "$course_env"
    exit 1
  fi
  uv venv --python 3.12.11 "$course_env"
  uv pip install --python "$course_env/bin/python" \
    'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' \
    --default-index https://download.pytorch.org/whl/cu128
  uv pip install --python "$course_env/bin/python" -r requirements.txt \
    'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128'
  uv pip check --python "$course_env/bin/python"
  "$course_env/bin/python" -m ipykernel install --prefix "$course_env" \
    --name ai4sci-physicsnemo-uv --display-name 'AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)'
)
```

After successful installation, activate it:

```bash
source "$HOME/.venvs/ai4sci-brev-cuda/bin/activate"
```

The second install explicitly retains the CUDA PyTorch versions while resolving the course requirements from PyPI. `nvidia-physicsnemo[sym]==2.2.2` is already in `requirements.txt`; do not add the separate legacy `nvidia-physicsnemo.sym` package. This recipe pins the core framework versions, **not every transitive dependency**. The desktop's `ETC/environment/requirements-wsl-cuda.lock.txt` is a local snapshot, not a file provided by a fresh GitHub clone. Fresh installation and driver compatibility still require verification on the target GPU. See the official [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for the index model.

## New Linux CPU environment

Use a different environment for CPU work. This path reproduces the framework versions in the preserved Linux CPU package snapshot, not the CUDA recipe above.

```bash
course_env="$HOME/.venvs/ai4sci-linux-cpu"
(
  set -eu
  if test -e "$course_env" || test -L "$course_env"; then
    printf 'Environment already exists: %s. Reuse it or choose a new path.\n' "$course_env"
    exit 1
  fi
  uv venv --python 3.12.11 "$course_env"
  uv pip install --python "$course_env/bin/python" \
    'torch==2.14.0+cpu' 'torchvision==0.29.0+cpu' \
    --default-index https://download.pytorch.org/whl/cpu
  uv pip install --python "$course_env/bin/python" -r ETC/environment/requirements-linux-cpu.lock.txt
  uv pip install --python "$course_env/bin/python" 'ipywidgets>=8.1,<9'
  uv pip check --python "$course_env/bin/python"
  "$course_env/bin/python" -m ipykernel install --prefix "$course_env" \
    --name ai4sci-physicsnemo-uv --display-name 'AI4Sci PhysicsNeMo 2.2.2 (uv CPU)'
)
```

After success, activate `source "$HOME/.venvs/ai4sci-linux-cpu/bin/activate"` and use `AI4SCI_DEVICE=cpu` and `--device cpu` below. The Linux snapshots are not macOS environments. The historical [macOS package snapshot](requirements-macos.lock.txt) used CPU PyTorch from PyPI; Apple MPS and AMD GPU execution have not been validated. A Mac used only as the browser and SSH client does not need the course Python packages.

## Check the environment

The current requirements include `ipywidgets` for notebook progress bars. When reusing an environment from either historical package snapshot, install it with `uv pip install --python "$VIRTUAL_ENV/bin/python" 'ipywidgets>=8.1,<9'`. Install it in the course environment, not on a Mac used only as the browser. After adding it to a running session, reopen the notebook and restart its kernel before running the checks. Save any work first. See the [Jupyter widgets installation guide](https://ipywidgets.readthedocs.io/en/stable/user_install.html).

In the activated environment, from the repository root:

```bash
python --version
uv pip check --python "$VIRTUAL_ENV/bin/python"
python -c 'import sys, torch; from importlib.metadata import version; print("Python:", sys.executable); print("PhysicsNeMo:", version("nvidia-physicsnemo")); print("PyTorch:", torch.__version__); print("PyTorch CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

For the CUDA path, verify `2.2.2`, `2.10.0+cu128`, CUDA availability, and the actual GPU model. The driver reported by `nvidia-smi` and the CUDA runtime reported by PyTorch describe different parts of the installation. Run [the environment-check notebook](../../00_Setup.ipynb) in the selected course kernel before beginning a lesson; `AI4SCI_DEVICE=cuda` makes unavailable CUDA an error.

## Start JupyterLab on the remote machine

Keep Jupyter running in the remote terminal. If `tmux` is available, start `tmux new -s ai4sci` first, then change to the repository root and activate the intended environment inside that session. Detach with **Ctrl+B, then D** and reconnect with `tmux attach -t ai4sci`. This keeps the remote process independent of the SSH terminal; it does not survive a machine restart or deletion.

For an instructor's short CUDA rehearsal:

```bash
AI4SCI_DEVICE=cuda AI4SCI_REFERENCE=0 AI4SCI_STEPS=20 \
  jupyter lab --no-browser --ip=127.0.0.1 --port=8888 \
  --ServerApp.port_retries=0 --ServerApp.root_dir="$PWD" \
  --LabApp.default_url=/lab/tree/Start_Here.ipynb
```

Keep the generated token authentication enabled and retain the printed localhost URL. If port 8888 is occupied, reuse the intended existing course server or choose another port and match the forward below; do not stop an unrelated server. All four Challenges start in student mode and ignore `AI4SCI_REFERENCE`. The 20-step override is a short check; omit `AI4SCI_STEPS` to use each notebook's stated lesson budget. Lab 4 restores the original 50,000-update budget and needs a longer run. Device and step environment variables select defaults when launching a new server; already-running kernels keep their environment. For a demonstration explicitly set `USE_REFERENCE = True` in the notebook, then return it to `False` for student work. No server restart is needed; check the mode printed before every run.

Open `Start_Here.ipynb`, run the environment check in the matching **AI4Sci** kernel, then follow the introduction and Labs. The introduction is reading material; Lab 1 is the first training exercise. The `ai4sci-physicsnemo-uv` kernelspec above matches the course notebooks.

## Read Markdown as a document

In JupyterLab, open **Settings > Settings Editor > Document Manager** and set the default viewer for `markdown` to `Markdown Preview`. In the JSON settings editor, the corresponding setting is:

```json
{
  "defaultViewers": {
    "markdown": "Markdown Preview"
  }
}
```

Reload the browser page after saving. Existing editor tabs keep their original view; close and reopen them, or right-click the file and choose **Open With > Markdown Preview**. Save any unsaved edits first. To edit Markdown source, use **Open With > Editor**.

The course starts at `Start_Here.ipynb`. The root `README.md` is a short introduction for GitHub, not a second lesson index.

## Connect from the Mac and reconnect after moving

In a **Mac terminal**, use the existing Brev instance name in place of `YOUR_INSTANCE`:

```bash
brev login
brev refresh
brev port-forward YOUR_INSTANCE --port 8888:8888
```

Leave the forwarding process running and open the remote server's token-bearing URL at local `127.0.0.1:8888`. If that local port is busy, use `--port 8889:8888` and change only the port in the browser URL to 8889. The mapping is **local:remote**. [Brev connectivity documentation](https://docs.nvidia.com/brev/cli/connectivity)

Moving networks, sleeping the Mac, or closing its forwarding process can disconnect the browser. Reconnect the Mac, rerun the port forward, and reload the URL. If the instance was restarted, run `brev refresh` first, reconnect to its shell, and check whether Jupyter needs starting again. A lost tunnel alone does not prove that remote training stopped. Check the remote process and current run's artifacts before launching a duplicate job. Keep token URLs and SSH keys out of GitHub and shared screenshots.

## Data, edits, and saved runs

For Challenge submission, instructors configure the [notebook judge connection](JUDGE_CONNECTION.md).
Students submit and view results inside the notebook; the separate web page is
only the scoreboard. This does not require installing another application on
the student's laptop.

Use a private writable checkout for each learner. Confirm which disk or mounted workspace survives your Brev instance/container lifecycle; a browser connection and an environment do not make storage persistent. Back up saved notebooks, edited `.py` files, and wanted outputs before replacing a workspace or deleting an instance.

The Operator lesson generates its own 64x64 reaction-diffusion dataset, with 8,000/1,000/1,000 train/validation/test samples. Generate it once per prepared workspace, validate it, and reuse it across its three levels. The notebook accepts `AI4SCI_DATA_DIR` for another prepared data location. Historical `Poisson_Fourier` files are not input for this implementation. Navier–Stokes uses its tracked original `data_lat.npy` array, with no synthetic fallback or bundled pretrained model. Prepare its longer 50,000-update run before a short live demonstration. Its notebook plays 11 predicted frames and exports a ParaView ZIP for local playback. See [data provenance](../../01_labs/04_navier_stokes/DATA_PROVENANCE.md).

Each successful training run saves `metrics.json`, `loss.csv`, `model.pt`, `predictions.npz`, and `preview.png`. Existing output directories are protected. Notebook training cells select a new result directory on every execution, so setup need not be rerun just to repeat a level. For a command-line run, choose a fresh `--output-dir` and an explicit `--steps` value yourself. These scripts save final artifacts; a killed training process is not guaranteed to have a resumable intermediate checkpoint. The default generated-data and output locations are ignored by Git, so pushing teaching code does not back them up. If you choose custom paths, check `git status` before publishing to avoid accidentally including generated data or private results.

## Short checks and lesson rehearsal

After activation, create a fresh parent directory for each validation session:

```bash
mkdir -p ETC/validation-runs
validation_root="$(mktemp -d "$PWD/ETC/validation-runs/cuda-check-XXXXXX")"
python ETC/course_materials/run_validation.py --suite unit --device cpu --output-dir "$validation_root/unit"
python ETC/course_materials/run_validation.py --suite smoke --case pinn_forward --device cuda --steps 2 --output-dir "$validation_root/first-gpu-check"
```

If that first GPU check succeeds, verify the broader execution paths:

```bash
python ETC/course_materials/run_validation.py --suite smoke --device cuda --steps 20 --output-dir "$validation_root/smoke"
python ETC/course_materials/run_notebooks.py --device cuda --steps 2 --output-dir "$validation_root/notebooks"
```

The smoke suite checks all 18 modes and their artifacts. The notebook suite runs 12 notebooks in instructor reference mode, verifies embedded plots, and saves executed copies outside the source notebooks. Its Operator notebook generates the full-size lesson dataset, while the smoke suite uses a smaller benchmark. Neither is a full training or lesson-duration measurement.

For a clean checkout intended for publication, run the strict material check:

```bash
python ETC/course_materials/validate_materials.py --output "$validation_root/materials-clean.json"
```

For a working copy whose notebook outputs you want to keep, use the explicit relaxed check instead:

```bash
python ETC/course_materials/validate_materials.py --allow-executed-notebooks --output "$validation_root/materials-working-copy.json"
```

The latter preserves outputs and reports that clean-output validation was skipped; it is not a clean-publication check. Do not clear a learner's results merely to satisfy the strict check.

The separate `--suite convergence --convergence-steps 500` option measures held-out improvement for selected cases. It is a longer experiment, not a certification of full convergence. A real teaching rehearsal additionally measures explanation, saved student edits, default-step runs, error recovery, and result interpretation on the intended hardware. L40 timings and multi-user resource planning remain unmeasured; do not scale the local short-check timings into those claims.

## Optional Docker path

The [Dockerfile](Dockerfile) uses the official PhysicsNeMo 26.08 image. The recorded uv/WSL results do not verify that image's build, host-driver compatibility, or execution. If preparing this separate path, retain authentication, publish Jupyter only on host localhost, and mount a dedicated persistent checkout:

```bash
docker build -f ETC/environment/Dockerfile -t ai4sci-physicsnemo:2.2.2 .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 -v "$PWD:/workspace/ai4sci" \
  -it --rm ai4sci-physicsnemo:2.2.2
```

The container's Jupyter binds inside the container; the host mapping above restricts access to localhost for forwarding. The mounted checkout retains its files when the container exits. Other unmounted container files disappear with `--rm`; do not mount a shared learner checkout that multiple people will edit concurrently.

[Complete course](../../Start_Here.ipynb) · [Instructor guide](../course_materials/INSTRUCTOR.md) · [uv environment selection](https://docs.astral.sh/uv/pip/environments/)
