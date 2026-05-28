# MuJoCo Setup Guide (Phase 0.2.1)

Reproduce a working MuJoCo runtime on the lab PC in under 60 minutes.

## Prerequisites

- Ubuntu 22.04 (or compatible)
- Python 3.10 (system Python 3.10.12 works)
- NVIDIA GPU + driver (for later rendering phases; not required for stepping)
- Git checkout of the VLA-LEGO repository

## 1. Install system-level OpenGL dependencies

These are needed for MuJoCo's rendering backends (interactive viewer in Phase 0.2.2, headless in Phase 0.2.3). Install them now to avoid revisiting later.

```bash
sudo apt-get update && sudo apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev
```

## 2. Create a virtual environment

A **virtual environment** isolates this project's Python packages from the system Python.
This prevents version conflicts with other projects and avoids the
`"Defaulting to user installation"` issues with system pip.

```bash
# Create the venv (one-time setup)
python3 -m venv .venv

# Activate it (run this every time you open a new terminal)
source .venv/bin/activate

# Your prompt will change to show (.venv):
# (.venv) pacquadr@elab:~/VLA-LEGO_Project$

# Upgrade pip and setuptools inside the venv
pip install --upgrade pip setuptools wheel
```

**Key commands:**
- `source .venv/bin/activate` — activate the venv (use it before working on the project)
- `deactivate` — exit the venv (return to system Python)
- `which python` — should show `.venv/bin/python` when activated

**VS Code integration:** VS Code should auto-detect `.venv/`. If not, open the
Command Palette (Ctrl+Shift+P) > "Python: Select Interpreter" > choose the one
at `./.venv/bin/python`.

## 3. Install Python dependencies

With the venv activated:

```bash
# Simulation deps only
pip install -e ".[sim]"

# Or full dev environment (includes sim + CI + training + pre-commit)
pip install -e ".[dev]"
```

This installs `mujoco>=3.1.0,<4.0.0` (DeepMind's official Python bindings) in editable
mode — code changes take effect immediately without reinstalling.

## 4. Validate the installation

```bash
python scripts/validate_mujoco.py
```

### Expected output

```
============================================================
Phase 0.2.1 Validation: MuJoCo Runtime Installation
============================================================

[1/4] Checking mujoco import...
  OK: mujoco 3.5.0

[2/4] Loading minimal MJCF scene...
  OK: model loaded (nq=7, nv=6, timestep=0.002)

[3/4] Validating deterministic stepping (1000 steps x 3 trials)...
  OK: stepping is deterministic

[4/4] Collecting environment metadata...
  os: Linux 6.8.0-40-generic
  python_version: 3.10.12 (...)
  gpu_driver: 535.183.01
  mujoco_version: 3.5.0
  git_commit: <hash>
  git_dirty: False
  deps_hash: <hash>

  Metadata saved to logs/mujoco_env_meta.json

============================================================
ALL CHECKS PASSED
============================================================
```

## 5. Run the test suite

```bash
# MuJoCo tests only
pytest tests/test_mujoco.py -v

# All tests (MuJoCo tests auto-skip if not installed)
pytest -m "not slow and not gpu"
```

## Troubleshooting

### `ImportError: libGL.so.1: cannot open shared object file`
Install Mesa OpenGL:
```bash
sudo apt-get install libgl1-mesa-glx
```

### `ImportError: No module named 'mujoco'`
Ensure your venv is activated and you installed with sim extras:
```bash
source .venv/bin/activate
pip install -e ".[sim]"
```

### `build_editable` error with `pip install -e`
Your pip or setuptools is too old. Upgrade inside the venv:
```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[sim]"
```

### `mujoco.FatalError` when loading MJCF
Check that `sim/assets/scenes/test_scene.xml` exists and is valid XML. Re-clone or `git checkout` if corrupted.

## 6. Interactive Viewer (Phase 0.2.2)

Once validation passes, you can open the interactive viewer to visually inspect scenes:

```bash
# Basic viewer
vla-viewer sim/assets/scenes/test_scene.xml

# With contact and joint overlays
vla-viewer sim/assets/scenes/test_scene.xml --show-contacts --show-joints
```

See [viewer-debug-checklist.md](viewer-debug-checklist.md) for the full debug workflow and
keyboard shortcuts reference.

### Wrong Python version
Verify you're using the venv Python:
```bash
which python  # Should show .venv/bin/python
python --version  # Should be 3.10.x
```
