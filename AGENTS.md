# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project Overview

VLA-Door-Opening is a Vision-Language-Action system for robotic door-opening manipulation with the IHMC Alex humanoid upper body. The active task is contact-rich articulated-object manipulation: perceive the door, handle, hinge, latch state, and current opening angle, then generate continuous Alex actions that unlatch and open the door to a target angle.

The repository was bootstrapped from an older manipulation codebase. Legacy LEGO task modules, assets, configs, scripts, and tests are archived under `archive/legacy/` for provenance only. Do not restore or extend them for new work unless the user explicitly asks for legacy comparison or migration.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pip install -e ".[vlm]"
pre-commit install
```

## Active Commands

### Door Simulation

```bash
python scripts/validate_door_scene.py
vla-viewer sim/assets/scenes/alex_door_workspace.xml --show-contacts --show-joints
python scripts/validate_mujoco.py
python scripts/validate_assets.py
python scripts/validate_alex_model.py
python scripts/validate_kinematics.py
ALEX_SDK_PATH=../ihmc-alex-sdk python scripts/validate_kinematics.py
python scripts/validate_action_space.py
python scripts/validate_robot_state.py
python scripts/validate_cameras.py
```

### Model Validation

```bash
python scripts/validate_vlm_backbone.py
python scripts/validate_vlm_backbone.py --model-config vlm
python scripts/profile_vlm_memory.py --quick
python scripts/validate_action_head.py
python scripts/validate_action_head.py --model-config vla
```

### Training

```bash
python -m train.trainer trainer=debug cluster=local
python -m train.trainer model=vla_dev cluster=local
python -m train.trainer model=vla cluster=gilbreth
python -m train.trainer trainer.optimizer.lr=1e-5 trainer.training.batch_size_per_device=16
```

### Testing

```bash
pytest
pytest -m "not slow and not gpu"
pytest tests/test_asset_loader.py -v
pytest tests/test_action_space.py -v
pytest tests/test_robot_state.py -v
pytest tests/test_cameras.py -v
pytest tests/test_action_head.py -v
pytest tests/test_vla_model.py -v -m "not slow and not gpu"
```

### Code Quality

```bash
black .
isort .
ruff check .
mypy sim models train eval tracking --ignore-missing-imports
python scripts/validate_configs.py
pre-commit run --all-files
```

## Architecture

- `configs/` - Hydra configuration hierarchy.
- `configs/sim/door.yaml` - active door-opening task contract.
- `models/` - VLM backbone, VLA model assembly, action head, and losses.
- `train/` - trainer entry point and distributed training logic.
- `data/` - dataset and dataloader utilities.
- `sim/` - MuJoCo runtime, Alex action/state contracts, controller, renderer, viewer, and assets.
- `sim/assets/scenes/alex_door_workspace.xml` - active door-opening workspace.
- `archive/legacy/` - historical LEGO baseline material only.
- `tracking/` - W&B tracking with `vla-door-opening` defaults.
- `infra/gilbreth/` - SLURM and HPC setup scripts.

## Configuration Principles

All task, model, and training values should flow through Hydra configs. Prefer `cfg.trainer.optimizer.lr`, `cfg.sim.door.target_open_angle_rad`, or equivalent config access over hardcoded values.

Active simulation work should use door-opening names and metrics:

- handle contact success
- latch release success
- maximum door angle
- final target-angle error
- contact stability
- force/torque limit violations
- recovery success

Avoid introducing new LEGO task references in active docs, commands, configs, tracking tags, or user-facing examples.

## VLA Stack

- Active VLM backbone: `Qwen/Qwen3.5-4B` through `models/vlm_backbone.py`.
- Active VLA assembly: `models/vla_model.py`.
- Frozen action contract: 17-D Alex action chunks, 16 steps at 20 Hz.
- Frozen robot-state contract: 52-D Alex state.
- Action head: flow matching with robot state and noisy action projectors.

## Console Scripts

- `vla-train` - `train.trainer:main`
- `vla-eval` - `eval.evaluate:main`
- `vla-viewer` - `sim.viewer:main`
- `vla-lint-assets` - `sim.asset_linter_cli:main`
- `vla-validate-door` - `scripts.validate_door_scene:main`

## HPC

```bash
sbatch infra/gilbreth/job_templates/01_smoke_1gpu.sh
sbatch infra/gilbreth/job_templates/04_smoke_8gpu_deepspeed.sh
sbatch infra/gilbreth/job_templates/06_smoke_sim_headless.sh
sbatch infra/gilbreth/job_templates/07_download_vlm_weights.sh
sbatch infra/gilbreth/job_templates/09_validate_action_head.sh
```
