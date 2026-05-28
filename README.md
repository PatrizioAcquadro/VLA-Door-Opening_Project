<div align="center">

# VLA-Door-Opening

**Vision-Language-Action System for Robotic Door-Opening Manipulation with IHMC Alex**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Installation](#installation) | [Quick Start](#quick-start) | [Architecture](#architecture) | [Documentation](#documentation)

</div>

---

## Overview

VLA-Door-Opening is a Vision-Language-Action project for contact-rich robotic door opening with the IHMC Alex humanoid upper body. The system focuses on articulated-object manipulation: perceiving a door, handle, hinge, latch state, and current opening angle, then producing continuous robot actions that unlatch and pull or push the door toward a target angle.

The repository was bootstrapped from an older manipulation codebase. The VLA backbone and action head (a flow-matching head derived from the EO-1 architecture), trainer, tracking, cluster, container, and Alex robot abstractions remain useful. Legacy code, assets, scripts, and tests from the prior task are archived under `archive/legacy/` for provenance only.

## Project Goals

- Replicate and adapt the EO-1 Vision-Language-Action architecture for door-opening manipulation.
- Build MuJoCo door, handle, hinge, and latch environments for IHMC Alex.
- Train continuous 17-D Alex action policies with flow matching action chunks.
- Evaluate handle contact, latch release, final door angle, contact stability, force limits, and recovery behavior.
- Transfer stable simulation behaviors toward the IHMC Alex humanoid robot.

## Current Features

- **VLA model stack**: Qwen3.5-4B VLM backbone wrapper, robot-state projector, noisy-action projector, flow matching module, and action output head.
- **Alex simulation contract**: 17-D action space, 52-D robot state, fixed-rate simulation runner, multi-view renderer, and EZGripper abstraction.
- **Door-opening scene scaffold**: `sim/assets/scenes/alex_door_workspace.xml` with a hinged door, lever handle, sensors, and validation script.
- **Hydra configuration**: model, trainer, data, cluster, logging, and door task configs.
- **Experiment tracking**: W&B integration with door-opening project defaults.
- **HPC support**: Gilbreth SLURM templates, DeepSpeed configs, Docker, and Apptainer wrappers.

## Architecture

```
RGB views + language + Alex state
              |
              v
      Qwen3.5-4B VLM backbone
              |
              v
  state/action token sequence assembly
              |
              v
 flow matching action head, 16-step chunks
              |
              v
  17-D Alex command: spine, arms, grippers
              |
              v
      door handle, latch, hinge motion
```

The active task output is a continuous action trajectory for Alex. Success is measured by physical door-opening metrics: handle contact, latch release, final door angle, contact stability, force/torque limits, and recovery.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ for GPU training (container uses cu121)
- MuJoCo for simulation validation
- Git

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pre-commit install
```

Install only VLM dependencies when needed:

```bash
pip install -e ".[vlm]"
```

## Experiment Tracking Setup

W&B is the primary experiment tracker for this project. Metrics, resolved Hydra configs,
GPU stats, throughput, validation losses, and selected artifacts are logged through the
`tracking.ExperimentTracker` integration. Full datasets and checkpoints stay on local or
Gilbreth scratch by default.

One-time local login:

```bash
wandb login
wandb status
```

Recommended environment variables:

```bash
export WANDB_PROJECT=vla-door-opening
export WANDB_ENTITY=<your-wandb-username-or-team>  # optional; omit for default entity
export WANDB_DIR=$PWD/wandb
export HF_HOME=$PWD/cache/huggingface
export PROJECT_ROOT=$PWD
```

On Gilbreth, keep large artifacts on scratch:

```bash
export VLA_SCRATCH_ROOT=/scratch/gilbreth/$USER/vla-door-opening
export WANDB_DIR=$VLA_SCRATCH_ROOT/wandb
export HF_HOME=$VLA_SCRATCH_ROOT/cache/huggingface
```

Use offline mode when a node has no internet access, then sync later:

```bash
WANDB_MODE=offline python -m train.trainer model=base data.dataset.name=dummy trainer=debug cluster=local
wandb sync "$WANDB_DIR"/offline-run-*
```

Checkpoint uploads are disabled unless explicitly requested:

```bash
export WANDB_LOG_MODEL=1  # upload selected checkpoint artifacts
```

## Quick Start

Validate the active door-opening scene:

```bash
python scripts/validate_door_scene.py
vla-viewer sim/assets/scenes/alex_door_workspace.xml --show-contacts --show-joints
```

Run the standard simulation and model checks:

```bash
python scripts/validate_mujoco.py
python scripts/validate_assets.py
python scripts/validate_alex_model.py
python scripts/validate_action_space.py
python scripts/validate_robot_state.py
python scripts/validate_cameras.py
python scripts/validate_action_head.py
```

Run active VLA training with recorded door episode manifests:

```bash
python -m train.trainer model=vla_dev cluster=local data.dataset.path=/path/to/processed
```

Run a transformer-only smoke test explicitly:

```bash
python -m train.trainer model=base data.dataset.name=dummy trainer=debug cluster=local
```

## Testing

```bash
pytest
pytest -m "not slow and not gpu"
pytest tests/test_asset_loader.py -v
pytest tests/test_action_head.py -v
pytest tests/test_vla_model.py -v -m "not slow and not gpu"
```

## Configuration

VLA-Door-Opening uses Hydra configs under `configs/`.

| Config Group | Options | Description |
|--------------|---------|-------------|
| `model` | `base`, `large`, `vlm`, `vlm_dev`, `vla`, `vla_dev` | Model architecture settings |
| `trainer` | `default`, `debug` | Training hyperparameters |
| `data` | `default` | Dataset and dataloader settings |
| `cluster` | `local`, `gilbreth` | Cluster-specific settings |
| `sim` | `default`, `door` | Simulation contracts; `door` is the active task |
| `logging` | `wandb` | Experiment tracking |

Example override:

```bash
python -m train.trainer \
    model=vla_dev \
    trainer.optimizer.lr=1e-5 \
    trainer.training.batch_size_per_device=16
```

## Project Structure

```
VLA-Door-Opening_Project/
├── configs/                 # Hydra configs
│   ├── model/               # Transformer, VLM, and VLA configs
│   ├── sim/                 # Alex control/state/camera and door task configs
│   └── cluster/             # Local and Gilbreth settings
├── data/                    # Dataset and dataloader utilities
├── models/                  # VLM backbone, VLA assembly, action head, losses
├── sim/                     # MuJoCo simulation, Alex control, cameras, assets
│   ├── assets/scenes/       # test, Alex, and door-opening scenes
├── train/                   # Trainer entry point
├── eval/                    # Evaluation entry point
├── tracking/                # W&B experiment tracking
├── infra/gilbreth/          # SLURM and setup scripts
├── scripts/                 # Validation and profiling utilities
├── tests/                   # Unit and integration tests
├── archive/legacy/          # Historical LEGO baseline material
└── docs/                    # Setup, reference, and validation docs
```

## HPC Cluster Usage

```bash
sbatch infra/gilbreth/job_templates/01_smoke_1gpu.sh
sbatch infra/gilbreth/job_templates/04_smoke_8gpu_deepspeed.sh
sbatch infra/gilbreth/job_templates/06_smoke_sim_headless.sh
```

## Containers

Docker:

```bash
./scripts/docker-run.sh python -m train.trainer --help
./scripts/docker-run.sh python -m train.trainer trainer=debug cluster=local
```

Apptainer:

```bash
./scripts/apptainer-run.sh python -m train.trainer cluster=gilbreth
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/door-opening-project-brief.md](docs/door-opening-project-brief.md) | Door-opening task scope and migration map |
| [docs/setup-mujoco.md](docs/setup-mujoco.md) | MuJoCo setup guide |
| [docs/kinematics-validation-report.md](docs/kinematics-validation-report.md) | Alex kinematics validation report |
| [docs/eo1-reference.md](docs/eo1-reference.md) | EO-1 architecture reference (provenance) |
| [docs/git-workflow.md](docs/git-workflow.md) | Git branching and workflow guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

## Roadmap

- [x] Alex action/state/camera simulation contracts
- [x] Qwen3.5-4B VLM wrapper and VLA action head scaffold
- [x] First Alex door-opening workspace scene
- [ ] Door hinge, handle, latch, and contact validation suite
- [ ] Scripted door-opening baseline controller
- [ ] Door-opening demonstration dataset schema and recorder
- [ ] VLA training/evaluation loop on door-opening episodes
- [ ] IHMC Alex transfer validation plan

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [EO-1](https://arxiv.org/abs/2508.21112) - Reference VLA architecture
- [PyTorch](https://pytorch.org/), [Hydra](https://hydra.cc/), and [Weights & Biases](https://wandb.ai/)

---

<div align="center">
  <sub>Vision-Language-Action Door-Opening Project</sub>
  <br>
  <sub>Author: Patrizio Acquadro</sub>
</div>
