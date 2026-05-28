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

VLA-Door-Opening is a Master's thesis research project for contact-rich robotic door opening with the IHMC Alex humanoid upper body. The active task is articulated-object manipulation: perceiving a door, handle, hinge, latch state, and current opening angle, then producing continuous robot actions that unlatch and pull or push the door toward a target angle.

The repository was bootstrapped from an older manipulation codebase. The EO-1-style VLA backbone, action head, trainer, tracking, cluster, container, and Alex robot abstractions remain useful. Legacy task modules under `sim/lego`, related assets, and `tests/test_lego_*` are retained only as baseline material until door-opening replacements are complete.

## Research Goals

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

The active task output is a continuous action trajectory for Alex. Success is measured by physical door-opening metrics, not LEGO assembly metrics.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ for GPU training
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

Run local debug training:

```bash
python -m train.trainer trainer=debug cluster=local
```

Train with the VLA model configuration:

```bash
python -m train.trainer model=vla_dev cluster=local
python -m train.trainer model=vla cluster=gilbreth
```

## Testing

```bash
pytest
pytest -m "not slow and not gpu"
pytest tests/test_asset_loader.py -v
pytest tests/test_action_head.py -v
pytest tests/test_vla_model.py -v -m "not slow and not gpu"
```

Legacy LEGO tests remain available for regression checks while that baseline code is still present:

```bash
pytest tests/test_lego_bricks.py tests/test_lego_contacts.py tests/test_lego_task.py -v
```

## Configuration

VLA-Door-Opening uses Hydra configs under `configs/`.

| Config Group | Options | Description |
|--------------|---------|-------------|
| `model` | `base`, `large`, `vlm`, `vlm_dev`, `vla`, `vla_dev` | Model architecture settings |
| `trainer` | `default`, `debug` | Training hyperparameters |
| `data` | `default` | Dataset and dataloader settings |
| `cluster` | `local`, `gilbreth` | Cluster-specific settings |
| `sim` | `default`, `door`, `lego` | Simulation contracts; `lego` is legacy |
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
│   └── lego/                # Legacy baseline implementation
├── train/                   # Trainer entry point
├── eval/                    # Evaluation entry point
├── tracking/                # W&B experiment tracking
├── infra/gilbreth/          # SLURM and setup scripts
├── scripts/                 # Validation and profiling utilities
├── tests/                   # Unit and integration tests
└── docs/                    # Roadmaps, reports, setup notes
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
| [docs/roadmap/phase1.2.md](docs/roadmap/phase1.2.md) | Door environment roadmap |
| [docs/phase3.1-3.2-report.md](docs/phase3.1-3.2-report.md) | VLM and action-head architecture report |
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

- **Politecnico di Milano** - Primary institution
- **Purdue University** - Exchange program host
- **Prof. Eugenio Culurciello** - Purdue University
- **Prof. Marcello Restelli** - Politecnico di Milano
- [EO-1](https://arxiv.org/abs/2508.21112) - Reference VLA architecture
- [PyTorch](https://pytorch.org/), [Hydra](https://hydra.cc/), and [Weights & Biases](https://wandb.ai/)

---

<div align="center">
  <sub>Master's Thesis Research - Politecnico di Milano / Purdue University</sub>
  <br>
  <sub>Author: Patrizio Acquadro</sub>
</div>
