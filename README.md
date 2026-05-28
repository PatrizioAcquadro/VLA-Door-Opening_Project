<div align="center">

# VLA-Door-Opening

**Vision-Language-Action System for Robotic Door-Opening Manipulation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

VLA-Door-Opening is a Master's thesis research project developing a **Vision-Language-Action (VLA) system** for robotic door-opening manipulation. The project reuses the EO-1-style training, simulation, tracking, and robot-control infrastructure from the original VLA-LEGO baseline, but the target task is now articulated-object manipulation: perceiving a handle, selecting a contact strategy, unlatching when needed, and pulling or pushing a door to a target opening angle.

This research is conducted as part of an exchange program between **Politecnico di Milano** and **Purdue University**, under the supervision of Prof. Eugenio Culurciello and Prof. Marcello Restelli.

This local copy is intentionally independent from `PatrizioAcquadro/VLA-LEGO_Project`.

### Research Goals

- Replicate the EO-1 Vision-Language-Action architecture
- Extend the model for manipulation tasks involving doors, handles, hinges, and latches
- Evaluate on LIBERO benchmark (Spatial, Object, Goal, Long subsets)
- Build a door-opening simulation and evaluation stack before transfer to IHMC Alex

## Features

- **EO-1 Architecture** — Unified decoder-only transformer with Qwen 2.5 VL backbone (3B parameters), combining discrete autoregressive decoding with continuous flow matching
- **Door-Opening Manipulation** — Contact-rich articulated-object manipulation with handle grasping, latch rotation, and door-angle control
- **LIBERO Evaluation** — Comprehensive benchmark evaluation across spatial, object, goal, and long-horizon tasks
- **Hydra Configuration** — Hierarchical, composable configs with CLI overrides
- **Distributed Training** — PyTorch DDP with NCCL backend, optimized for 8x A100 GPUs
- **HPC Ready** — Pre-configured for Slurm clusters with container support (Docker + Apptainer)
- **Experiment Tracking** — Weights & Biases integration for logging and visualization
- **Reproducible** — Deterministic training with seed control and checkpoint management

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VLA-Door-Opening Pipeline (EO-1)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────┐    │
│   │   Vision    │    │    Language     │    │      Action         │    │
│   │   Encoder   │───▶│    Reasoning    │───▶│    Generation       │    │
│   │             │    │                 │    │                     │    │
│   │  Qwen 2.5   │    │  Interleaved    │    │  Autoregressive +   │    │
│   │  VL (3B)    │    │  Vision-Text    │    │  Flow Matching      │    │
│   └─────────────┘    └─────────────────┘    └─────────────────────┘    │
│                                                                         │
│   Input: RGB Images + Language Instructions                             │
│   Output: Continuous Action Trajectories (Bimanual)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git

### Setup

```bash
# Clone the new independent repository once it exists
git clone <new-repository-url>
cd VLA-Door-Opening_Project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Quick Start

### Local Training (Debug)

```bash
# Quick smoke test (100 steps, small batch)
python -m train.trainer trainer=debug cluster=local
```

### Full Training

```bash
# Train with base model configuration
python -m train.trainer cluster=local

# Train with large model on GPU cluster
python -m train.trainer model=large cluster=gilbreth
```

### Configuration Overrides

All settings can be overridden from the command line:

```bash
python -m train.trainer \
    model=large \
    trainer.optimizer.lr=1e-5 \
    trainer.training.batch_size_per_device=16 \
    trainer.training.max_steps=50000
```

## Configuration

VLA-Door-Opening uses [Hydra](https://hydra.cc/) for configuration management. All configs are in `configs/`:

| Config Group | Options | Description |
|-------------|---------|-------------|
| `model` | `base`, `large` | Model architecture settings |
| `trainer` | `default`, `debug` | Training hyperparameters |
| `data` | `default` | Dataset and dataloader settings |
| `cluster` | `local`, `gilbreth` | Cluster-specific settings |
| `logging` | `wandb` | Experiment tracking |

### Example: Custom Configuration

```bash
# Combine multiple config overrides
python -m train.trainer \
    model=large \
    trainer=default \
    cluster=local \
    experiment.seed=123 \
    trainer.optimizer.lr=5e-5
```

## Project Structure

```
VLA-Door-Opening_Project/
├── configs/                 # Hydra configuration files
│   ├── config.yaml          # Main config (composes defaults)
│   ├── model/               # Model architectures (base, large)
│   ├── trainer/             # Training settings (default, debug)
│   ├── data/                # Dataset configuration
│   ├── cluster/             # Cluster settings (local, gilbreth)
│   └── logging/             # W&B integration
├── data/                    # Data loading and processing
│   ├── dataset.py           # Dataset classes
│   └── loader.py            # DataLoader utilities
├── models/                  # Model implementations
│   ├── transformer.py       # TransformerModel
│   └── utils.py             # Model utilities
├── train/                   # Training code
│   └── trainer.py           # Main Trainer class
├── eval/                    # Evaluation code
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
├── Dockerfile               # Container definition
└── apptainer.def            # Singularity/Apptainer definition
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type checking
mypy sim models train eval --ignore-missing-imports

# Run all checks (pre-commit)
pre-commit run --all-files
```

### Validating Configurations

```bash
# Validate all config combinations
python scripts/validate_configs.py
```

## HPC Cluster Usage

### Gilbreth (Purdue)

```bash
# Submit training job
sbatch scripts/train.sh

# Interactive session
sinteractive -A <account> -n 1 -g 1 -t 4:00:00

# Load container and run
apptainer exec --nv vla-door-opening.sif python -m train.trainer cluster=gilbreth
```

See [docs/git-workflow.md](docs/git-workflow.md) for detailed cluster instructions.

## Running with Containers (Deps-Only Model)

Container images contain **only dependencies** (CUDA, Python, PyTorch, etc.). Your code is bind-mounted at runtime from your git checkout.

### Why deps-only?

- **No rebuilds for code changes** — `git pull` updates your code instantly
- **Reproducibility** — run = (git commit) + (image digest/tag)
- **Smaller images** — no repo code baked in

### Docker (Lab PC)

```bash
# Using wrapper script (recommended)
./scripts/docker-run.sh python -m train.trainer --help
./scripts/docker-run.sh python -m train.trainer trainer=debug cluster=local

# Or directly with docker
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    ghcr.io/patrizioacquadro/vla-door-opening_project:latest \
    python -m train.trainer cluster=local
```

### Apptainer (HPC Cluster)

```bash
# Download image once (or use release artifact)
apptainer pull vla-door-opening.sif docker://ghcr.io/patrizioacquadro/vla-door-opening_project:latest

# Using wrapper script (recommended)
./scripts/apptainer-run.sh python -m train.trainer cluster=gilbreth
```

### Reproducibility

Each container run prints git commit, Python version, and GPU info. This output is saved to `/tmp/vla_run_info.txt` inside the container. Record this for experiment tracking:

```
=== VLA-Door-Opening Container Run ===
Timestamp: 2024-01-15T10:30:00+00:00
Python: Python 3.10.12
Git commit: abc1234...
Git branch: main
Git dirty: 0 files
PyTorch: 2.2.0
CUDA: True
GPU: NVIDIA A100-SXM4-80GB
==============================
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/git-workflow.md](docs/git-workflow.md) | Git branching and workflow guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

## Roadmap

- [ ] EO-1 architecture implementation (Qwen 2.5 VL backbone)
- [ ] Data pipeline for LIBERO benchmark
- [ ] Flow matching action head integration
- [ ] Bimanual action space extension
- [ ] LIBERO benchmark evaluation
- [ ] IHMC Alex deployment

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a PR.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Academic Institutions
- **Politecnico di Milano** — Primary institution
- **Purdue University** — Exchange program host

### Advisors
- **Prof. Eugenio Culurciello** — Purdue University
- **Prof. Marcello Restelli** — Politecnico di Milano

### Technical Foundations
- [EO-1: A Unified Model for Embodied AI](https://arxiv.org/abs/2508.21112) — Base architecture
- [Lerobot](https://github.com/huggingface/lerobot) — Training framework
- [LIBERO](https://libero-project.github.io/) — Evaluation benchmark
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Hydra](https://hydra.cc/) — Configuration management
- [Weights & Biases](https://wandb.ai/) — Experiment tracking

---

<div align="center">
  <sub>Master's Thesis Research — Politecnico di Milano / Purdue University</sub>
  <br>
  <sub>Author: Patrizio Acquadro</sub>
</div>
