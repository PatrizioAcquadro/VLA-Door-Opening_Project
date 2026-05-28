# Structure Migration Report

**Date:** January 26, 2026
**Purpose:** Consolidate VLA-Project_Gilbreth (legacy) and VLA-LEGO_Project (canonical) into a single cross-device repository

## Summary

This migration creates a unified "golden" repository that works across Mac, lab Linux, and the Gilbreth HPC cluster, with heavy artifacts stored on SCRATCH.

## Before vs After

### Before
```
/home/pacquadr/
├── VLA-Project_Gilbreth/    # Legacy folder (336K)
│   ├── CLAUDE.md
│   ├── .claude/
│   ├── GILBRETH_TRAINING_GUIDE.md
│   ├── Phase0/
│   │   ├── tracking/       # Experiment tracking module
│   │   ├── scripts/        # Setup scripts
│   │   ├── job_templates/  # SLURM job templates
│   │   ├── configs/        # DeepSpeed configs
│   │   └── wandb/          # W&B run data
│   └── Phase1/
│       ├── Phase1.1.md
│       └── Phase1.2.md
│
└── VLA-LEGO_Project/        # Canonical repo (245K)
    ├── models/, sim/, eval/, data/, train/, tests/
    ├── configs/
    └── docs/

/scratch/gilbreth/pacquadr/
└── worldsim/                # Old scratch layout
    ├── checkpoints/ (37GB)
    ├── logs/
    └── cache/
```

### After
```
/home/pacquadr/
└── VLA-LEGO_Project/        # Unified canonical repo
    ├── models/, sim/, eval/, data/, train/, tests/
    ├── tracking/            # NEW: Migrated from legacy
    ├── infra/gilbreth/      # NEW: Cluster infrastructure
    │   ├── job_templates/
    │   └── scripts/
    ├── configs/
    │   ├── cluster/         # Updated gilbreth.yaml
    │   └── deepspeed/       # NEW: DeepSpeed configs
    ├── docs/
    │   ├── roadmap/         # NEW: Phase documentation
    │   └── cluster/         # NEW: Training guide
    ├── .env.example         # NEW: Environment template
    └── [symlinks to SCRATCH]
        ├── runs -> /scratch/.../vla-lego/runs
        ├── checkpoints -> /scratch/.../vla-lego/checkpoints
        ├── logs -> /scratch/.../vla-lego/logs
        ├── wandb -> /scratch/.../vla-lego/wandb
        └── cache -> /scratch/.../vla-lego/cache

/scratch/gilbreth/pacquadr/
└── vla-lego/                # NEW: Renamed scratch layout
    ├── runs/
    ├── checkpoints/         # Migrated from worldsim
    ├── logs/                # Migrated from worldsim
    ├── wandb/               # Migrated from legacy
    ├── cache/               # Migrated from worldsim
    ├── datasets/
    ├── renders/
    └── archive/             # Backup of legacy folder
```

## Migrations Performed

### Code/Configs → Canonical Repo

| Source | Destination | Files |
|--------|-------------|-------|
| `VLA-Project_Gilbreth/Phase0/tracking/*.py` | `VLA-LEGO_Project/tracking/` | 6 Python files + README |
| `VLA-Project_Gilbreth/Phase0/job_templates/*.sh` | `VLA-LEGO_Project/infra/gilbreth/job_templates/` | 5 SLURM scripts |
| `VLA-Project_Gilbreth/Phase0/scripts/*.sh` | `VLA-LEGO_Project/infra/gilbreth/scripts/` | 3 setup scripts |
| `VLA-Project_Gilbreth/Phase0/configs/deepspeed_*.json` | `VLA-LEGO_Project/configs/deepspeed/` | 2 DeepSpeed configs |
| `VLA-Project_Gilbreth/Phase0/configs/tracking_config.yaml` | `VLA-LEGO_Project/configs/tracking.yaml` | Tracking config |
| `VLA-Project_Gilbreth/GILBRETH_TRAINING_GUIDE.md` | `VLA-LEGO_Project/docs/cluster/` | Training guide (2100+ lines) |
| `VLA-Project_Gilbreth/Phase0/Phase0.2.md` | `VLA-LEGO_Project/docs/roadmap/phase0.2.md` | Phase 0.2 doc |
| `VLA-Project_Gilbreth/Phase1/Phase1.*.md` | `VLA-LEGO_Project/docs/roadmap/` | Phase 1.1 and 1.2 docs |

### Artifacts → SCRATCH

| Source | Destination | Size |
|--------|-------------|------|
| `/scratch/.../worldsim/checkpoints/` | `/scratch/.../vla-lego/checkpoints/` | ~37 GB |
| `/scratch/.../worldsim/logs/` | `/scratch/.../vla-lego/logs/` | ~804 KB |
| `/scratch/.../worldsim/cache/` | `/scratch/.../vla-lego/cache/` | ~8 KB |
| `VLA-Project_Gilbreth/Phase0/wandb/` | `/scratch/.../vla-lego/wandb/` | W&B runs |

## Excluded from Git

### Never Committed (Security/Privacy)

| File/Directory | Reason |
|----------------|--------|
| `CLAUDE.md` | Claude Code instructions - private |
| `.claude/` | Claude local settings - private |
| `.env` | May contain API keys |
| `*.pem`, `*.key` | Private keys |
| `credentials.json` | Credentials |

### Symlinked to SCRATCH (Too Large for Git)

| Directory | Points To |
|-----------|-----------|
| `runs/` | `/scratch/gilbreth/pacquadr/vla-lego/runs/` |
| `checkpoints/` | `/scratch/gilbreth/pacquadr/vla-lego/checkpoints/` |
| `logs/` | `/scratch/gilbreth/pacquadr/vla-lego/logs/` |
| `wandb/` | `/scratch/gilbreth/pacquadr/vla-lego/wandb/` |
| `cache/` | `/scratch/gilbreth/pacquadr/vla-lego/cache/` |

## Security Checks Performed

1. **Searched for secrets in both directories:**
   - No `.env` files with real values found
   - No API keys or tokens detected
   - No SSH keys or credentials found
   - No `*.pem` or `*.key` files present

2. **Verified .gitignore coverage:**
   - Added `CLAUDE.md` and `.claude/` exclusions
   - Added all symlinked artifact directories
   - Added additional secret patterns (`*.token`, `credentials.json`)

3. **Template created instead of real configs:**
   - `.env.example` contains only placeholders, not real values

## Config Updates Made

### configs/cluster/gilbreth.yaml

**Changes:**
1. **Removed incorrect `socket_ifname: "ib0"`** - Interface names vary by node
2. **Added single-node vs multi-node NCCL settings** - Different configs needed
3. **Updated paths to use `vla-lego`** instead of `worldsim`
4. **Added environment variable support** - `VLA_SCRATCH_ROOT` fallback
5. **Updated modules** - Correct versions: external, cuda/12.1.1, anaconda/2024.10-py312
6. **Added launcher configuration** - Documents srun+torchrun requirement

## Deletions/Deprecations

| Item | Action | Reason |
|------|--------|--------|
| `/scratch/.../worldsim/` | Will be emptied | Renamed to vla-lego |
| `VLA-Project_Gilbreth/` | Backed up, then deleted | Consolidated into canonical repo |
| `__pycache__/` directories | Not migrated | Generated files |

## Cross-Device Workflow

### On Mac/Lab Linux
```bash
export VLA_PROJECT_ROOT=$PWD
export VLA_SCRATCH_ROOT=$HOME/vla-lego-data
mkdir -p $VLA_SCRATCH_ROOT/{runs,checkpoints,logs,wandb,cache}
```

### On Gilbreth HPC
```bash
export VLA_PROJECT_ROOT=$HOME/VLA-LEGO_Project
export VLA_SCRATCH_ROOT=/scratch/gilbreth/$USER/vla-lego
# Symlinks already created
```

## Verification Checklist

- [x] SCRATCH directory structure created
- [x] Artifacts migrated from worldsim to vla-lego
- [x] W&B data moved from legacy to SCRATCH
- [x] Tracking module copied and functional
- [x] Job templates and scripts copied
- [x] DeepSpeed configs copied
- [x] Phase docs moved to docs/roadmap/
- [x] Training guide moved to docs/cluster/
- [x] .gitignore updated with exclusions
- [x] .env.example created
- [x] gilbreth.yaml updated with correct settings
- [x] Symlinks created and verified
- [x] No sensitive files in git staging area

## Files Created

| Path | Purpose |
|------|---------|
| `tracking/__init__.py` | Tracking module entry point |
| `tracking/experiment.py` | ExperimentTracker class |
| `tracking/metrics.py` | Metric utilities |
| `tracking/metadata.py` | Reproducibility metadata |
| `tracking/naming.py` | Run naming conventions |
| `tracking/gpu_monitor.py` | GPU monitoring utilities |
| `tracking/README.md` | Tracking module documentation |
| `infra/gilbreth/README.md` | Cluster infra documentation |
| `infra/gilbreth/job_templates/*.sh` | 5 SLURM job templates |
| `infra/gilbreth/scripts/*.sh` | 3 setup scripts |
| `configs/deepspeed/zero1.json` | DeepSpeed ZeRO-1 config |
| `configs/deepspeed/zero1_fp16.json` | DeepSpeed FP16 config |
| `configs/tracking.yaml` | Tracking configuration |
| `docs/roadmap/phase0.2.md` | Phase 0.2 documentation |
| `docs/roadmap/phase1.1.md` | Phase 1.1 documentation |
| `docs/roadmap/phase1.2.md` | Phase 1.2 documentation |
| `docs/cluster/gilbreth-training-guide.md` | Comprehensive cluster guide |
| `.env.example` | Environment variable template |
| `docs/structure_migration_report.md` | This report |

## Next Steps

1. Review git status and stage changes
2. Create commit with descriptive message
3. Verify legacy folder backup exists
4. Remove legacy folder from HOME
5. Test job submission to verify paths work
