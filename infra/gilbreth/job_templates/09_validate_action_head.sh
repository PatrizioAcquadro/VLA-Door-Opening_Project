#!/bin/bash
#===============================================================================
# JOB TEMPLATE 9: Validate Action Head (Phase 3.2.5)
# Purpose: End-to-end validation of all Phase 3.2 action head components + VLA model
# Milestone: "10/10 checks passed, VRAM overhead < 1 GB, artifacts in logs/action_head/"
# Phase: 3.2.5
#
# Runs scripts/validate_action_head.py with the full Qwen3.5-4B backbone on
# an A100 GPU. Verifies: config parsing, component instantiation, flow matching
# math, projector/output head shapes, VLA forward pass, ODE inference, gradient
# routing, numerical stability, and VRAM overhead.
#
# Requires:
#   - Weights pre-cached by job 07 (sbatch .../07_download_vlm_weights.sh)
#   - pip install -e ".[vlm]" in your environment (transformers, accelerate)
#
# After this job completes, verify with:
#   cat logs/action_head/validation_report.json | python -c "import sys,json; r=json.load(sys.stdin); print(r['summary'])"
# Expected: {'passed': 10, 'total': 10, 'all_passed': True}
#===============================================================================

#SBATCH --job-name=validate_action_head
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "JOB: Validate Action Head (Phase 3.2.5)"
echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "========================================"

# Load environment
cd $SLURM_SUBMIT_DIR
source /scratch/gilbreth/$(whoami)/vla-lego/activate_env.sh 2>/dev/null || {
    echo "Activating environment manually..."
    module purge
    module load external
    module load cuda/12.1.1 anaconda/2024.10-py312
    conda activate vla_lego_env 2>/dev/null || source activate vla_lego_env
}

mkdir -p logs

#===============================================================================
# CACHE DIRECTORY
#===============================================================================
SCRATCH_ROOT=${VLA_SCRATCH_ROOT:-/scratch/gilbreth/$(whoami)/vla-lego}
export HF_HOME=${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${HF_HOME}

echo ""
echo "=== Cache Directory ==="
echo "HF_HOME: $HF_HOME"

# Verify weights are pre-cached (job 07 must have run first)
if [ ! -d "$HF_HOME" ]; then
    echo "[ERROR] HF_HOME not found: $HF_HOME"
    echo "  Run job 07 first: sbatch infra/gilbreth/job_templates/07_download_vlm_weights.sh"
    exit 1
fi

#===============================================================================
# VLM DEPENDENCIES
#===============================================================================
echo ""
echo "=== Ensuring VLM dependencies ==="
python -c "import transformers; print(f'transformers already installed: {transformers.__version__}')" 2>/dev/null || {
    echo "Installing VLM deps..."
    pip install --quiet -e ".[vlm]"
    python -c "import transformers; print(f'transformers installed: {transformers.__version__}')"
}

#===============================================================================
# ENVIRONMENT VERIFICATION
#===============================================================================
echo ""
echo "=== Environment Verification ==="
echo "Python: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
python -c "
import torch, transformers, accelerate
print(f'torch: {torch.__version__}  (CUDA: {torch.version.cuda})')
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
"

#===============================================================================
# RUN VALIDATION
#===============================================================================
echo ""
echo "=== Running Action Head Validation (10 checks) ==="
echo ""

python scripts/validate_action_head.py --model-config vla

VALIDATION_EXIT=$?

#===============================================================================
# VERIFY ARTIFACTS
#===============================================================================
echo ""
echo "=== Verifying Artifacts ==="

python << 'PYEOF'
import json
from pathlib import Path

out_dir = Path("logs/action_head")
all_ok = True

# Check validation_report.json
report_path = out_dir / "validation_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    print(f"  [OK] validation_report.json: {summary['passed']}/{summary['total']} passed")
    if not summary["all_passed"]:
        all_ok = False
        print("  [FAIL] Not all checks passed:")
        for ch in report["checks"]:
            if not ch["passed"]:
                print(f"    - {ch['name']}: {ch['detail']}")
else:
    print(f"  [WARN] validation_report.json not found at {report_path}")
    all_ok = False

# Check param_counts.json
param_path = out_dir / "param_counts.json"
if param_path.exists():
    params = json.loads(param_path.read_text())
    total_m = params["total_action_head"] / 1e6
    print(f"  [OK] param_counts.json: total action head = {total_m:.1f}M params")
else:
    print(f"  [WARN] param_counts.json not found")

# Check memory_overhead.json (GPU only)
mem_path = out_dir / "memory_overhead.json"
if mem_path.exists():
    mem = json.loads(mem_path.read_text())
    print(f"  [OK] memory_overhead.json: overhead={mem['overhead_gb']:.3f} GB, "
          f"within_budget={mem['within_budget']}")
    if not mem["within_budget"]:
        all_ok = False
else:
    print("  [INFO] memory_overhead.json not present (GPU check skipped or not run)")

if all_ok:
    print()
    print("Validation artifacts: PASS")
else:
    print()
    print("Validation artifacts: FAIL — see above for details")
    import sys
    sys.exit(1)
PYEOF

ARTIFACT_EXIT=$?

#===============================================================================
# JOB COMPLETE
#===============================================================================
echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "Artifacts: logs/action_head/"
echo "========================================"

# Return failure if either the validation or artifact check failed
if [ $VALIDATION_EXIT -ne 0 ] || [ $ARTIFACT_EXIT -ne 0 ]; then
    exit 1
fi
exit 0
