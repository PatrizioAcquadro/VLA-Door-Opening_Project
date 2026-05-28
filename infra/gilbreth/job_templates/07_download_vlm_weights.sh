#!/bin/bash
#===============================================================================
# JOB TEMPLATE 7: Download VLM Weights (Qwen3.5-4B)
# Purpose: Pre-cache ~8 GB model weights on scratch before GPU training jobs
# Milestone: "model config loads, all weight files present in HF cache"
# Phase: 3.1.1
#
# No GPU needed — this is a download-only job. Run this once before any
# GPU job that uses model=vlm or model=vlm_dev to avoid download timeouts
# during GPU-allocated training runs.
#
# Requires:
#   - pip install -e ".[vlm]" in your environment (transformers, accelerate)
#   - Outbound internet access on the compute node (or proxy configured)
#   - HF_HOME set to your scratch cache directory
#
# After this job completes, verify with:
#   python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('Qwen/Qwen3.5-4B'); print(c.text_config.hidden_size)"
# Expected output: 2560
#===============================================================================

#SBATCH --job-name=download_vlm_weights
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=euge
#SBATCH --partition=shared
#SBATCH --qos=standby

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "JOB: Download VLM Weights (Phase 3.1.1)"
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
# HF_HOME should point to scratch. Default fallback uses project root.
# To override: export HF_HOME=/scratch/gilbreth/<user>/cache/huggingface
# before submitting this job.
SCRATCH_ROOT=${VLA_SCRATCH_ROOT:-/scratch/gilbreth/$(whoami)/vla-lego}
export HF_HOME=${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}
export TRANSFORMERS_CACHE=${HF_HOME}

echo ""
echo "=== Cache Directory ==="
echo "HF_HOME: $HF_HOME"
mkdir -p "$HF_HOME"

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
python -c "
import transformers, accelerate, sentencepiece
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'sentencepiece: {sentencepiece.__version__}')
"

#===============================================================================
# DOWNLOAD WEIGHTS
#===============================================================================
echo ""
echo "=== Downloading Qwen3.5-4B Weights (~8 GB) ==="
echo "This may take 10-30 minutes depending on network speed."
echo ""

python << 'PYEOF'
import os
import time
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = os.environ["HF_HOME"]

print(f"Downloading to: {CACHE_DIR}")
print(f"Model: {MODEL_ID}")
print()

# Step 1: Download config (fast, ~few KB)
print("Step 1/3: Downloading model config...")
t0 = time.time()
config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
print(f"  Config loaded in {time.time() - t0:.1f}s")
print(f"  hidden_size: {config.text_config.hidden_size}")
print(f"  num_hidden_layers: {config.text_config.num_hidden_layers}")
print(f"  vocab_size: {config.text_config.vocab_size}")

# Step 2: Download processor (tokenizer + image preprocessor, ~few MB)
print()
print("Step 2/3: Downloading processor...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
print(f"  Processor loaded in {time.time() - t0:.1f}s")
print(f"  Processor type: {type(processor).__name__}")

# Step 3: Download model weights (~8 GB)
print()
print("Step 3/3: Downloading model weights (~8 GB)...")
print("  (This is the slow step — please wait)")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # CPU only for download; no GPU needed
)
elapsed = time.time() - t0
print(f"  Model loaded in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

# Compute and report param count
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params / 1e9:.2f}B")

# Verify hidden_size
assert config.text_config.hidden_size == 2560, (
    f"Unexpected hidden_size: {config.text_config.hidden_size}"
)
print()
print("Download complete. All checks passed.")
print(f"  hidden_size: {config.text_config.hidden_size} (expected 2560)")
print(f"  param count: {total_params / 1e9:.2f}B (expected ~4.54B)")

PYEOF

#===============================================================================
# VERIFY CACHE
#===============================================================================
echo ""
echo "=== Verifying Cache Contents ==="

python << 'PYEOF'
import os
from pathlib import Path

hf_home = Path(os.environ["HF_HOME"])
model_dir = hf_home / "models--Qwen--Qwen3.5-4B"

if not model_dir.exists():
    # HuggingFace Hub may use a different directory structure
    # Search for any Qwen3 model cache
    matches = list(hf_home.rglob("*Qwen3.5-4B*"))
    if matches:
        print(f"  [OK] Found Qwen3.5-4B cache at: {matches[0]}")
    else:
        print("  [WARN] Could not find expected cache directory. Model may be cached elsewhere.")
        print(f"  HF_HOME: {hf_home}")
        print(f"  Contents: {list(hf_home.iterdir()) if hf_home.exists() else 'directory does not exist'}")
else:
    # Check snapshot files
    snapshots = list((model_dir / "snapshots").glob("*/*.safetensors")) if (model_dir / "snapshots").exists() else []
    if snapshots:
        total_size_gb = sum(f.stat().st_size for f in snapshots) / 1e9
        print(f"  [OK] Found {len(snapshots)} weight file(s), total {total_size_gb:.1f} GB")
    else:
        print(f"  [INFO] Cache directory exists: {model_dir}")
        print("  Weights may be in a different format or location within the cache.")

print()
print("Weight pre-download complete.")
print("Next steps:")
print("  1. Submit job 08: sbatch infra/gilbreth/job_templates/08_profile_vlm_memory.sh")
print("  2. Or run training: sbatch infra/gilbreth/job_templates/01_smoke_1gpu.sh (with model=vlm)")

PYEOF

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "Cache: $HF_HOME"
echo "========================================"
