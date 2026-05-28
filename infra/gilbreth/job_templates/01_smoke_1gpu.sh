#!/bin/bash
#===============================================================================
# JOB TEMPLATE 1: Single GPU Smoke Test
# Purpose: Verify CUDA, PyTorch, basic training works
# Milestone: "single-GPU forward/backward is stable, checkpoint reloads"
#===============================================================================

#SBATCH --job-name=smoke_1gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=00:30:00

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "SMOKE TEST: 1 GPU"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
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

# Create directories
mkdir -p logs checkpoints

#===============================================================================
# VERIFICATION
#===============================================================================
echo ""
echo "=== Environment Verification ==="
echo "Python: $(which python)"
python --version
echo ""
echo "PyTorch:"
python -c "
import torch
print(f'  Version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  cuDNN version: {torch.backends.cudnn.version()}')
print(f'  Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== nvidia-smi ==="
nvidia-smi

#===============================================================================
# TRAINING TEST
#===============================================================================
echo ""
echo "=== Micro-Training Test ==="

python << 'PYEOF'
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

class DummyModel(nn.Module):
    """Simple model to test forward/backward."""
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        return self.layers(x)

def main():
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # Create model
    model = DummyModel(hidden_size=1024).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    print("\nRunning 100 training steps...")
    start = time.time()

    for step in range(100):
        x = torch.randn(32, 1024, device=device)
        y = torch.randn(32, 1024, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0:
            print(f"  Step {step+1}/100 | Loss: {loss.item():.4f}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")

    # Memory stats
    max_mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"Peak GPU memory: {max_mem:.2f} GB")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/smoke_1gpu.pt"
    checkpoint = {
        "step": 100,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # Verify checkpoint reload
    loaded = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded["model_state_dict"])
    optimizer.load_state_dict(loaded["optimizer_state_dict"])
    print("Checkpoint reload: SUCCESS ✓")

    print("\n" + "="*50)
    print("SMOKE TEST 1 GPU: PASSED ✓")
    print("="*50)

if __name__ == "__main__":
    main()
PYEOF

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "========================================"
