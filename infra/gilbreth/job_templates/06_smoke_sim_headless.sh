#!/bin/bash
#===============================================================================
# JOB TEMPLATE 6: Headless Simulation Smoke Test
# Purpose: Verify MuJoCo headless rendering on cluster (same artifacts as lab PC)
# Milestone: "smoke_video.mp4, frames/, sim_smoke_meta.json reproduced on HPC"
# Phase: 0.2.6
#
# ThinLinc policy: This job is fully headless — no GUI needed.
# Use ThinLinc only as a last resort to diagnose visual bugs that cannot be
# identified from saved videos/frames.
#===============================================================================

#SBATCH --job-name=smoke_sim_headless
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
#SBATCH --time=00:15:00

#===============================================================================
# SETUP
#===============================================================================
set -e

echo "========================================"
echo "SMOKE TEST: Headless Sim (Phase 0.2.6)"
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
mkdir -p logs

#===============================================================================
# SIM DEPENDENCIES
#===============================================================================
# The base conda env (01_setup_env.sh) doesn't include mujoco or imageio.
# Install them if missing. This is idempotent — skips if already present.
echo ""
echo "=== Ensuring sim dependencies ==="
python -c "import mujoco; print(f'mujoco already installed: {mujoco.__version__}')" 2>/dev/null || {
    echo "Installing mujoco and imageio[ffmpeg]..."
    pip install --quiet "mujoco>=3.1.0,<4.0.0" "imageio[ffmpeg]>=2.31.0"
    python -c "import mujoco; print(f'mujoco installed: {mujoco.__version__}')"
}

#===============================================================================
# RENDERING BACKEND
#===============================================================================
echo ""
echo "=== Rendering Backend ==="

# Use NVIDIA EGL on GPU nodes (available via NVIDIA driver on A100 nodes).
# No system packages needed — EGL is provided by libEGL_nvidia.so.
#
# Fallback: If EGL fails on your node, use the Apptainer container instead:
#   ./scripts/apptainer-run.sh python scripts/validate_sim_smoke.py
# The container sets MUJOCO_GL=osmesa with libosmesa6-dev pre-installed.
export MUJOCO_GL=egl

# Quick render probe — fail fast if EGL is broken
python -c "
import os
print(f'MUJOCO_GL={os.environ.get(\"MUJOCO_GL\", \"not set\")}')
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')

# Minimal render test
xml = '<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"plane\" size=\"1 1 0.1\"/></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
img = r.render()
assert img.shape == (64, 64, 3), f'Unexpected shape: {img.shape}'
assert img.sum() > 0, 'Render produced black image'
r.close()
print('EGL rendering backend: OK')
"

#===============================================================================
# VERIFICATION
#===============================================================================
echo ""
echo "=== Environment Verification ==="
echo "Python: $(which python)"
python --version
echo ""
python -c "
import mujoco, imageio
print(f'MuJoCo: {mujoco.__version__}')
print(f'imageio: {imageio.__version__}')
"
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

#===============================================================================
# RUN HEADLESS SIM SMOKE TEST
#===============================================================================
# Disable W&B by default. To attach artifacts to W&B, set WANDB_MODE=online.
export WANDB_MODE=disabled

echo ""
echo "=== Running validate_sim_smoke.py ==="

python scripts/validate_sim_smoke.py

#===============================================================================
# VERIFY ARTIFACTS
#===============================================================================
echo ""
echo "=== Verifying Artifacts ==="

python << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("logs/sim_smoke")
errors = []

# Check video
video = output_dir / "smoke_video.mp4"
if video.exists() and video.stat().st_size > 0:
    print(f"  [OK] Video: {video} ({video.stat().st_size / 1024:.1f} KB)")
else:
    errors.append(f"Video missing or empty: {video}")
    print(f"  [FAIL] Video: {video}")

# Check frames
frames_dir = output_dir / "frames"
if frames_dir.exists():
    pngs = list(frames_dir.glob("*.png"))
    if len(pngs) > 0:
        print(f"  [OK] Frames: {len(pngs)} PNGs in {frames_dir}")
    else:
        errors.append("No PNG frames found")
        print(f"  [FAIL] Frames: 0 PNGs")
else:
    errors.append(f"Frames directory missing: {frames_dir}")
    print(f"  [FAIL] Frames dir missing")

# Check metadata
meta = output_dir / "sim_smoke_meta.json"
if meta.exists():
    data = json.loads(meta.read_text())
    passed = all(
        r.get("passed", False)
        for r in data.get("results", {}).values()
    )
    status = "OK" if passed else "FAIL"
    print(f"  [{status}] Metadata: {meta} (all_passed={passed})")
    if not passed:
        errors.append("Some smoke checks failed (see metadata)")
else:
    errors.append(f"Metadata missing: {meta}")
    print(f"  [FAIL] Metadata missing")

if errors:
    print(f"\nARTIFACT VERIFICATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)
else:
    print("\nAll artifacts verified successfully.")
PYEOF

echo ""
echo "========================================"
echo "JOB COMPLETE: $(date)"
echo "Artifacts: logs/sim_smoke/"
echo "========================================"
