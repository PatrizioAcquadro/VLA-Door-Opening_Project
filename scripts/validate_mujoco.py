"""Phase 0.2.1 validation script: MuJoCo runtime installation check.

Run:
    python scripts/validate_mujoco.py

Checks:
    1. mujoco imports successfully
    2. Minimal MJCF loads
    3. Deterministic stepping (1000 steps, 3 trials)
    4. Prints environment metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Run all Phase 0.2.1 validation checks. Returns 0 on success, 1 on failure."""
    print("=" * 60)
    print("Phase 0.2.1 Validation: MuJoCo Runtime Installation")
    print("=" * 60)

    # Check 1: Import
    print("\n[1/4] Checking mujoco import...")
    try:
        import mujoco

        print(f"  OK: mujoco {mujoco.__version__}")
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  Hint: run `pip install -e '.[sim]'`")
        return 1

    # Check 2: MJCF load
    print("\n[2/4] Loading minimal MJCF scene...")
    from sim.mujoco_env import load_model

    scene_path = PROJECT_ROOT / "sim" / "assets" / "scenes" / "test_scene.xml"
    try:
        model = load_model(scene_path)
        print(f"  OK: model loaded (nq={model.nq}, nv={model.nv}, timestep={model.opt.timestep})")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 3: Deterministic stepping
    print("\n[3/4] Validating deterministic stepping (1000 steps x 3 trials)...")
    from sim.mujoco_env import check_deterministic

    try:
        is_deterministic = check_deterministic(scene_path, n_steps=1000, n_trials=3)
        if is_deterministic:
            print("  OK: stepping is deterministic")
        else:
            print("  FAIL: non-deterministic stepping detected")
            return 1
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 4: Environment metadata
    print("\n[4/4] Collecting environment metadata...")
    from sim.env_meta import collect_metadata

    meta = collect_metadata(PROJECT_ROOT)
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # Save metadata to JSON for records
    meta_path = PROJECT_ROOT / "logs"
    meta_path.mkdir(exist_ok=True)
    meta_file = meta_path / "mujoco_env_meta.json"
    # Convert non-serializable values
    serializable_meta = {
        k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()
    }
    meta_file.write_text(json.dumps(serializable_meta, indent=2))
    print(f"\n  Metadata saved to {meta_file}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
