"""Phase 0.2.5 validation script: Asset Pathing & Loader Contract.

Run:
    python scripts/validate_assets.py

Checks:
    1. Canonical directory layout exists
    2. Asset linter passes on all MJCF files (no errors)
    3. load_scene("test_scene") loads successfully
    4. All scenes under sim/assets/scenes/ load without error
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Run all Phase 0.2.5 validation checks. Returns 0 on success, 1 on failure."""
    print("=" * 60)
    print("Phase 0.2.5 Validation: Asset Pathing & Loader Contract")
    print("=" * 60)

    # Check 1: Directory layout
    print("\n[1/4] Checking canonical directory layout...")
    from sim.asset_loader import ASSETS_DIR, ROBOTS_DIR, SCENES_DIR

    layout_ok = True
    for d, name in [(ASSETS_DIR, "assets"), (SCENES_DIR, "scenes"), (ROBOTS_DIR, "robots")]:
        if d.is_dir():
            print(f"  OK: {name}/ exists")
        else:
            print(f"  FAIL: {name}/ missing ({d})")
            layout_ok = False

    if not layout_ok:
        return 1

    # Check 2: Asset linter
    print("\n[2/4] Running asset linter on all MJCF files...")
    from sim.asset_linter import Severity, lint_mjcf

    mjcf_files = sorted(ASSETS_DIR.rglob("*.xml"))
    total_errors = 0
    total_warnings = 0

    if not mjcf_files:
        print("  WARNING: No MJCF files found under sim/assets/")
    else:
        for f in mjcf_files:
            issues = lint_mjcf(f)
            rel = f.relative_to(ASSETS_DIR)
            if not issues:
                print(f"  OK: {rel} — clean")
            else:
                for issue in issues:
                    prefix = issue.severity.value
                    print(f"  [{prefix}] {rel}: {issue.message}")
                    if issue.severity == Severity.ERROR:
                        total_errors += 1
                    else:
                        total_warnings += 1

        print(
            f"  Linted {len(mjcf_files)} file(s): {total_errors} error(s), {total_warnings} warning(s)"
        )

    if total_errors > 0:
        print("  FAIL: linter found errors")
        return 1

    # Check 3: load_scene entrypoint
    print('\n[3/4] Testing load_scene("test_scene")...')
    try:
        from sim.asset_loader import load_scene

        model = load_scene("test_scene")
        print(f"  OK: loaded (nq={model.nq}, nv={model.nv}, timestep={model.opt.timestep})")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Check 4: Bulk load all scenes
    print("\n[4/4] Loading all scenes under scenes/...")
    scene_files = sorted(SCENES_DIR.glob("*.xml"))
    all_loaded = True
    for sf in scene_files:
        try:
            from sim.mujoco_env import load_model

            m = load_model(sf)
            print(f"  OK: {sf.name} (nq={m.nq})")
        except Exception as e:
            print(f"  FAIL: {sf.name} — {e}")
            all_loaded = False

    if not all_loaded:
        return 1

    # Save metadata
    print("\n--- Saving metadata ---")
    try:
        from sim.env_meta import collect_metadata

        meta = collect_metadata(PROJECT_ROOT)
        meta["phase"] = "0.2.5"
        meta["mjcf_files_linted"] = len(mjcf_files)
        meta["scenes_loaded"] = len(scene_files)
        meta["lint_errors"] = total_errors
        meta["lint_warnings"] = total_warnings

        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)
        meta_path = logs_dir / "asset_validation_meta.json"
        serializable = {
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()
        }
        meta_path.write_text(json.dumps(serializable, indent=2))
        print(f"  Metadata saved to {meta_path}")
    except Exception as e:
        print(f"  Warning: metadata save failed: {e}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
