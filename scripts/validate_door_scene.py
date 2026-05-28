"""Validate the active Alex door-opening MuJoCo scene.

Run:
    python scripts/validate_door_scene.py

Artifacts are saved to logs/door_scene/validation_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCENE_NAME = "alex_door_workspace"
LOGS_DIR = PROJECT_ROOT / "logs" / "door_scene"


def _check(condition: bool, label: str, report: dict[str, bool | int | float | str]) -> None:
    report[label] = bool(condition)
    print(f"  {'OK' if condition else 'FAIL'}: {label}")


def main() -> int:
    """Run door-scene validation checks. Returns 0 on success, 1 on failure."""
    print("=" * 60)
    print("Door Scene Validation: Alex Door-Opening Workspace")
    print("=" * 60)

    try:
        import mujoco
    except ImportError as exc:
        print(f"  FAIL: mujoco import failed: {exc}")
        print("  Hint: run `pip install -e '.[sim]'`")
        return 1

    try:
        import numpy as np
    except ImportError as exc:
        print(f"  FAIL: numpy import failed: {exc}")
        print("  Hint: run `pip install -e '.[sim]'` or `pip install -e '.[dev]'`")
        return 1

    from sim.asset_loader import resolve_scene_path
    from sim.mujoco_env import load_model

    report: dict[str, bool | int | float | str] = {}

    print("\n[1/5] Resolving scene path...")
    scene_path = resolve_scene_path(SCENE_NAME)
    _check(scene_path.exists(), f"{SCENE_NAME}.xml exists", report)

    print("\n[2/5] Loading MuJoCo model...")
    try:
        model = load_model(scene_path)
        data = mujoco.MjData(model)
        _check(model.nq > 0 and model.nv > 0, "model has dynamic coordinates", report)
        report["nq"] = int(model.nq)
        report["nv"] = int(model.nv)
        report["ngeom"] = int(model.ngeom)
    except Exception as exc:
        print(f"  FAIL: model load failed: {exc}")
        return 1

    print("\n[3/5] Checking door task contract...")
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
    handle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "door_handle_site")
    panel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "door_panel")
    handle_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "door_handle")

    _check(hinge_id >= 0, "door_hinge joint exists", report)
    _check(handle_site_id >= 0, "door_handle_site exists", report)
    _check(panel_geom_id >= 0, "door_panel geom exists", report)
    _check(handle_geom_id >= 0, "door_handle geom exists", report)

    if hinge_id < 0 or handle_site_id < 0:
        return 1

    hinge_axis = model.jnt_axis[hinge_id]
    hinge_range = model.jnt_range[hinge_id]
    _check(np.allclose(hinge_axis, np.array([0.0, 0.0, 1.0])), "hinge axis is vertical", report)
    _check(hinge_range[1] >= 1.0, "hinge opens at least 1 radian", report)
    _check(model.jnt_limited[hinge_id] == 1, "hinge joint is limited", report)
    report["hinge_range_min"] = float(hinge_range[0])
    report["hinge_range_max"] = float(hinge_range[1])

    print("\n[4/5] Stepping scene for numerical stability...")
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    finite_state = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    _check(finite_state, "qpos and qvel remain finite after 500 steps", report)
    report["door_angle_after_settle"] = float(data.qpos[model.jnt_qposadr[hinge_id]])

    print("\n[5/5] Checking sensor contract...")
    sensor_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        for sensor_id in range(model.nsensor)
    }
    _check("door_angle" in sensor_names, "door_angle sensor exists", report)
    _check("door_angular_velocity" in sensor_names, "door_angular_velocity sensor exists", report)
    _check("handle_touch" in sensor_names, "handle_touch sensor exists", report)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved to {report_path}")

    failed = [name for name, ok in report.items() if isinstance(ok, bool) and not ok]
    if failed:
        print("\nFAILED CHECKS:")
        for name in failed:
            print(f"  - {name}")
        return 1

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
