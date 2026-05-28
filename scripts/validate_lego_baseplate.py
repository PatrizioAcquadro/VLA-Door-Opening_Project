#!/usr/bin/env python3
"""Standalone validation for LEGO baseplate & workspace (Phase 1.2.3).

Runs key acceptance tests for baseplate generation, contact physics,
and workspace scene loading. Writes artifacts to ``logs/lego_baseplate/``.

Usage:
    python scripts/validate_lego_baseplate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> int:
    import mujoco

    from sim.lego.baseplate_generator import generate_baseplate_mjcf, write_baseplate_assets
    from sim.lego.connector import get_baseplate_connectors
    from sim.lego.constants import BASEPLATE_TYPES, BRICK_TYPES
    from sim.lego.contact_scene import (
        generate_workspace_scene,
        load_baseplate_insertion_scene,
    )
    from sim.lego.contact_utils import (
        apply_force_ramp,
        measure_position_drift,
        run_insertion,
    )
    from sim.lego.mass import compute_baseplate_mass

    out_dir = _ROOT / "logs" / "lego_baseplate"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
        results.append((name, passed, detail))

    bp_8x8 = BASEPLATE_TYPES["8x8"]
    brick_2x2 = BRICK_TYPES["2x2"]

    print("=" * 60)
    print("LEGO Baseplate Validation (Phase 1.2.3)")
    print("=" * 60)

    # ── 1. Asset generation ──
    print("\n--- Asset Generation ---")
    t0 = time.time()
    paths = write_baseplate_assets()
    dt = time.time() - t0
    record("asset_generation", len(paths) > 0, f"{len(paths)} files in {dt:.2f}s")

    for p in paths:
        record(f"asset_exists_{p.stem}", p.exists(), str(p))

    # ── 2. MJCF loading ──
    print("\n--- MJCF Loading ---")
    xml = generate_baseplate_mjcf(bp_8x8)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        record("mjcf_loads", True, f"ngeom={model.ngeom}")
        record("no_nan_after_forward", not np.any(np.isnan(data.qpos)))
        record("no_freejoint", model.nq == 0 and model.nv == 0)
    except Exception as e:
        record("mjcf_loads", False, str(e))

    # ── 3. Connector metadata ──
    print("\n--- Connector Metadata ---")
    conn = get_baseplate_connectors(bp_8x8)
    record("stud_count", conn.n_studs == 64, f"got {conn.n_studs}")
    ids = [s.id for s in conn.studs]
    record("unique_ids", len(ids) == len(set(ids)))
    mass = compute_baseplate_mass(bp_8x8)
    record("mass_positive", mass > 0, f"{mass * 1000:.2f} g")

    # ── 4. Contact class verification ──
    print("\n--- Contact Classes ---")
    stud_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "baseplate_8x8_stud_0_0")
    record("stud_contype", model.geom_contype[stud_id] == 6, f"got {model.geom_contype[stud_id]}")
    record(
        "stud_conaffinity",
        model.geom_conaffinity[stud_id] == 7,
        f"got {model.geom_conaffinity[stud_id]}",
    )

    surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "baseplate_8x8_surface")
    record(
        "surface_contype",
        model.geom_contype[surface_id] == 2,
        f"got {model.geom_contype[surface_id]}",
    )
    record(
        "surface_conaffinity",
        model.geom_conaffinity[surface_id] == 3,
        f"got {model.geom_conaffinity[surface_id]}",
    )

    # ── 5. Brick-on-baseplate insertion ──
    print("\n--- Brick-on-Baseplate Insertion ---")
    model2, data2 = load_baseplate_insertion_scene(bp_8x8, brick_2x2)
    ins_result = run_insertion(
        model2,
        data2,
        base_brick_name=brick_2x2.name,
        base_height=0.0,
        base_surface_height=bp_8x8.thickness,
    )
    record("insertion_success", ins_result.success)
    record(
        "insertion_penetration",
        ins_result.max_penetration_m < 0.002,
        f"{ins_result.max_penetration_m * 1000:.3f} mm",
    )

    # ── 6. Retention ──
    print("\n--- Retention ---")
    if ins_result.success:
        pull_off = apply_force_ramp(
            model2,
            data2,
            body_name="top_2x2",
            direction=np.array([0, 0, 1]),
            force_rate=0.5,
            max_force=5.0,
        )
        record("retention_pull_off", pull_off > 0.01, f"{pull_off:.4f} N")
    else:
        record("retention_pull_off", False, "skipped (insertion failed)")

    # ── 7. Static hold / drift ──
    print("\n--- Static Hold ---")
    model3, data3 = load_baseplate_insertion_scene(bp_8x8, brick_2x2)
    ins3 = run_insertion(
        model3,
        data3,
        base_brick_name=brick_2x2.name,
        base_height=0.0,
        base_surface_height=bp_8x8.thickness,
    )
    if ins3.success:
        drift = measure_position_drift(model3, data3, "top_2x2", duration_s=2.0)
        record("static_hold_drift", drift < 0.005, f"{drift * 1000:.3f} mm")
    else:
        record("static_hold_drift", False, "skipped (insertion failed)")

    # ── 8. Numerical stability ──
    print("\n--- Numerical Stability ---")
    model4, data4 = load_baseplate_insertion_scene(bp_8x8, brick_2x2)
    nan_detected = False
    for _ in range(2000):
        mujoco.mj_step(model4, data4)
        if np.any(np.isnan(data4.qpos)):
            nan_detected = True
            break
    record("numerical_stability_2000_steps", not nan_detected)

    # ── 9. Workspace scene ──
    print("\n--- Workspace Scene ---")
    try:
        ws_xml = generate_workspace_scene(bp_8x8)
        scenes_dir = _ROOT / "sim" / "assets" / "scenes"
        tmp_path = scenes_dir / "_validate_workspace_tmp.xml"
        tmp_path.write_text(ws_xml)
        ws_model = mujoco.MjModel.from_xml_path(str(tmp_path))
        ws_data = mujoco.MjData(ws_model)
        mujoco.mj_forward(ws_model, ws_data)
        tmp_path.unlink(missing_ok=True)

        record("workspace_loads", True, f"ngeom={ws_model.ngeom}")
        table_id = mujoco.mj_name2id(ws_model, mujoco.mjtObj.mjOBJ_BODY, "table")
        record("workspace_has_table", table_id >= 0)
        bp_body_id = mujoco.mj_name2id(ws_model, mujoco.mjtObj.mjOBJ_BODY, "baseplate_8x8")
        record("workspace_has_baseplate", bp_body_id >= 0)
        record("workspace_has_actuators", ws_model.nu == 17, f"nu={ws_model.nu}")
    except Exception as e:
        record("workspace_loads", False, str(e))

    # ── Summary ──
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = sum(1 for _, p, _ in results if not p)
    print(f"Results: {n_pass} passed, {n_fail} failed, {len(results)} total")
    print("=" * 60)

    # Write results to file
    report_path = out_dir / "validation_report.txt"
    with report_path.open("w") as f:
        for name, passed, detail in results:
            status = "PASS" if passed else "FAIL"
            f.write(f"[{status}] {name}" + (f" -- {detail}" if detail else "") + "\n")
        f.write(f"\n{n_pass}/{len(results)} passed\n")
    print(f"\nReport written to: {report_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
