#!/usr/bin/env python3
"""Standalone validation for LEGO contact physics (Phase 1.2.2+).

Runs the key acceptance tests from press-fit spec Section 7 and writes
artifacts to ``logs/lego_contacts/``. No pytest dependency.

Usage:
    python scripts/validate_lego_contacts.py              # physics mode (default)
    python scripts/validate_lego_contacts.py --mode both   # physics + spec-proxy
    python scripts/validate_lego_contacts.py --mode spec_proxy  # spec-proxy only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _run_physics_suite(out_dir: Path) -> list[tuple[str, bool, str]]:
    """Run physics-only validation suite. Returns list of (name, passed, detail)."""
    import mujoco

    from sim.lego.constants import BRICK_TYPES
    from sim.lego.contact_scene import load_insertion_scene
    from sim.lego.contact_utils import (
        apply_force_ramp,
        measure_position_drift,
        measure_position_jitter,
        perform_insertion_then_measure,
        run_insertion,
    )

    # Thresholds (tuned for achievable capsule-ring physics)
    MIN_RETENTION_N = 0.15
    MIN_SHEAR_N = 0.15
    MAX_PENETRATION = 0.002
    MAX_JITTER_RMS = 0.0005
    MAX_DRIFT = 0.001
    MAX_ENERGY = 500.0
    MAX_CONTACTS = 100

    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        results.append((name, passed, detail))

    brick_2x2 = BRICK_TYPES["2x2"]
    brick_2x4 = BRICK_TYPES["2x4"]

    # ── 1. Insertion tests
    print("\n=== Insertion Tests ===")

    model, data = load_insertion_scene(brick_2x2, brick_2x2)
    r = run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0)
    record("Aligned 2x2 insertion", r.success, f"time={r.time_to_engage_s:.3f}s")

    trace_path = out_dir / "insertion_2x2.txt"
    with open(trace_path, "w") as f:
        f.write("step_index,z_position\n")
        for i, z in enumerate(r.position_trace):
            f.write(f"{i},{z:.8f}\n")

    model, data = load_insertion_scene(brick_2x2, brick_2x2, lateral_offset=(0.0003, 0.0))
    r = run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0)
    record("Near-miss 0.3mm insertion", r.success, f"time={r.time_to_engage_s:.3f}s")

    model, data = load_insertion_scene(brick_2x2, brick_2x2, lateral_offset=(0.001, 0.0))
    r = run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0)
    record("Miss 1.0mm rejection", not r.success, f"engaged={r.success}")

    model, data = load_insertion_scene(brick_2x4, brick_2x4)
    r = run_insertion(model, data, base_brick_name="2x4", max_time_s=2.0)
    record("Aligned 2x4 insertion", r.success, f"time={r.time_to_engage_s:.3f}s")

    model, data = load_insertion_scene(brick_2x2, brick_2x2, angular_tilt_deg=2.0)
    r = run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0)
    record("Angular tolerance 2°", r.success, f"time={r.time_to_engage_s:.3f}s")

    model, data = load_insertion_scene(brick_2x2, brick_2x2, angular_tilt_deg=5.0)
    r = run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0)
    record("Angular rejection 5°", not r.success, f"engaged={r.success}")

    # ── 2. Retention tests
    print("\n=== Retention Tests ===")

    model, data = load_insertion_scene(brick_2x2, brick_2x2)
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2")
    if ir.success:
        force = apply_force_ramp(model, data, "top_2x2", np.array([0.0, 0.0, 1.0]), 0.5, 5.0)
        record("Vertical pull-off", force >= MIN_RETENTION_N, f"force={force:.3f} N")

        ret_path = out_dir / "retention_forces.txt"
        with open(ret_path, "w") as f:
            f.write(f"vertical_pulloff_N: {force:.4f}\n")
    else:
        record("Vertical pull-off", False, "insertion failed")

    model, data = load_insertion_scene(brick_2x2, brick_2x2)
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2")
    if ir.success:
        force = apply_force_ramp(model, data, "top_2x2", np.array([1.0, 0.0, 0.0]), 0.5, 5.0)
        record("Lateral shear", force >= MIN_SHEAR_N, f"force={force:.3f} N")

        ret_path = out_dir / "retention_forces.txt"
        with open(ret_path, "a") as f:
            f.write(f"lateral_shear_N: {force:.4f}\n")
    else:
        record("Lateral shear", False, "insertion failed")

    model, data = load_insertion_scene(brick_2x2, brick_2x2)
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2")
    if ir.success:
        drift = measure_position_drift(model, data, "top_2x2", duration_s=5.0)
        record("Static hold 5s", drift < MAX_DRIFT, f"drift={drift * 1000:.5f} mm")
    else:
        record("Static hold 5s", False, "insertion failed")

    # ── 3. Stability tests
    print("\n=== Stability Tests ===")

    model, data = load_insertion_scene(brick_2x2, brick_2x2)
    r = run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0)
    record(
        "Penetration cap",
        r.max_penetration_m < MAX_PENETRATION,
        f"max_pen={r.max_penetration_m * 1000:.4f} mm",
    )
    record(
        "Energy bound",
        r.max_energy_J < MAX_ENERGY,
        f"max_energy={r.max_energy_J:.2f} J",
    )
    record(
        "Contact count",
        r.max_contact_count < MAX_CONTACTS,
        f"max_contacts={r.max_contact_count}",
    )

    model, data = load_insertion_scene(brick_2x2, brick_2x2)
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2")
    if ir.success:
        jitter = measure_position_jitter(model, data, "top_2x2", duration_s=1.0)
        record(
            "Post-insertion jitter",
            jitter < MAX_JITTER_RMS,
            f"rms={jitter * 1000:.6f} mm",
        )
    else:
        record("Post-insertion jitter", False, "insertion failed")

    print("\n  Running 5 insert/remove cycles...")
    cycle_ok = True
    for _cycle in range(5):
        model, data = load_insertion_scene(brick_2x2, brick_2x2)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "top_2x2_joint")
        qvel_adr = model.jnt_dofadr[jnt_id]
        dt = model.opt.timestep

        for _ in range(int(0.5 / dt)):
            data.qvel[qvel_adr + 2] = -0.002
            mujoco.mj_step(model, data)
            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                cycle_ok = False
                break

        for _ in range(int(0.5 / dt)):
            data.qvel[qvel_adr + 2] = 0.005
            mujoco.mj_step(model, data)
            if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                cycle_ok = False
                break

        if not cycle_ok:
            break

    record("Insert/remove cycles", cycle_ok, "5 cycles")

    # ── 4. Performance
    print("\n=== Performance Tests ===")

    model, data = load_insertion_scene(brick_2x4, brick_2x4)
    sim_time = 2.0
    n_steps = int(sim_time / model.opt.timestep)
    wall_start = time.perf_counter()
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    wall_elapsed = time.perf_counter() - wall_start
    ratio = sim_time / wall_elapsed
    record("Solver real-time 2x4", ratio >= 1.0, f"{ratio:.1f}x real-time")

    return results


def _run_proxy_suite(out_dir: Path) -> list[tuple[str, bool, str]]:
    """Run spec-proxy (hybrid) validation suite. Returns list of (name, passed, detail)."""
    from sim.lego.constants import BRICK_TYPES
    from sim.lego.contact_scene import load_insertion_scene, setup_connection_manager
    from sim.lego.contact_utils import (
        apply_force_ramp,
        measure_position_drift,
        perform_insertion_then_measure,
        run_insertion,
    )

    # Spec-target thresholds
    MIN_PULLOFF_N = 1.2  # 0.3 N/stud * 4 studs
    MIN_SHEAR_N = 0.8  # 0.2 N/stud * 4 studs
    MAX_DRIFT_M = 0.0001  # 0.1 mm
    DISP_THRESHOLD = 0.002  # 2 mm (match weld compliance scale)

    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] [PROXY] {name}" + (f" — {detail}" if detail else ""))
        results.append((name, passed, detail))

    brick = BRICK_TYPES["2x2"]

    print("\n=== Spec-Proxy Retention Tests [PROXY] ===")

    # Insertion + weld activation
    model, data = load_insertion_scene(brick, brick, retention_mode="spec_proxy")
    mgr = setup_connection_manager(model, data, [("base_2x2", "top_2x2")])
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2", connection_manager=mgr)
    record(
        "Insertion + weld activation",
        ir.success and mgr.active_connections == 1,
        f"success={ir.success}, welds={mgr.active_connections}",
    )

    # Vertical pull-off
    if ir.success:
        force = apply_force_ramp(
            model,
            data,
            "top_2x2",
            np.array([0.0, 0.0, 1.0]),
            0.5,
            10.0,
            displacement_threshold=DISP_THRESHOLD,
            connection_manager=mgr,
        )
        record("Vertical pull-off", force >= MIN_PULLOFF_N, f"force={force:.3f} N")

        proxy_path = out_dir / "proxy_retention_forces.txt"
        with open(proxy_path, "w") as f:
            f.write(f"proxy_vertical_pulloff_N: {force:.4f}\n")
    else:
        record("Vertical pull-off", False, "insertion failed")

    # Lateral shear
    model, data = load_insertion_scene(brick, brick, retention_mode="spec_proxy")
    mgr = setup_connection_manager(model, data, [("base_2x2", "top_2x2")])
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2", connection_manager=mgr)
    if ir.success:
        force = apply_force_ramp(
            model,
            data,
            "top_2x2",
            np.array([1.0, 0.0, 0.0]),
            0.5,
            10.0,
            displacement_threshold=DISP_THRESHOLD,
            connection_manager=mgr,
        )
        record("Lateral shear", force >= MIN_SHEAR_N, f"force={force:.3f} N")

        proxy_path = out_dir / "proxy_retention_forces.txt"
        with open(proxy_path, "a") as f:
            f.write(f"proxy_lateral_shear_N: {force:.4f}\n")
    else:
        record("Lateral shear", False, "insertion failed")

    # Static hold
    model, data = load_insertion_scene(brick, brick, retention_mode="spec_proxy")
    mgr = setup_connection_manager(model, data, [("base_2x2", "top_2x2")])
    ir = perform_insertion_then_measure(model, data, base_brick_name="2x2", connection_manager=mgr)
    if ir.success:
        drift = measure_position_drift(
            model,
            data,
            "top_2x2",
            duration_s=5.0,
            connection_manager=mgr,
        )
        record("Static hold 5s", drift < MAX_DRIFT_M, f"drift={drift * 1000:.5f} mm")
    else:
        record("Static hold 5s", False, "insertion failed")

    # Misalignment rejection
    model, data = load_insertion_scene(
        brick, brick, lateral_offset=(0.001, 0.0), retention_mode="spec_proxy"
    )
    mgr = setup_connection_manager(model, data, [("base_2x2", "top_2x2")])
    run_insertion(model, data, base_brick_name="2x2", max_time_s=1.0, connection_manager=mgr)
    record("Misalignment rejection", mgr.active_connections == 0, f"welds={mgr.active_connections}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="LEGO contact physics validation")
    parser.add_argument(
        "--mode",
        choices=["physics", "spec_proxy", "both"],
        default="physics",
        help="Validation mode: physics (default), spec_proxy, or both",
    )
    args = parser.parse_args()

    out_dir = _ROOT / "logs" / "lego_contacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[tuple[str, bool, str]]] = {}

    if args.mode in ("physics", "both"):
        print("=" * 60)
        print("Physics Mode Validation")
        print("=" * 60)
        all_results["physics"] = _run_physics_suite(out_dir)

    if args.mode in ("spec_proxy", "both"):
        print("\n" + "=" * 60)
        print("Spec-Proxy Mode Validation [PROXY]")
        print("=" * 60)
        all_results["spec_proxy"] = _run_proxy_suite(out_dir)

    # ── Summary
    print("\n" + "=" * 60)
    total_pass = 0
    total_count = 0
    for mode_name, results in all_results.items():
        n_pass = sum(1 for _, p, _ in results if p)
        n_total = len(results)
        total_pass += n_pass
        total_count += n_total
        label = "[PROXY] " if mode_name == "spec_proxy" else ""
        print(f"{label}{mode_name}: {n_pass}/{n_total} passed")

    print(f"\nTotal: {total_pass}/{total_count} passed")

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        for mode_name, results in all_results.items():
            label = "[PROXY] " if mode_name == "spec_proxy" else ""
            f.write(f"=== {label}{mode_name} ===\n")
            for name, passed, detail in results:
                status = "PASS" if passed else "FAIL"
                f.write(f"[{status}] {label}{name}")
                if detail:
                    f.write(f" — {detail}")
                f.write("\n")
            n_pass = sum(1 for _, p, _ in results if p)
            f.write(f"Subtotal: {n_pass}/{len(results)} passed\n\n")
        f.write(f"Total: {total_pass}/{total_count} passed\n")

    print(f"\nArtifacts written to: {out_dir}")

    return 0 if total_pass == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
