#!/usr/bin/env python3
"""Bounded parameter sweep for LEGO contact physics (Phase 1.2.2b pre-work).

Sweeps tube_capsule_radius x friction to document the physics ceiling:
can any pure-contact parameter combination achieve both insertion >=95%
AND retention >=0.3 N/stud simultaneously?

Usage:
    python scripts/sweep_lego_physics.py
"""

from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _patch_defaults(root: ET.Element, capsule_radius: float, friction: float) -> None:
    """Patch stud/tube default classes with sweep parameters."""
    for default in root.iter("default"):
        cls = default.get("class", "")
        if cls in ("lego/stud", "lego/tube"):
            for geom in default.iter("geom"):
                geom.set("friction", f"{friction} 0.005 0.005")


def run_single(
    capsule_radius_m: float,
    friction: float,
    n_trials: int = 10,
) -> dict:
    """Run insertion + retention for one parameter combo."""
    import mujoco

    from sim.lego.constants import (
        BRICK_HEIGHT,
        BRICK_TYPES,
        STUD_HALF_HEIGHT,
    )
    from sim.lego.contact_scene import generate_insertion_scene
    from sim.lego.contact_utils import apply_force_ramp, get_top_body_id

    brick = BRICK_TYPES["2x2"]
    base_height = 0.05

    # Generate base XML, then patch capsule radius and friction
    xml_str = generate_insertion_scene(brick, brick)
    root = ET.fromstring(xml_str)

    # Patch capsule radius on tube geoms
    for body in root.iter("body"):
        for geom in body.iter("geom"):
            geom_class = geom.get("class", "")
            if geom_class == "lego/tube":
                geom.set("size", f"{capsule_radius_m:.6f} {STUD_HALF_HEIGHT:.6f}")

    # Patch friction in defaults
    _patch_defaults(root, capsule_radius_m, friction)

    patched_xml = ET.tostring(root, encoding="unicode")

    # Insertion trials
    successes = 0
    rng = np.random.default_rng(seed=42)
    sigma = 0.0002  # 0.2 mm noise

    for _trial in range(n_trials):
        try:
            # Re-parse for each trial (fresh state)
            trial_root = ET.fromstring(patched_xml)

            # Add small XY offset noise to top brick
            dx = float(rng.normal(0, sigma))
            dy = float(rng.normal(0, sigma))
            for body in trial_root.iter("body"):
                if body.get("name", "").startswith("top_"):
                    pos = body.get("pos", "0 0 0").split()
                    pos[0] = f"{float(pos[0]) + dx:.8f}"
                    pos[1] = f"{float(pos[1]) + dy:.8f}"
                    body.set("pos", " ".join(pos))

            trial_xml = ET.tostring(trial_root, encoding="unicode")
            model = mujoco.MjModel.from_xml_string(trial_xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            body_id = get_top_body_id(model, "top", "2x2")
            brick_mass = model.body_mass[body_id]
            insertion_force = brick_mass * 9.81 * 5.0
            engage_z = base_height + BRICK_HEIGHT + STUD_HALF_HEIGHT

            dt = model.opt.timestep
            engaged = False
            for _step in range(int(2.0 / dt)):
                if not engaged:
                    data.xfrc_applied[body_id, 2] = -insertion_force
                mujoco.mj_step(model, data)
                if np.any(np.isnan(data.qpos)):
                    break
                if not engaged and data.xpos[body_id][2] <= engage_z:
                    engaged = True
                    data.xfrc_applied[body_id, :] = 0.0

            if engaged:
                successes += 1
        except Exception:
            pass

    insertion_rate = successes / n_trials

    # Retention measurement (single aligned trial)
    try:
        model = mujoco.MjModel.from_xml_string(patched_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        body_id = get_top_body_id(model, "top", "2x2")
        brick_mass = model.body_mass[body_id]
        insertion_force = brick_mass * 9.81 * 5.0
        engage_z = base_height + BRICK_HEIGHT + STUD_HALF_HEIGHT

        dt = model.opt.timestep
        engaged = False
        for _step in range(int(2.0 / dt)):
            if not engaged:
                data.xfrc_applied[body_id, 2] = -insertion_force
            mujoco.mj_step(model, data)
            if not engaged and data.xpos[body_id][2] <= engage_z:
                engaged = True
                data.xfrc_applied[body_id, :] = 0.0
                # Zero velocity and settle
                joint_name = "top_2x2_joint"
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qvel_adr = model.jnt_dofadr[jnt_id]
                data.qvel[qvel_adr : qvel_adr + 6] = 0.0

        # Settle
        for _ in range(int(0.5 / dt)):
            mujoco.mj_step(model, data)

        if engaged:
            pulloff = apply_force_ramp(
                model,
                data,
                body_name="top_2x2",
                direction=np.array([0.0, 0.0, 1.0]),
                force_rate=0.5,
                max_force=5.0,
            )
        else:
            pulloff = 0.0
    except Exception:
        pulloff = 0.0

    per_stud = pulloff / 4.0  # 2x2 has 4 studs

    return {
        "capsule_radius_mm": capsule_radius_m * 1000,
        "friction": friction,
        "insertion_rate": insertion_rate,
        "pulloff_N": pulloff,
        "per_stud_N": per_stud,
        "both_pass": insertion_rate >= 0.95 and per_stud >= 0.3,
    }


def main() -> int:
    out_dir = _ROOT / "logs" / "lego_physics_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    radii_mm = [0.50, 0.52, 0.55, 0.58, 0.60]
    frictions = [0.5, 0.65, 0.8, 1.0]

    print("LEGO Physics Parameter Sweep")
    print("=" * 70)
    print(f"Capsule radii (mm): {radii_mm}")
    print(f"Friction values: {frictions}")
    print(f"Combinations: {len(radii_mm) * len(frictions)}")
    print()

    results = []
    for r_mm in radii_mm:
        for fric in frictions:
            r_m = r_mm / 1000.0
            print(f"  r={r_mm:.2f}mm, friction={fric:.2f} ... ", end="", flush=True)
            res = run_single(r_m, fric)
            results.append(res)
            status = "BOTH PASS" if res["both_pass"] else ""
            ins_mark = "ok" if res["insertion_rate"] >= 0.95 else "FAIL"
            ret_mark = "ok" if res["per_stud_N"] >= 0.3 else "FAIL"
            print(
                f"ins={res['insertion_rate']:.0%}[{ins_mark}]  "
                f"pull={res['pulloff_N']:.3f}N ({res['per_stud_N']:.3f}N/stud)[{ret_mark}]  "
                f"{status}"
            )

    # Write CSV
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written to: {csv_path}")

    # Summary
    any_pass = any(r["both_pass"] for r in results)
    print("\n" + "=" * 70)
    if any_pass:
        print("RESULT: Found parameter combo that passes BOTH gates.")
        print("        Hybrid mode may not be needed. Review results.")
    else:
        print("RESULT: No parameter combo achieves both insertion >=95%")
        print("        AND retention >=0.3 N/stud simultaneously.")
        print("        This confirms the need for hybrid retention mode.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
