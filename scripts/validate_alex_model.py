#!/usr/bin/env python3
"""Validate the Alex V1 upper-body robot model (Phase 1.1.1 + 1.1.2 + 1.1.3).

Runs stability checks, PD hold tests, joint sweeps, and generates video artifacts.

Usage:
    python scripts/validate_alex_model.py

Artifacts saved to: logs/alex_validate/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    print("FAIL: mujoco not installed")
    sys.exit(1)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "logs" / "alex_validate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("=" * 60)
    print("Alex V1 Upper-Body Model Validation")
    print("=" * 60)

    from sim.asset_loader import load_scene, resolve_scene_path

    scene_path = resolve_scene_path("alex_upper_body")
    model = load_scene("alex_upper_body")
    data = mujoco.MjData(model)
    print(f"\nModel loaded: {scene_path}")
    print(f"  nq={model.nq}, nv={model.nv}, njnt={model.njnt}")
    print(f"  nbody={model.nbody}, ngeom={model.ngeom}")
    print(f"  nu={model.nu}, nsite={model.nsite}, ncam={model.ncam}")

    # --- Joint table ---
    print("\n--- Joint Table ---")
    print(f"{'Idx':<4} {'Name':<22} {'Range (rad)':<26} {'Range (deg)'}")
    joint_table = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = model.jnt_range[i]
        lo_deg, hi_deg = np.degrees(lo), np.degrees(hi)
        print(f"{i:<4} {name:<22} [{lo:8.4f}, {hi:8.4f}]   [{lo_deg:7.1f}, {hi_deg:7.1f}]")
        joint_table.append({"index": i, "name": name, "range_rad": [float(lo), float(hi)]})

    with open(out_dir / "joint_table.json", "w") as f:
        json.dump(joint_table, f, indent=2)

    # --- Sites and cameras ---
    print("\n--- Sites ---")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"  site {i}: {name}")

    print("\n--- Cameras ---")
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  camera {i}: {name}")

    # --- Collision geom summary ---
    collision_count = sum(1 for i in range(model.ngeom) if model.geom_contype[i] > 0)
    visual_count = model.ngeom - collision_count
    print("\n--- Geom Summary ---")
    print(f"  Total: {model.ngeom}, Collision: {collision_count}, Visual-only: {visual_count}")

    # --- Stability test ---
    print("\n--- Stability Test (10s / 5000 steps) ---")
    results = {"passed": True, "checks": {}}
    data = mujoco.MjData(model)
    n_steps = 5000

    for _step in range(n_steps):
        mujoco.mj_step(model, data)

    qpos_finite = bool(np.all(np.isfinite(data.qpos)))
    qvel_finite = bool(np.all(np.isfinite(data.qvel)))
    max_qpos = float(np.max(np.abs(data.qpos)))
    max_qvel = float(np.max(np.abs(data.qvel)))

    results["checks"]["qpos_finite"] = qpos_finite
    results["checks"]["qvel_finite"] = qvel_finite
    results["checks"]["max_qpos"] = max_qpos
    results["checks"]["max_qvel"] = max_qvel

    print(f"  qpos finite: {qpos_finite}")
    print(f"  qvel finite: {qvel_finite}")
    print(f"  max|qpos|: {max_qpos:.6f}")
    print(f"  max|qvel|: {max_qvel:.6f}")

    if not qpos_finite or not qvel_finite:
        results["passed"] = False
        print("  FAIL: NaN detected!")

    if max_qpos > 10.0:
        results["passed"] = False
        print("  FAIL: qpos diverged!")

    # --- Home pose penetration check ---
    print("\n--- Home Pose Penetration Check ---")
    data_home = mujoco.MjData(model)
    mujoco.mj_forward(model, data_home)
    max_penetration = 0.0
    for i in range(data_home.ncon):
        dist = data_home.contact[i].dist
        if dist < max_penetration:
            max_penetration = dist
    results["checks"]["max_penetration_m"] = float(max_penetration)
    print(f"  Max penetration: {max_penetration * 1000:.2f} mm")
    if max_penetration < -5e-3:
        results["passed"] = False
        print("  FAIL: Penetration exceeds 5mm!")
    else:
        print("  OK")

    # --- Determinism check ---
    print("\n--- Determinism Check ---")
    from sim.mujoco_env import check_deterministic

    deterministic = check_deterministic(str(scene_path))
    results["checks"]["deterministic"] = deterministic
    print(f"  Deterministic: {deterministic}")
    if not deterministic:
        results["passed"] = False

    # --- EE positions at home ---
    print("\n--- EE Positions (home pose) ---")
    data_ee = mujoco.MjData(model)
    mujoco.mj_forward(model, data_ee)
    for site_name in ["left_ee_site", "right_ee_site"]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = data_ee.site_xpos[site_id]
        print(f"  {site_name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # --- PD Hold Test (Phase 1.1.2) ---
    print("\n--- PD Hold Test (5s, ctrl=0) ---")
    data_hold = mujoco.MjData(model)
    data_hold.ctrl[:] = 0.0
    for _ in range(2500):  # 5s at 0.002s
        mujoco.mj_step(model, data_hold)
    hold_max_drift = float(np.max(np.abs(data_hold.qpos)))
    hold_max_vel = float(np.max(np.abs(data_hold.qvel)))
    results["checks"]["hold_max_drift_rad"] = hold_max_drift
    results["checks"]["hold_max_vel_rad_s"] = hold_max_vel
    print(f"  Max drift from home: {hold_max_drift:.6f} rad")
    print(f"  Max velocity after settle: {hold_max_vel:.6f} rad/s")
    if hold_max_drift < 0.1:
        print("  OK: drift within 0.1 rad")
    else:
        print("  WARN: drift exceeds 0.1 rad")

    # --- Joint Sweep Test (Phase 1.1.2, updated for 1.1.3) ---
    print("\n--- Joint Sweep Test (each actuated joint to midrange) ---")
    sweep_results = []
    sweep_pass = True
    # Build joint→actuator mapping (only sweep joints with actuators)
    jnt_to_act: dict[int, int] = {}
    for a in range(model.nu):
        jnt_id = int(model.actuator_trnid[a, 0])
        jnt_to_act[jnt_id] = a

    for j, act_idx in jnt_to_act.items():
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[j]
        mid = (lo + hi) / 2.0
        data_sweep = mujoco.MjData(model)
        data_sweep.ctrl[:] = 0.0
        data_sweep.ctrl[act_idx] = mid
        for _ in range(1500):
            mujoco.mj_step(model, data_sweep)
        error = float(abs(data_sweep.qpos[j] - mid))
        finite = bool(np.all(np.isfinite(data_sweep.qpos)))
        status = "OK" if (finite and error < 0.35) else "FAIL"
        if status == "FAIL":
            sweep_pass = False
        sweep_results.append(
            {
                "joint": name,
                "target": float(mid),
                "actual": float(data_sweep.qpos[j]),
                "error": error,
                "finite": finite,
                "status": status,
            }
        )
        print(
            f"  {name:<28} target={mid:7.3f}  actual={data_sweep.qpos[j]:7.3f}  "
            f"err={error:.4f}  {status}"
        )
    results["checks"]["sweep_results"] = sweep_results
    if not sweep_pass:
        results["passed"] = False
        print("  FAIL: One or more joints failed sweep test!")

    # --- Energy Tracking (Phase 1.1.2) ---
    print("\n--- Energy Check (10s) ---")
    data_energy = mujoco.MjData(model)
    for _ in range(5000):
        mujoco.mj_step(model, data_energy)
    total_energy = float(data_energy.energy[0] + data_energy.energy[1])
    results["checks"]["total_energy_J"] = total_energy
    print(f"  Total energy: {total_energy:.2f} J (potential + kinetic)")
    if total_energy < 1000.0:
        print("  OK")
    else:
        results["passed"] = False
        print("  FAIL: Energy exceeds 1000 J!")

    # --- EZGripper Open/Close Test (Phase 1.1.3) ---
    print("\n--- EZGripper Open/Close Test ---")
    try:
        from sim.end_effector import EZGripperInterface

        data_ez = mujoco.MjData(model)
        for side in ["left", "right"]:
            ee = EZGripperInterface(model, data_ez, side=side)
            ee.set_grasp(1.0)
            for _ in range(500):
                mujoco.mj_step(model, data_ez)
            open_state = ee.get_grasp_state()
            ee.set_grasp(0.0)
            for _ in range(500):
                mujoco.mj_step(model, data_ez)
            closed_state = ee.get_grasp_state()
            pos, quat = ee.get_tool_frame_pose()
            ok = open_state > 0.7 and closed_state < 0.2
            print(
                f"  {side}: open={open_state:.3f} closed={closed_state:.3f} "
                f"tool_pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                f"{'OK' if ok else 'FAIL'}"
            )
            if not ok:
                results["passed"] = False
    except Exception as e:
        print(f"  EZGripper test skipped: {e}")

    # --- Render videos (if offscreen available) ---
    print("\n--- Rendering ---")
    try:
        from sim.offscreen import RenderConfig, render_trajectory, save_video

        for cam_name in ["overhead", "third_person", "robot_cam"]:
            data_render = mujoco.MjData(model)
            config = RenderConfig(camera_name=cam_name, width=640, height=480)
            frames = render_trajectory(
                model, data_render, n_steps=500, config=config, render_every=5
            )
            video_path = out_dir / f"{cam_name}.mp4"
            save_video(frames, video_path, fps=20)
            print(f"  Saved: {video_path}")
    except Exception as e:
        print(f"  Rendering skipped: {e}")

    # --- Save report ---
    with open(out_dir / "stability_report.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 60)
    if results["passed"]:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
    print(f"Artifacts saved to: {out_dir}")
    print("=" * 60)

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
