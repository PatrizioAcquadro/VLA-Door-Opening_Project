#!/usr/bin/env python3
"""Validate robot state contract and consistency (Phase 1.1.6).

Run:
    python scripts/validate_robot_state.py

Artifacts are saved to ``logs/robot_state_validation/``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from sim.action_space import ACTION_DIM, ARM_DIM, GRIPPER_DIM  # noqa: E402
from sim.asset_loader import load_scene  # noqa: E402
from sim.robot_state import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRIPPER_SLICE,
    LEFT_EE_POS_SLICE,
    LEFT_EE_QUAT_SLICE,
    LEFT_EE_VEL_SLICE,
    Q_DOT_SLICE,
    Q_SLICE,
    RIGHT_EE_POS_SLICE,
    RIGHT_EE_QUAT_SLICE,
    RIGHT_EE_VEL_SLICE,
    STATE_DIM,
    AlexRobotState,
)
from sim.sim_runner import SimRunner  # noqa: E402

ARTIFACT_DIR = _project_root / "logs" / "robot_state_validation"


def print_contract(rs: AlexRobotState) -> None:
    """Print the frozen state contract."""
    print("=" * 70)
    print("ROBOT STATE CONTRACT (Phase 1.1.6)")
    print("=" * 70)
    print(f"  State dim:         {rs.state_dim}")
    print("  Reference frame:   World (Z-up, X-forward, Y-left)")
    print("  Quat convention:   [w, x, y, z] (MuJoCo)")
    print()
    print("  State vector layout:")
    print("    [0:15]   q           -- arm joint positions (rad)")
    print("    [15:30]  q_dot       -- arm joint velocities (rad/s)")
    print("    [30:32]  gripper     -- gripper state [0=closed, 1=open]")
    print("    [32:35]  left_ee_pos -- left EE position (m)")
    print("    [35:39]  left_ee_quat-- left EE quaternion [w,x,y,z]")
    print("    [39:42]  right_ee_pos-- right EE position (m)")
    print("    [42:46]  right_ee_quat-- right EE quaternion [w,x,y,z]")
    print("    [46:49]  left_ee_vel -- left EE lin. vel (m/s)")
    print("    [49:52]  right_ee_vel-- right EE lin. vel (m/s)")
    print()

    nr = rs.norm_ranges
    print("  Arm joint normalization ranges:")
    for i, name in enumerate(ARM_JOINT_NAMES):
        print(
            f"    [{i:2d}] {name:30s}  q in [{nr.q_lo[i]:+7.3f}, {nr.q_hi[i]:+7.3f}]  "
            f"vel_lim = {nr.q_dot_hi[i]:.1f} rad/s"
        )
    print(f"\n  Workspace bounds:  lo={nr.ee_pos_lo}  hi={nr.ee_pos_hi}")
    print(f"  EE vel max:        {nr.ee_vel_max} m/s")
    print()


def print_state_at_rest(rs: AlexRobotState, data: mujoco.MjData) -> bool:
    """Print state at rest and check basic validity."""
    print("-" * 70)
    print("State at rest (home keyframe)")
    print("-" * 70)

    state = rs.get_state(data)
    flat = state.to_flat_array()
    warnings = state.validate()

    print(f"  timestamp:     {state.timestamp:.4f} s")
    print(f"  q (15):        {state.q}")
    print(f"  q_dot (15):    {state.q_dot}")
    print(f"  gripper (2):   {state.gripper_state}")
    print(f"  left_ee_pos:   {state.left_ee_pos}")
    print(f"  left_ee_quat:  {state.left_ee_quat}")
    print(f"  right_ee_pos:  {state.right_ee_pos}")
    print(f"  right_ee_quat: {state.right_ee_quat}")
    print(f"  left_ee_vel:   {state.left_ee_vel}")
    print(f"  right_ee_vel:  {state.right_ee_vel}")
    print(f"  All finite:    {np.all(np.isfinite(flat))}")
    print(f"  Warnings:      {warnings if warnings else 'None'}")
    print()

    return len(warnings) == 0


def test_random_trajectory(
    model: mujoco.MjModel, n_steps: int = 200, seed: int = 42
) -> tuple[bool, list[np.ndarray]]:
    """Run random actions and track state bounds."""
    print("-" * 70)
    print(f"Random action state tracking ({n_steps} steps, seed={seed})")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rs = runner.robot_state
    rng = np.random.default_rng(seed)

    all_states: list[np.ndarray] = []
    has_nan = False

    for step in range(n_steps):
        action = rng.uniform(-1, 1, size=ACTION_DIM)
        action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
        runner.step(action)

        flat = rs.get_flat_state(data)
        all_states.append(flat)

        if not np.all(np.isfinite(flat)):
            print(f"  FAIL: NaN in state at step {step}")
            has_nan = True
            break

    if not has_nan:
        arr = np.array(all_states)
        print("  Per-component min/max over trajectory:")

        slices = [
            ("q", Q_SLICE),
            ("q_dot", Q_DOT_SLICE),
            ("gripper", GRIPPER_SLICE),
            ("left_ee_pos", LEFT_EE_POS_SLICE),
            ("left_ee_quat", LEFT_EE_QUAT_SLICE),
            ("right_ee_pos", RIGHT_EE_POS_SLICE),
            ("right_ee_quat", RIGHT_EE_QUAT_SLICE),
            ("left_ee_vel", LEFT_EE_VEL_SLICE),
            ("right_ee_vel", RIGHT_EE_VEL_SLICE),
        ]
        for name, sl in slices:
            vals = arr[:, sl]
            print(
                f"    {name:16s}  min={np.min(vals, axis=0).round(4)}  "
                f"max={np.max(vals, axis=0).round(4)}"
            )

        # Verify EE quats stayed unit norm
        left_quat_norms = np.linalg.norm(arr[:, LEFT_EE_QUAT_SLICE], axis=1)
        right_quat_norms = np.linalg.norm(arr[:, RIGHT_EE_QUAT_SLICE], axis=1)
        max_quat_dev = max(
            np.max(np.abs(left_quat_norms - 1.0)),
            np.max(np.abs(right_quat_norms - 1.0)),
        )
        print(f"\n  Max quaternion norm deviation: {max_quat_dev:.6f}")

    passed = not has_nan
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed, all_states


def test_normalization_roundtrip(model: mujoco.MjModel) -> bool:
    """Test normalization round-trip accuracy."""
    print("-" * 70)
    print("Normalization round-trip test")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rs = runner.robot_state
    rng = np.random.default_rng(77)

    max_error = 0.0
    for _ in range(50):
        action = rng.uniform(-0.5, 0.5, size=ACTION_DIM)
        action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
        runner.step(action)

        flat = rs.get_flat_state(data)
        normed = rs.normalize(flat)
        recovered = rs.denormalize(normed)

        error = np.max(np.abs(recovered - flat))
        max_error = max(max_error, error)

    passed = max_error < 1e-8
    print(f"  Max round-trip error: {max_error:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_ee_consistency(model: mujoco.MjModel) -> bool:
    """Verify EE pose from state matches direct site query."""
    print("-" * 70)
    print("EE pose consistency check")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rs = runner.robot_state
    rng = np.random.default_rng(55)

    max_pos_err = 0.0
    max_quat_err = 0.0

    for _ in range(30):
        action = rng.uniform(-0.5, 0.5, size=ACTION_DIM)
        action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
        runner.step(action)

        state = rs.get_state(data)

        # Direct site query
        for side, state_pos, state_quat in [
            ("left", state.left_ee_pos, state.left_ee_quat),
            ("right", state.right_ee_pos, state.right_ee_quat),
        ]:
            site_name = f"{side}_tool_frame"
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            direct_pos = data.site_xpos[sid].copy()
            direct_mat = data.site_xmat[sid].reshape(3, 3)
            direct_quat = np.zeros(4)
            mujoco.mju_mat2Quat(direct_quat, direct_mat.flatten())

            pos_err = float(np.max(np.abs(state_pos - direct_pos)))
            quat_err = float(np.max(np.abs(state_quat - direct_quat)))
            max_pos_err = max(max_pos_err, pos_err)
            max_quat_err = max(max_quat_err, quat_err)

    passed = max_pos_err < 1e-10 and max_quat_err < 1e-10
    print(f"  Max EE pos error:  {max_pos_err:.2e}")
    print(f"  Max EE quat error: {max_quat_err:.2e}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_action_state_coupling(model: mujoco.MjModel) -> bool:
    """Verify state changes match action intent."""
    print("-" * 70)
    print("Action-state coupling check")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    runner = SimRunner(model, data)

    # Step with positive spine delta — expect q[0] to increase
    action = np.zeros(ACTION_DIM)
    action[0] = 0.8  # spine_z positive delta
    state = runner.step(action)

    spine_moved = state.q[0] > 0.001
    print(
        f"  Spine +delta -> q[0] = {state.q[0]:.6f} (expect > 0): {'OK' if spine_moved else 'FAIL'}"
    )

    # Reset and test gripper
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    runner2 = SimRunner(model, data)

    action = np.zeros(ACTION_DIM)
    action[ARM_DIM] = 1.0  # open left gripper
    for _ in range(40):
        state = runner2.step(action)
    gripper_opened = state.gripper_state[0] > 0.5
    print(
        f"  Gripper open cmd -> state = {state.gripper_state[0]:.4f} "
        f"(expect > 0.5): {'OK' if gripper_opened else 'FAIL'}"
    )

    passed = spine_moved and gripper_opened
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def save_trajectory_csv(all_states: list[np.ndarray]) -> None:
    """Save state trajectory to CSV for offline inspection."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ARTIFACT_DIR / "state_trajectory.csv"

    # Build header
    header = []
    for i in range(15):
        header.append(f"q_{ARM_JOINT_NAMES[i]}")
    for i in range(15):
        header.append(f"qdot_{ARM_JOINT_NAMES[i]}")
    header.extend(["gripper_left", "gripper_right"])
    header.extend(["left_ee_x", "left_ee_y", "left_ee_z"])
    header.extend(["left_ee_qw", "left_ee_qx", "left_ee_qy", "left_ee_qz"])
    header.extend(["right_ee_x", "right_ee_y", "right_ee_z"])
    header.extend(["right_ee_qw", "right_ee_qx", "right_ee_qy", "right_ee_qz"])
    header.extend(["left_ee_vx", "left_ee_vy", "left_ee_vz"])
    header.extend(["right_ee_vx", "right_ee_vy", "right_ee_vz"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + header)
        for i, state in enumerate(all_states):
            writer.writerow([i] + [f"{v:.8f}" for v in state])

    print(f"  Trajectory CSV saved: {csv_path} ({len(all_states)} rows x {STATE_DIM} cols)")


def main() -> None:
    print("Loading Alex upper-body model...")
    model = load_scene("alex_upper_body")
    rs = AlexRobotState(model)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print_contract(rs)

    results: dict[str, bool] = {}
    results["rest_state"] = print_state_at_rest(rs, data)

    passed, all_states = test_random_trajectory(model)
    results["random_trajectory"] = passed

    results["normalization"] = test_normalization_roundtrip(model)
    results["ee_consistency"] = test_ee_consistency(model)
    results["action_coupling"] = test_action_state_coupling(model)

    if all_states:
        save_trajectory_csv(all_states)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
    print()

    if all_pass:
        print("All validation checks PASSED.")
    else:
        print("Some checks FAILED -- see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
