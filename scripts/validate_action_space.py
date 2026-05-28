#!/usr/bin/env python3
"""Validate action space contract and stability (Phase 1.1.5).

Run:
    python scripts/validate_action_space.py

Artifacts are saved to ``logs/action_space_validation/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from sim.action_space import (  # noqa: E402
    ACTION_DIM,
    ARM_ACTUATOR_NAMES,
    ARM_DIM,
    GRIPPER_DIM,
    AlexActionSpace,
)
from sim.asset_loader import load_scene  # noqa: E402
from sim.sim_runner import SimRunner  # noqa: E402

ARTIFACT_DIR = _project_root / "logs" / "action_space_validation"


def print_contract(action_space: AlexActionSpace) -> None:
    """Print the frozen action contract."""
    print("=" * 70)
    print("ACTION SPACE CONTRACT (Phase 1.1.5)")
    print("=" * 70)
    print(f"  Action dim:     {action_space.action_dim}")
    print(f"  Arm dim:        {action_space.arm_dim}")
    print(f"  Gripper dim:    {action_space.gripper_dim}")
    print(f"  Control rate:   {action_space.control_hz} Hz")
    print(f"  Control period: {action_space.control_dt * 1000:.1f} ms")
    print()
    print("  Arm joint ordering & delta_q_max:")
    dqm = action_space.delta_q_max
    for i, name in enumerate(ARM_ACTUATOR_NAMES):
        print(f"    [{i:2d}] {name:30s}  delta_q_max = {dqm[i]:.4f} rad")
    print(f"    [{ARM_DIM}] gripper_left               cmd in [0, 1]")
    print(f"    [{ARM_DIM + 1}] gripper_right              cmd in [0, 1]")
    print()


def test_random_actions(model: mujoco.MjModel, n_steps: int = 200, seed: int = 42) -> dict:
    """Run random actions and report stability metrics."""
    print("-" * 70)
    print(f"Random action stability test ({n_steps} steps, seed={seed})")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rng = np.random.default_rng(seed)

    max_energy = 0.0
    max_penetration = 0.0
    max_vel = 0.0

    for step in range(n_steps):
        action = rng.uniform(-1, 1, size=ACTION_DIM)
        action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
        runner.step(action)

        energy = data.energy[0] + data.energy[1]
        max_energy = max(max_energy, energy)
        max_vel = max(max_vel, float(np.max(np.abs(data.qvel))))

        for i in range(data.ncon):
            pen = -data.contact[i].dist
            max_penetration = max(max_penetration, pen)

        if not np.all(np.isfinite(data.qpos)):
            print(f"  FAIL: NaN in qpos at step {step}")
            return {"pass": False, "step": step}

    passed = max_energy < 1000.0 and max_penetration < 0.05
    status = "PASS" if passed else "FAIL"
    print(f"  Status:          {status}")
    print(f"  Max energy:      {max_energy:.2f} J (limit: 1000)")
    print(f"  Max penetration: {max_penetration * 100:.2f} cm (limit: 5 cm)")
    print(f"  Max velocity:    {max_vel:.2f} rad/s")
    print(f"  Final sim time:  {data.time:.2f} s")
    print()

    return {
        "pass": passed,
        "max_energy": max_energy,
        "max_penetration": max_penetration,
        "max_vel": max_vel,
    }


def test_action_chunks(
    model: mujoco.MjModel,
    chunk_size: int = 16,
    n_chunks: int = 10,
    seed: int = 77,
) -> dict:
    """Run repeated action chunks and report drift metrics."""
    print("-" * 70)
    print(f"Action chunk test (h={chunk_size}, {n_chunks} chunks, seed={seed})")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rng = np.random.default_rng(seed)

    chunk = rng.uniform(-0.3, 0.3, size=(chunk_size, ACTION_DIM))
    chunk[:, ARM_DIM:] = rng.uniform(0.2, 0.8, size=(chunk_size, GRIPPER_DIM))

    max_energy = 0.0
    max_vel = 0.0

    for _c in range(n_chunks):
        runner.step_sequence(chunk)
        energy = data.energy[0] + data.energy[1]
        max_energy = max(max_energy, energy)
        max_vel = max(max_vel, float(np.max(np.abs(data.qvel))))

    passed = np.all(np.isfinite(data.qpos)) and max_energy < 1000.0
    status = "PASS" if passed else "FAIL"
    total_steps = n_chunks * chunk_size
    print(f"  Status:          {status}")
    print(f"  Total steps:     {total_steps} ({total_steps * 0.05:.1f} s)")
    print(f"  Max energy:      {max_energy:.2f} J")
    print(f"  Max velocity:    {max_vel:.2f} rad/s")
    print(f"  Final qpos NaN:  {not np.all(np.isfinite(data.qpos))}")
    print()

    return {"pass": passed, "max_energy": max_energy, "max_vel": max_vel}


def generate_video(model: mujoco.MjModel) -> None:
    """Generate a video of the robot executing random actions."""
    try:
        from sim.offscreen import RenderConfig, RenderedFrame, save_video
    except ImportError:
        print("  Skipped: imageio not available for video generation")
        return

    print("-" * 70)
    print("Generating validation video...")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rng = np.random.default_rng(42)

    config = RenderConfig(
        width=640,
        height=480,
        camera_name="overhead",
    )
    try:
        renderer = mujoco.Renderer(model, config.height, config.width)
    except Exception as exc:
        print(f"  Skipped: cannot create renderer ({exc})")
        print("  Hint: set MUJOCO_GL=egl for headless rendering, or run on a display")
        return

    frames: list[RenderedFrame] = []
    n_steps = 100  # 5 seconds at 20 Hz

    for step in range(n_steps):
        action = rng.uniform(-0.5, 0.5, size=ACTION_DIM)
        action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
        runner.step(action)

        # Render every 2 control steps (10 fps video)
        if step % 2 == 0:
            mujoco.mj_forward(model, data)
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.camera_name)
            renderer.update_scene(data, camera=cam_id)
            rgb = renderer.render()
            frames.append(RenderedFrame(rgb=rgb.copy(), step=step, time=data.time))

    renderer.close()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = ARTIFACT_DIR / "random_actions.mp4"
    save_video(frames, video_path, fps=10)
    print(f"  Video saved: {video_path}")
    print()


def main() -> None:
    print("Loading Alex upper-body model...")
    model = load_scene("alex_upper_body")
    action_space = AlexActionSpace(model)

    print_contract(action_space)

    results = {}
    results["random"] = test_random_actions(model)
    results["chunks"] = test_action_chunks(model)

    generate_video(model)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = all(r["pass"] for r in results.values())
    for name, r in results.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {name:20s} {status}")
    print()

    if all_pass:
        print("All validation checks PASSED.")
    else:
        print("Some checks FAILED — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
