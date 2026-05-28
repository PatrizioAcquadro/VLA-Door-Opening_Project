#!/usr/bin/env python3
"""Generate Phase 1.1 proof-of-work artifacts: videos and frames.

Demonstrates all Phase 1.1 capabilities:
  1. Robot model: home pose + rest pose renders from all cameras
  2. Joint dynamics: gravity settling + PD hold
  3. EZGripper: open/close cycle
  4. Action space: random bounded actions producing stable motion
  5. Robot state: state vector extraction with overlay
  6. Multi-view cameras: synchronized dual-camera captures
  7. Grasp scene: cube grasp + lift

Artifacts saved to: logs/phase11_artifacts/

Usage:
    python scripts/generate_phase11_artifacts.py
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

try:
    import imageio.v3 as iio
except ImportError:
    print("FAIL: imageio not installed (pip install imageio[ffmpeg])")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "logs" / "phase11_artifacts"


def save_video(frames: list[np.ndarray], path: Path, fps: float = 20.0) -> None:
    """Save a list of RGB arrays as MP4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack(frames)
    iio.imwrite(str(path), stack, fps=fps)
    print(f"  Saved video: {path.relative_to(PROJECT_ROOT)} ({len(frames)} frames)")


def save_frame(rgb: np.ndarray, path: Path) -> None:
    """Save a single RGB array as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), rgb)
    print(f"  Saved frame: {path.relative_to(PROJECT_ROOT)}")


def render_from_camera(
    model: mujoco.MjModel, data: mujoco.MjData, cam_name: str, w: int = 640, h: int = 480
) -> np.ndarray:
    """Render a single RGB frame from a named camera."""
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer.update_scene(data, cam_id)
    rgb = renderer.render().copy()
    renderer.close()
    return rgb


def render_trajectory_cam(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    n_steps: int,
    render_every: int = 5,
    w: int = 640,
    h: int = 480,
) -> list[np.ndarray]:
    """Step simulation and capture frames from a camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer = mujoco.Renderer(model, height=h, width=w)
    frames: list[np.ndarray] = []
    try:
        for step in range(n_steps):
            if step % render_every == 0:
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, cam_id)
                frames.append(renderer.render().copy())
            mujoco.mj_step(model, data)
    finally:
        renderer.close()
    return frames


def artifact_1_static_poses(model: mujoco.MjModel) -> None:
    """Render home and rest poses from all 3 cameras."""
    print("\n[1/7] Static pose renders (home + rest, 3 cameras)")
    subdir = OUT_DIR / "01_static_poses"

    for pose_name in ["home", "rest"]:
        data = mujoco.MjData(model)

        # Apply keyframe
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose_name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Settle physics briefly (250 steps = 0.5s)
        for _ in range(250):
            mujoco.mj_step(model, data)

        for cam in ["overhead", "third_person", "robot_cam"]:
            rgb = render_from_camera(model, data, cam)
            save_frame(rgb, subdir / f"{pose_name}_{cam}.png")


def artifact_2_gravity_settle(model: mujoco.MjModel) -> None:
    """Video of gravity settling from home pose (shows stability)."""
    print("\n[2/7] Gravity settle + PD hold video (5s)")
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0  # PD targets at home

    frames = render_trajectory_cam(model, data, "third_person", n_steps=2500, render_every=5)
    save_video(frames, OUT_DIR / "02_gravity_settle" / "gravity_settle.mp4")


def artifact_3_ezgripper(model: mujoco.MjModel) -> None:
    """Video of EZGripper open/close cycles."""
    print("\n[3/7] EZGripper open/close cycle video")
    from sim.end_effector import EZGripperInterface

    data = mujoco.MjData(model)
    left_ee = EZGripperInterface(model, data, side="left")
    right_ee = EZGripperInterface(model, data, side="right")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []

    try:
        # 3 open/close cycles
        for _ in range(3):
            # Open
            left_ee.set_grasp(1.0)
            right_ee.set_grasp(1.0)
            for step in range(500):
                mujoco.mj_step(model, data)
                if step % 5 == 0:
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, cam_id)
                    frames.append(renderer.render().copy())

            # Close
            left_ee.set_grasp(0.0)
            right_ee.set_grasp(0.0)
            for step in range(500):
                mujoco.mj_step(model, data)
                if step % 5 == 0:
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, cam_id)
                    frames.append(renderer.render().copy())
    finally:
        renderer.close()

    save_video(frames, OUT_DIR / "03_ezgripper" / "ezgripper_open_close.mp4")


def artifact_4_action_space(model: mujoco.MjModel) -> None:
    """Video of random bounded actions showing stable controlled motion."""
    print("\n[4/7] Action space: random bounded actions (stable motion)")
    from sim.sim_runner import SimRunner

    data = mujoco.MjData(model)
    runner = SimRunner(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []

    rng = np.random.default_rng(42)

    try:
        # 200 control ticks = 10s at 20 Hz
        for tick in range(200):
            # Small random actions (arm in [-0.3, 0.3], grippers oscillate)
            arm_action = rng.uniform(-0.3, 0.3, size=15)
            gripper_action = np.array(
                [0.5 + 0.5 * np.sin(tick * 0.1), 0.5 + 0.5 * np.sin(tick * 0.1 + np.pi)]
            )
            action = np.concatenate([arm_action, gripper_action])
            state = runner.step(action)

            # Render every other tick (10 fps in video)
            if tick % 2 == 0:
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, cam_id)
                frames.append(renderer.render().copy())
    finally:
        renderer.close()

    save_video(frames, OUT_DIR / "04_action_space" / "random_actions.mp4", fps=10)

    # Log final state as proof
    flat = state.to_flat_array()
    report = {
        "total_ticks": 200,
        "sim_time_s": float(data.time),
        "state_dim": len(flat),
        "state_finite": bool(np.all(np.isfinite(flat))),
        "max_q": float(np.max(np.abs(flat[:15]))),
        "max_q_dot": float(np.max(np.abs(flat[15:30]))),
    }
    report_path = OUT_DIR / "04_action_space" / "random_action_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path.relative_to(PROJECT_ROOT)}")


def artifact_5_robot_state(model: mujoco.MjModel) -> None:
    """Extract and log robot state at several poses."""
    print("\n[5/7] Robot state extraction")
    from sim.robot_state import STATE_DIM, AlexRobotState

    subdir = OUT_DIR / "05_robot_state"
    subdir.mkdir(parents=True, exist_ok=True)

    state_extractor = AlexRobotState(model)
    states = {}

    for pose_name in ["home", "rest"]:
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, pose_name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)

        rs = state_extractor.get_state(data)
        flat = rs.to_flat_array()
        assert len(flat) == STATE_DIM, f"State dim mismatch: {len(flat)} != {STATE_DIM}"

        states[pose_name] = {
            "flat_state_dim": len(flat),
            "q": rs.q.tolist(),
            "q_dot": rs.q_dot.tolist(),
            "gripper_state": rs.gripper_state.tolist(),
            "left_ee_pos": rs.left_ee_pos.tolist(),
            "left_ee_quat": rs.left_ee_quat.tolist(),
            "right_ee_pos": rs.right_ee_pos.tolist(),
            "right_ee_quat": rs.right_ee_quat.tolist(),
            "left_ee_vel": rs.left_ee_vel.tolist(),
            "right_ee_vel": rs.right_ee_vel.tolist(),
            "warnings": rs.validate(),
            "valid": len(rs.validate()) == 0,
        }
        warnings = states[pose_name]["warnings"]
        print(
            f"  {pose_name}: state_dim={len(flat)}, valid={states[pose_name]['valid']}"
            + (f", warnings={warnings}" if warnings else "")
        )
        print(f"    left_ee_pos={rs.left_ee_pos.round(4).tolist()}")
        print(f"    right_ee_pos={rs.right_ee_pos.round(4).tolist()}")

    with open(subdir / "state_snapshots.json", "w") as f:
        json.dump(states, f, indent=2)
    print(f"  Saved: {(subdir / 'state_snapshots.json').relative_to(PROJECT_ROOT)}")


def artifact_6_multiview(model: mujoco.MjModel) -> None:
    """Synchronized dual-camera captures at rest pose."""
    print("\n[6/7] Multi-view synchronized captures")
    from sim.camera import CAMERA_NAMES, NUM_VIEWS, MultiViewRenderer

    subdir = OUT_DIR / "06_multiview"

    data = mujoco.MjData(model)
    # Apply rest keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rest")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    for _ in range(250):
        mujoco.mj_step(model, data)

    with MultiViewRenderer(model, width=640, height=480) as mv:
        frame = mv.capture(data, step_index=0)
        assert len(frame.views) == NUM_VIEWS

        for cam_name in CAMERA_NAMES:
            rgb = frame.views[cam_name].rgb
            save_frame(rgb, subdir / f"rest_{cam_name}_sync.png")

        # Get metadata
        mujoco.mj_forward(model, data)
        meta = mv.get_metadata(data)
        meta_dict = {name: m.to_dict() for name, m in meta.items()}
        with open(subdir / "camera_metadata.json", "w") as f:
            json.dump(meta_dict, f, indent=2)
        print(f"  Camera metadata: {(subdir / 'camera_metadata.json').relative_to(PROJECT_ROOT)}")

    # Also render a multi-view video with both cameras side by side
    data2 = mujoco.MjData(model)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data2, key_id)

    cam_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in CAMERA_NAMES
    }
    renderer = mujoco.Renderer(model, height=240, width=320)
    side_by_side_frames: list[np.ndarray] = []

    try:
        for step in range(1000):
            if step % 5 == 0:
                mujoco.mj_forward(model, data2)
                views = []
                for name in CAMERA_NAMES:
                    renderer.update_scene(data2, cam_ids[name])
                    views.append(renderer.render().copy())
                combined = np.concatenate(views, axis=1)  # side-by-side
                side_by_side_frames.append(combined)
            mujoco.mj_step(model, data2)
    finally:
        renderer.close()

    save_video(
        side_by_side_frames,
        subdir / "dual_camera_sync.mp4",
        fps=20,
    )


def artifact_7_grasp_scene(model_grasp: mujoco.MjModel) -> None:
    """Grasp scene: approach and close grippers on a cube."""
    print("\n[7/7] Grasp scene: cube interaction")
    from sim.end_effector import EZGripperInterface

    subdir = OUT_DIR / "07_grasp_scene"
    data = mujoco.MjData(model_grasp)

    # Apply pregrasp keyframe if available
    key_id = mujoco.mj_name2id(model_grasp, mujoco.mjtObj.mjOBJ_KEY, "pregrasp")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model_grasp, data, key_id)

    left_ee = EZGripperInterface(model_grasp, data, side="left")
    right_ee = EZGripperInterface(model_grasp, data, side="right")

    # Open grippers
    left_ee.set_grasp(1.0)
    right_ee.set_grasp(1.0)

    cam_id = mujoco.mj_name2id(model_grasp, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")
    renderer = mujoco.Renderer(model_grasp, height=480, width=640)
    frames: list[np.ndarray] = []

    try:
        # Phase 1: Settle with grippers open (1s)
        for step in range(500):
            mujoco.mj_step(model_grasp, data)
            if step % 5 == 0:
                mujoco.mj_forward(model_grasp, data)
                renderer.update_scene(data, cam_id)
                frames.append(renderer.render().copy())

        # Phase 2: Close grippers (1s)
        left_ee.set_grasp(0.0)
        right_ee.set_grasp(0.0)
        for step in range(500):
            mujoco.mj_step(model_grasp, data)
            if step % 5 == 0:
                mujoco.mj_forward(model_grasp, data)
                renderer.update_scene(data, cam_id)
                frames.append(renderer.render().copy())

        # Phase 3: Hold (1s)
        for step in range(500):
            mujoco.mj_step(model_grasp, data)
            if step % 5 == 0:
                mujoco.mj_forward(model_grasp, data)
                renderer.update_scene(data, cam_id)
                frames.append(renderer.render().copy())

        # Save a closeup if available
        grasp_cam = mujoco.mj_name2id(model_grasp, mujoco.mjtObj.mjOBJ_CAMERA, "grasp_closeup")
        if grasp_cam >= 0:
            mujoco.mj_forward(model_grasp, data)
            renderer.update_scene(data, grasp_cam)
            save_frame(renderer.render().copy(), subdir / "grasp_closeup.png")

    finally:
        renderer.close()

    save_video(frames, subdir / "grasp_scene.mp4")


def main() -> None:
    print("=" * 60)
    print("Phase 1.1 Proof Artifacts Generator")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from sim.asset_loader import load_scene

    # Load main scene
    print("\nLoading alex_upper_body scene...")
    model = load_scene("alex_upper_body")
    print(f"  njnt={model.njnt}, nu={model.nu}, ncam={model.ncam}")

    # Run artifact generators
    artifact_1_static_poses(model)
    artifact_2_gravity_settle(model)
    artifact_3_ezgripper(model)
    artifact_4_action_space(model)
    artifact_5_robot_state(model)
    artifact_6_multiview(model)

    # Load grasp scene for artifact 7
    print("\nLoading alex_grasp_test scene...")
    model_grasp = load_scene("alex_grasp_test")
    print(f"  njnt={model_grasp.njnt}, nu={model_grasp.nu}")
    artifact_7_grasp_scene(model_grasp)

    # Summary
    print("\n" + "=" * 60)
    print("All Phase 1.1 artifacts generated successfully!")
    print(f"Output directory: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    # List all generated files
    all_files = sorted(OUT_DIR.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    print(f"\nGenerated {len(all_files)} files:")
    for f in all_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
