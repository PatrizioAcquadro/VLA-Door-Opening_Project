#!/usr/bin/env python3
"""Generate Phase 1.2 proof-of-work artifacts: videos, frames, and reports.

Demonstrates all Phase 1.2 LEGO simulation capabilities:
  1. Brick gallery: all 3 brick types rendered from multiple angles
  2. Baseplate: 8x8 baseplate closeup with stud grid
  3. Contact insertion: brick-on-brick insertion sequence (approach -> engage -> settle)
  4. Retention test: pull-off force ramp showing brick holding under load
  5. Workspace scene: Alex robot + table + baseplate (multi-camera)
  6. Episode manager: multiple resets showing random brick spawns
  7. Scripted assembly: single brick placed on baseplate (headline demo)
  8. Multi-brick assembly: 3-brick assembly sequence on baseplate
  9. Summary report: all validation results

Artifacts saved to: logs/phase12_artifacts/

Usage:
    python scripts/generate_phase12_artifacts.py
"""

from __future__ import annotations

import json
import sys
import time
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
OUT_DIR = PROJECT_ROOT / "logs" / "phase12_artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def make_free_cam(
    lookat: tuple[float, float, float] = (0, 0, 0),
    distance: float = 0.1,
    azimuth: float = 135,
    elevation: float = -25,
) -> mujoco.MjvCamera:
    """Create a free camera with given parameters."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_free(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: mujoco.MjvCamera,
    renderer: mujoco.Renderer,
) -> np.ndarray:
    """Render a single frame using a free camera."""
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam)
    return renderer.render().copy()


def render_named(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    w: int = 640,
    h: int = 480,
) -> np.ndarray:
    """Render a single RGB frame from a named camera."""
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer.update_scene(data, cam_id)
    rgb = renderer.render().copy()
    renderer.close()
    return rgb


class VideoRecorder:
    """Accumulates frames from a named camera during simulation stepping."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cam_name: str,
        w: int = 640,
        h: int = 480,
    ) -> None:
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.renderer = mujoco.Renderer(model, height=h, width=w)
        self.frames: list[np.ndarray] = []

    def capture(self, data: mujoco.MjData) -> None:
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, self.cam_id)
        self.frames.append(self.renderer.render().copy())

    def close(self) -> None:
        self.renderer.close()


class MultiCamRecorder:
    """Records from multiple cameras and tiles them into a grid."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cam_names: list[str],
        w: int = 320,
        h: int = 240,
        grid_cols: int = 2,
    ) -> None:
        self.model = model
        self.cam_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in cam_names]
        self.renderer = mujoco.Renderer(model, height=h, width=w)
        self.frames: list[np.ndarray] = []
        self.grid_cols = grid_cols

    def capture(self, data: mujoco.MjData) -> None:
        mujoco.mj_forward(self.model, data)
        views = []
        for cid in self.cam_ids:
            self.renderer.update_scene(data, cid)
            views.append(self.renderer.render().copy())
        # Tile into grid
        rows = []
        for i in range(0, len(views), self.grid_cols):
            row_views = views[i : i + self.grid_cols]
            while len(row_views) < self.grid_cols:
                row_views.append(np.zeros_like(views[0]))
            rows.append(np.concatenate(row_views, axis=1))
        self.frames.append(np.concatenate(rows, axis=0))

    def close(self) -> None:
        self.renderer.close()


# ---------------------------------------------------------------------------
# Artifact 1: Brick gallery
# ---------------------------------------------------------------------------


def artifact_1_brick_gallery() -> None:
    """Render all 3 brick types from a closeup camera angle."""
    print("\n[1/9] LEGO brick gallery (2x2, 2x4, 2x6)")
    from sim.lego.constants import BRICK_TYPES

    subdir = OUT_DIR / "01_brick_gallery"

    for bt_name in ["2x2", "2x4", "2x6"]:
        bt = BRICK_TYPES[bt_name]

        brick_path = PROJECT_ROOT / "sim" / "assets" / "lego" / "bricks" / f"brick_{bt_name}.xml"
        model = mujoco.MjModel.from_xml_path(str(brick_path))
        data = mujoco.MjData(model)

        for _ in range(200):
            mujoco.mj_step(model, data)

        cam = make_free_cam(
            lookat=(0, 0, bt.shell_half_z),
            distance=max(0.06, bt.shell_half_x * 6),
            azimuth=135,
            elevation=-25,
        )

        # Static frame
        renderer = mujoco.Renderer(model, height=480, width=640)
        save_frame(render_free(model, data, cam, renderer), subdir / f"brick_{bt_name}.png")
        renderer.close()

        # Spinning video
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames: list[np.ndarray] = []
        for angle_deg in range(0, 360, 3):
            cam = make_free_cam(
                lookat=(0, 0, bt.shell_half_z),
                distance=max(0.06, bt.shell_half_x * 6),
                azimuth=float(angle_deg),
                elevation=-25,
            )
            frames.append(render_free(model, data, cam, renderer))
        renderer.close()
        save_video(frames, subdir / f"brick_{bt_name}_spin.mp4", fps=30)


# ---------------------------------------------------------------------------
# Artifact 2: Baseplate
# ---------------------------------------------------------------------------


def artifact_2_baseplate() -> None:
    """Render the 8x8 baseplate from closeup angles."""
    print("\n[2/9] LEGO baseplate (8x8)")
    from sim.lego.constants import BASEPLATE_TYPES

    subdir = OUT_DIR / "02_baseplate"

    bp_path = PROJECT_ROOT / "sim" / "assets" / "lego" / "baseplates" / "baseplate_8x8.xml"
    model = mujoco.MjModel.from_xml_path(str(bp_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bp = BASEPLATE_TYPES["8x8"]

    # Angled view
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = make_free_cam(lookat=(0, 0, bp.thickness / 2), distance=0.12, azimuth=135, elevation=-30)
    save_frame(render_free(model, data, cam, renderer), subdir / "baseplate_8x8.png")
    renderer.close()

    # Top-down view
    renderer = mujoco.Renderer(model, height=480, width=480)
    cam = make_free_cam(lookat=(0, 0, bp.thickness / 2), distance=0.12, azimuth=0, elevation=-90)
    save_frame(render_free(model, data, cam, renderer), subdir / "baseplate_8x8_topdown.png")
    renderer.close()


# ---------------------------------------------------------------------------
# Artifact 3: Contact insertion video
# ---------------------------------------------------------------------------


def artifact_3_contact_insertion() -> None:
    """Video of brick-on-brick insertion (approach -> engage -> settle)."""
    print("\n[3/9] Contact insertion video (2x4 on 2x4)")
    from sim.lego.constants import BRICK_TYPES
    from sim.lego.contact_scene import load_insertion_scene

    subdir = OUT_DIR / "03_contact_insertion"

    bt = BRICK_TYPES["2x4"]
    model, data = load_insertion_scene(bt, bt)

    top_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"top_{bt.name}")
    brick_mass = model.body_mass[top_body_id]
    insertion_force = brick_mass * 9.81 * 5.0

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []
    cam = make_free_cam(lookat=(0, 0, 0.06), distance=0.12, azimuth=135, elevation=-20)

    def capture() -> None:
        frames.append(render_free(model, data, cam, renderer))

    # Phase 1: Show initial state (0.5s)
    for step in range(250):
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            capture()

    # Phase 2: Apply insertion force (1.5s)
    engage_threshold = 0.05 + 0.0096 + 0.00085  # base_height + brick_h + stud_half
    for step in range(750):
        top_z = data.xpos[top_body_id][2]
        if top_z > engage_threshold:
            data.xfrc_applied[top_body_id, 2] = -insertion_force
        else:
            data.xfrc_applied[top_body_id, :] = 0.0
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            capture()

    # Phase 3: Settle (1.0s)
    data.xfrc_applied[top_body_id, :] = 0.0
    for step in range(500):
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            capture()

    renderer.close()
    save_video(frames, subdir / "insertion_2x4.mp4", fps=20)

    # Save final state closeup
    renderer2 = mujoco.Renderer(model, height=480, width=640)
    cam2 = make_free_cam(lookat=(0, 0, 0.06), distance=0.08, azimuth=135, elevation=-15)
    save_frame(render_free(model, data, cam2, renderer2), subdir / "insertion_2x4_final.png")
    renderer2.close()


# ---------------------------------------------------------------------------
# Artifact 4: Retention test
# ---------------------------------------------------------------------------


def artifact_4_retention_test() -> None:
    """Video of pull-off force ramp on an engaged brick."""
    print("\n[4/9] Retention test video (pull-off 2x2)")
    from sim.lego.constants import BRICK_TYPES
    from sim.lego.contact_scene import load_insertion_scene
    from sim.lego.contact_utils import run_insertion

    subdir = OUT_DIR / "04_retention_test"

    bt = BRICK_TYPES["2x2"]
    model, data = load_insertion_scene(bt, bt)

    result = run_insertion(model, data, base_brick_name="2x2")
    print(f"  Insertion: success={result.success}, time={result.time_to_engage_s:.3f}s")

    for _ in range(500):
        mujoco.mj_step(model, data)

    top_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"top_{bt.name}")
    initial_pos = data.xpos[top_body_id].copy()

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []
    force_log: list[dict] = []
    cam = make_free_cam(lookat=(0, 0, 0.06), distance=0.10, azimuth=135, elevation=-15)

    direction = np.array([0.0, 0.0, 1.0])  # pull upward
    dt = model.opt.timestep
    force_rate = 0.5  # N/s
    max_force = 3.0
    current_force = 0.0
    detached = False
    detach_force = max_force

    step_count = 0
    while current_force < max_force:
        current_force += force_rate * dt
        data.xfrc_applied[top_body_id, :3] = direction * current_force
        mujoco.mj_step(model, data)
        step_count += 1

        displacement = np.linalg.norm(data.xpos[top_body_id] - initial_pos)

        if step_count % 10 == 0:
            frames.append(render_free(model, data, cam, renderer))
            force_log.append(
                {
                    "step": step_count,
                    "force_N": round(float(current_force), 4),
                    "displacement_mm": round(float(displacement * 1000), 4),
                }
            )

        if not detached and displacement > 0.001:
            detached = True
            detach_force = current_force
            print(f"  Detachment at force={detach_force:.3f} N, disp={displacement*1000:.2f} mm")

    data.xfrc_applied[top_body_id, :] = 0.0
    for step in range(300):
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            frames.append(render_free(model, data, cam, renderer))

    renderer.close()
    save_video(frames, subdir / "pulloff_2x2.mp4", fps=20)

    report = {
        "brick_type": "2x2",
        "insertion_success": result.success,
        "detach_force_N": round(float(detach_force), 4),
        "force_log_samples": len(force_log),
    }
    report_path = subdir / "retention_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Artifact 5: Workspace scene (Alex + table + baseplate)
# ---------------------------------------------------------------------------


def artifact_5_workspace() -> None:
    """Multi-camera renders of the workspace scene."""
    print("\n[5/9] Workspace scene (Alex + table + baseplate)")
    from sim.asset_loader import load_scene

    subdir = OUT_DIR / "05_workspace"

    model = load_scene("alex_lego_workspace")
    data = mujoco.MjData(model)

    for _ in range(500):
        mujoco.mj_step(model, data)

    for cam_name in ["overhead", "third_person", "workspace_closeup"]:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            print(f"  WARNING: camera '{cam_name}' not found, skipping")
            continue
        save_frame(render_named(model, data, cam_name), subdir / f"{cam_name}.png")

    # Multi-camera tiled video
    cam_names = [
        n
        for n in ["overhead", "third_person", "workspace_closeup"]
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0
    ]

    data2 = mujoco.MjData(model)
    mc = MultiCamRecorder(model, cam_names, w=320, h=240, grid_cols=2)
    for step in range(1000):
        mujoco.mj_step(model, data2)
        if step % 5 == 0:
            mc.capture(data2)
    mc.close()
    save_video(mc.frames, subdir / "workspace_multicam.mp4")


# ---------------------------------------------------------------------------
# Artifact 6: Episode manager resets
# ---------------------------------------------------------------------------


def artifact_6_episode_resets() -> None:
    """Video montage of episode resets at different curriculum levels."""
    print("\n[6/9] Episode manager resets (5 episodes)")
    from sim.lego.episode_manager import (
        LEVEL_MULTI_STEP,
        LEVEL_SINGLE_BRICK,
        LEVEL_SINGLE_CONNECTION,
        EpisodeManager,
    )

    subdir = OUT_DIR / "06_episode_resets"

    em = EpisodeManager(max_bricks=4, brick_set=("2x2", "2x4", "2x6"))

    cam_names = [
        n
        for n in ["workspace_closeup", "overhead", "third_person"]
        if mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0
    ]
    if not cam_names:
        print("  WARNING: no cameras found in episode scene, skipping")
        return

    recorder = VideoRecorder(em.model, cam_names[0], w=640, h=480)

    episodes = [
        (42, LEVEL_SINGLE_BRICK, "L1 single brick"),
        (7, LEVEL_SINGLE_CONNECTION, "L2 single connection"),
        (100, LEVEL_MULTI_STEP, "L3 multi-step"),
        (200, LEVEL_MULTI_STEP, "L3 multi-step (seed=200)"),
        (999, LEVEL_MULTI_STEP, "L3 multi-step (seed=999)"),
    ]

    episode_report = []

    for seed, level, desc in episodes:
        info = em.reset(seed=seed, level=level)
        print(
            f"  Episode seed={seed}: {desc} — {len(info.brick_types)} bricks, "
            f"settle={info.settle_steps} steps, success={info.settle_success}"
        )

        episode_report.append(
            {
                "seed": seed,
                "level": level,
                "description": desc,
                "n_bricks": len(info.brick_types),
                "brick_types": info.brick_types,
                "settle_steps": info.settle_steps,
                "settle_success": info.settle_success,
            }
        )

        # Render 1s of this settled episode
        for step in range(250):
            mujoco.mj_step(em.model, em.data)
            if step % 5 == 0:
                recorder.capture(em.data)

        # Separator frames
        for _ in range(10):
            recorder.capture(em.data)

    recorder.close()
    save_video(recorder.frames, subdir / "episode_resets.mp4")

    report_path = subdir / "episode_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(episode_report, f, indent=2)
    print(
        f"  Metrics: success_rate={em.metrics.success_rate:.0%}, "
        f"avg_settle={em.metrics.avg_settle_steps:.0f} steps"
    )


# ---------------------------------------------------------------------------
# Artifact 7: Single brick scripted assembly (headline demo)
# ---------------------------------------------------------------------------


def artifact_7_single_assembly() -> None:
    """Video of a single brick being placed on the baseplate via scripted assembly."""
    print("\n[7/9] Single brick scripted assembly (headline demo)")
    from sim.lego.constants import BASEPLATE_TYPES, BRICK_TYPES, STUD_HALF_HEIGHT
    from sim.lego.episode_manager import LEVEL_SINGLE_CONNECTION, EpisodeManager
    from sim.lego.scripted_assembly import ScriptedAssembler
    from sim.lego.task import generate_assembly_goal

    subdir = OUT_DIR / "07_single_assembly"

    bp_type = BASEPLATE_TYPES["8x8"]
    em = EpisodeManager(baseplate=bp_type, brick_slots=["2x4"], max_bricks=1)

    info = em.reset(seed=42, level=LEVEL_SINGLE_CONNECTION)
    print(f"  Episode: bricks={info.brick_types}, settle={info.settle_steps}")

    bp_body_id = mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_BODY, f"baseplate_{bp_type.name}")
    mujoco.mj_forward(em.model, em.data)
    bp_world_pos = tuple(float(v) for v in em.data.xpos[bp_body_id])

    goal = generate_assembly_goal(info, bp_type, bp_world_pos, seed=42)
    target = goal.targets[0]
    print(f"  Target: type={target.brick_type}, pos={target.target_position}")

    cam_names = [
        n
        for n in ["workspace_closeup", "third_person", "overhead"]
        if mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0
    ]
    if not cam_names:
        print("  WARNING: no cameras found, skipping")
        return

    recorder = VideoRecorder(em.model, cam_names[0], w=640, h=480)

    bt = BRICK_TYPES[target.brick_type]
    body_name = f"brick_0_{bt.name}"
    joint_name = f"brick_0_{bt.name}_joint"
    body_id = mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = int(em.model.jnt_qposadr[joint_id])
    vel_addr = int(em.model.jnt_dofadr[joint_id])

    # Phase 1: Show spawned state (0.5s)
    for step in range(250):
        mujoco.mj_step(em.model, em.data)
        if step % 5 == 0:
            recorder.capture(em.data)

    # Phase 2: Kinematic move to approach
    approach_pos = (
        target.target_position[0],
        target.target_position[1],
        target.target_position[2] + 0.02,
    )
    em.data.qpos[qpos_addr : qpos_addr + 3] = approach_pos
    em.data.qpos[qpos_addr + 3 : qpos_addr + 7] = target.target_quaternion
    em.data.qvel[vel_addr : vel_addr + 6] = 0.0
    em.data.xfrc_applied[body_id, :] = 0.0
    mujoco.mj_forward(em.model, em.data)

    # Pause at approach (0.25s)
    for step in range(125):
        mujoco.mj_step(em.model, em.data)
        if step % 5 == 0:
            recorder.capture(em.data)

    # Phase 3: Force-based insertion
    brick_mass = float(em.model.body_mass[body_id])
    insertion_force = brick_mass * 9.81 * 5.0
    engage_z = target.target_position[2] + STUD_HALF_HEIGHT
    engaged = False

    for step in range(2000):
        if not engaged:
            em.data.xfrc_applied[body_id, 2] = -insertion_force
        mujoco.mj_step(em.model, em.data)

        if step % 10 == 0:
            body_z = float(em.data.xpos[body_id][2])
            if not engaged and body_z <= engage_z:
                engaged = True
                em.data.xfrc_applied[body_id, :] = 0.0
                print(f"  Engaged at step {step}, z={body_z:.6f}")

        if step % 5 == 0:
            recorder.capture(em.data)

    em.data.xfrc_applied[body_id, :] = 0.0

    # Phase 4: Settle and hold (1.5s)
    for step in range(750):
        mujoco.mj_step(em.model, em.data)
        if step % 5 == 0:
            recorder.capture(em.data)

    recorder.close()
    save_video(recorder.frames, subdir / "single_assembly.mp4")

    # Final state frames from all cameras
    mujoco.mj_forward(em.model, em.data)
    for cam in cam_names:
        save_frame(render_named(em.model, em.data, cam), subdir / f"final_{cam}.png")

    # Metrics via ScriptedAssembler
    em2 = EpisodeManager(baseplate=bp_type, brick_slots=["2x4"], max_bricks=1)
    info2 = em2.reset(seed=42, level=LEVEL_SINGLE_CONNECTION)
    mujoco.mj_forward(em2.model, em2.data)
    bp_pos2 = tuple(float(v) for v in em2.data.xpos[bp_body_id])
    goal2 = generate_assembly_goal(info2, bp_type, bp_pos2, seed=42)
    assembler = ScriptedAssembler(em2.model, em2.data)
    result = assembler.execute_assembly(goal2)

    report = {
        "brick_type": target.brick_type,
        "target_position": list(target.target_position),
        "engaged": engaged,
        "n_successful": result.n_successful,
        "n_total": result.n_total,
        "all_placed": result.all_placed,
        "structure_stable": result.structure_stable,
        "total_physics_steps": result.total_physics_steps,
        "max_penetration_mm": round(result.max_penetration_m * 1000, 4),
    }
    report_path = subdir / "assembly_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Artifact 8: Multi-brick assembly
# ---------------------------------------------------------------------------


def artifact_8_multi_assembly() -> None:
    """Video of 3 bricks placed sequentially on the baseplate."""
    print("\n[8/9] Multi-brick assembly (3 bricks)")
    from sim.lego.constants import BASEPLATE_TYPES, BRICK_TYPES, STUD_HALF_HEIGHT
    from sim.lego.episode_manager import LEVEL_MULTI_STEP, EpisodeManager
    from sim.lego.scripted_assembly import ScriptedAssembler
    from sim.lego.task import generate_assembly_goal

    subdir = OUT_DIR / "08_multi_assembly"

    bp_type = BASEPLATE_TYPES["8x8"]
    em = EpisodeManager(
        baseplate=bp_type,
        brick_slots=["2x2", "2x4", "2x2"],
        max_bricks=3,
    )

    info = em.reset(seed=77, level=LEVEL_MULTI_STEP, n_active=3)
    print(f"  Episode: bricks={info.brick_types}, settle={info.settle_steps}")

    bp_body_id = mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_BODY, f"baseplate_{bp_type.name}")
    mujoco.mj_forward(em.model, em.data)
    bp_world_pos = tuple(float(v) for v in em.data.xpos[bp_body_id])

    goal = generate_assembly_goal(info, bp_type, bp_world_pos, seed=77)

    cam_names = [
        n
        for n in ["workspace_closeup", "third_person", "overhead"]
        if mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0
    ]
    if not cam_names:
        print("  WARNING: no cameras found, skipping")
        return

    recorder = VideoRecorder(em.model, cam_names[0], w=640, h=480)

    # Record initial state
    for step in range(125):
        mujoco.mj_step(em.model, em.data)
        if step % 5 == 0:
            recorder.capture(em.data)

    # Place each brick
    for i, target in enumerate(goal.targets):
        bt = BRICK_TYPES[target.brick_type]
        body_name = f"brick_{target.slot_index}_{bt.name}"
        joint_name = f"brick_{target.slot_index}_{bt.name}_joint"
        body_id = mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = mujoco.mj_name2id(em.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = int(em.model.jnt_qposadr[joint_id])
        vel_addr = int(em.model.jnt_dofadr[joint_id])

        print(
            f"  Placing brick {i}: {target.brick_type} at "
            f"({target.target_position[0]:.4f}, {target.target_position[1]:.4f})"
        )

        # Move to approach
        approach_pos = (
            target.target_position[0],
            target.target_position[1],
            target.target_position[2] + 0.02,
        )
        em.data.qpos[qpos_addr : qpos_addr + 3] = approach_pos
        em.data.qpos[qpos_addr + 3 : qpos_addr + 7] = target.target_quaternion
        em.data.qvel[vel_addr : vel_addr + 6] = 0.0
        em.data.xfrc_applied[body_id, :] = 0.0
        mujoco.mj_forward(em.model, em.data)

        # Pause at approach
        for step in range(75):
            mujoco.mj_step(em.model, em.data)
            if step % 5 == 0:
                recorder.capture(em.data)

        # Insertion
        brick_mass = float(em.model.body_mass[body_id])
        insertion_force = brick_mass * 9.81 * 5.0
        engage_z = target.target_position[2] + STUD_HALF_HEIGHT
        engaged = False

        for step in range(2000):
            if not engaged:
                em.data.xfrc_applied[body_id, 2] = -insertion_force
            mujoco.mj_step(em.model, em.data)

            if step % 10 == 0:
                body_z = float(em.data.xpos[body_id][2])
                if not engaged and body_z <= engage_z:
                    engaged = True
                    em.data.xfrc_applied[body_id, :] = 0.0
                    print(f"    Engaged at step {step}")

            if step % 5 == 0:
                recorder.capture(em.data)

        em.data.xfrc_applied[body_id, :] = 0.0

        # Settle between bricks
        for step in range(250):
            mujoco.mj_step(em.model, em.data)
            if step % 5 == 0:
                recorder.capture(em.data)

    # Final hold (1s)
    for step in range(500):
        mujoco.mj_step(em.model, em.data)
        if step % 5 == 0:
            recorder.capture(em.data)

    recorder.close()
    save_video(recorder.frames, subdir / "multi_assembly.mp4")

    # Final frames
    mujoco.mj_forward(em.model, em.data)
    for cam in cam_names:
        save_frame(render_named(em.model, em.data, cam), subdir / f"final_{cam}.png")

    # Metrics via ScriptedAssembler
    em2 = EpisodeManager(baseplate=bp_type, brick_slots=["2x2", "2x4", "2x2"], max_bricks=3)
    info2 = em2.reset(seed=77, level=LEVEL_MULTI_STEP, n_active=3)
    mujoco.mj_forward(em2.model, em2.data)
    bp_pos2 = tuple(float(v) for v in em2.data.xpos[bp_body_id])
    goal2 = generate_assembly_goal(info2, bp_type, bp_pos2, seed=77)
    assembler = ScriptedAssembler(em2.model, em2.data)
    result = assembler.execute_assembly(goal2)

    report = {
        "n_bricks": 3,
        "brick_types": info.brick_types,
        "n_successful": result.n_successful,
        "n_total": result.n_total,
        "all_placed": result.all_placed,
        "structure_stable": result.structure_stable,
        "total_physics_steps": result.total_physics_steps,
        "max_penetration_mm": round(result.max_penetration_m * 1000, 4),
        "per_brick": [
            {
                "brick_type": p.target.brick_type,
                "success": p.success,
                "z_engaged": p.z_engaged,
                "position_error_mm": round(p.position_error_m * 1000, 4),
            }
            for p in result.placements
        ],
    }
    report_path = subdir / "multi_assembly_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"  Result: {result.n_successful}/{result.n_total} placed, "
        f"stable={result.structure_stable}"
    )


# ---------------------------------------------------------------------------
# Artifact 9: Summary report
# ---------------------------------------------------------------------------


def artifact_9_summary() -> None:
    """Collect all validation results into a summary report."""
    print("\n[9/9] Summary report")

    subdir = OUT_DIR / "09_summary"
    subdir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "phase": "1.2",
        "description": "LEGO simulation: procedural bricks, contact physics, "
        "episode management, scripted assembly",
        "sub_phases": {
            "1.2.0": "Press-fit specification",
            "1.2.1": "Procedural LEGO bricks (2x2, 2x4, 2x6)",
            "1.2.2a": "Contact physics (insertion, retention, stability)",
            "1.2.2b": "Hybrid retention (ConnectionManager + weld constraints)",
            "1.2.3": "Baseplate (8x8) + workspace scene",
            "1.2.4": "4-view camera contract (overhead, wrist L/R, third_person)",
            "1.2.5": "Episode manager (template model, deterministic resets, curriculum)",
            "1.2.6": "MVP-3 task (goal generation, scripted assembly, evaluation)",
        },
    }

    contact_summary = PROJECT_ROOT / "logs" / "lego_contacts" / "summary.txt"
    if contact_summary.exists():
        summary["contact_validation"] = contact_summary.read_text().strip()

    episode_report = PROJECT_ROOT / "logs" / "episode_manager" / "validation_report.txt"
    if episode_report.exists():
        summary["episode_validation"] = episode_report.read_text().strip()

    task_results = PROJECT_ROOT / "logs" / "lego_task" / "validation_results.json"
    if task_results.exists():
        with open(task_results) as f:
            task_data = json.load(f)
        n_pass = sum(1 for t in task_data if t.get("passed"))
        n_total = len(task_data)
        summary["task_validation"] = f"{n_pass}/{n_total} passed"
        summary["task_validation_detail"] = task_data

    with open(subdir / "phase12_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {(subdir / 'phase12_summary.json').relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Phase 1.2 Proof Artifacts Generator")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact_1_brick_gallery()
    artifact_2_baseplate()
    artifact_3_contact_insertion()
    artifact_4_retention_test()
    artifact_5_workspace()
    artifact_6_episode_resets()
    artifact_7_single_assembly()
    artifact_8_multi_assembly()
    artifact_9_summary()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"All Phase 1.2 artifacts generated in {elapsed:.1f}s!")
    print(f"Output directory: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    all_files = sorted(OUT_DIR.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    print(f"\nGenerated {len(all_files)} files:")
    for f in all_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
