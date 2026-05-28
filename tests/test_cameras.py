"""Tests for the frozen 4-view camera contract (Phase 1.2.4).

Verifies:
  - Frozen constants: 4 views, 320×320 square, 20 Hz
  - Wrist cameras track arm motion; overhead/third_person are static
  - RGB + depth + segmentation rendering per view
  - CameraIntrinsics computation (pinhole model from fovy)
  - Per-capture metadata (live extrinsics included in MultiViewFrame)

Run:
    pytest tests/test_cameras.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.camera import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CAPTURE_HZ,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    NUM_VIEWS,
    CameraIntrinsics,
    CameraMetadata,
    MultiViewFrame,
    MultiViewRenderer,
    compute_intrinsics,
    get_camera_metadata,
)
from sim.offscreen import resolve_camera_id  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alex_model():
    """Load the Alex upper-body scene (has overhead, third_person, wrist cams)."""
    from sim.asset_loader import load_scene

    return load_scene("alex_upper_body")


@pytest.fixture(scope="module")
def alex_data(alex_model):
    """Create MjData at rest (shared, read-only)."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    mujoco.mj_forward(alex_model, data)
    return data


@pytest.fixture()
def fresh_data(alex_model):
    """Fresh MjData for tests that mutate state."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    mujoco.mj_forward(alex_model, data)
    return data


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraContract:
    """Verify frozen 4-view camera constants and MJCF definitions."""

    def test_num_views_is_4(self) -> None:
        assert NUM_VIEWS == 4

    def test_camera_names_frozen(self) -> None:
        assert CAMERA_NAMES == ("overhead", "left_wrist_cam", "right_wrist_cam", "third_person")

    def test_default_resolution_square(self) -> None:
        assert DEFAULT_WIDTH == 320
        assert DEFAULT_HEIGHT == 320

    def test_default_capture_hz(self) -> None:
        assert DEFAULT_CAPTURE_HZ == 20.0

    def test_cameras_exist_in_model(self, alex_model) -> None:
        for name in CAMERA_NAMES:
            cam_id = resolve_camera_id(alex_model, name)
            assert cam_id >= 0, f"Camera '{name}' not found in model"


# ---------------------------------------------------------------------------
# Multi-view capture tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestMultiViewCapture:
    """Test synchronized 4-view rendering."""

    def test_capture_returns_all_views(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert isinstance(frame, MultiViewFrame)
            assert len(frame.views) == NUM_VIEWS
            for name in CAMERA_NAMES:
                assert name in frame.views

    def test_rgb_shapes_square(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                rgb = frame.views[name].rgb
                assert rgb.shape == (
                    DEFAULT_HEIGHT,
                    DEFAULT_WIDTH,
                    3,
                ), f"{name}: expected ({DEFAULT_HEIGHT},{DEFAULT_WIDTH},3), got {rgb.shape}"
                assert rgb.dtype == np.uint8

    def test_rgb_not_black(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                assert np.any(frame.views[name].rgb > 0), f"{name} rendered all-black"

    def test_step_index_matches(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=42)
            assert frame.step_index == 42
            for name in CAMERA_NAMES:
                assert frame.views[name].step_index == 42

    def test_timestamp_populated(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert frame.timestamp >= 0.0

    def test_custom_resolution(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model, width=160, height=120) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                assert frame.views[name].rgb.shape == (120, 160, 3)

    def test_depth_rendering_all_views(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model, render_depth=True) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                depth = frame.views[name].depth
                assert depth is not None, f"{name}: depth is None"
                assert depth.shape == (
                    DEFAULT_HEIGHT,
                    DEFAULT_WIDTH,
                ), f"{name}: depth shape {depth.shape}"
                assert np.all(np.isfinite(depth)), f"{name}: depth has non-finite values"

    def test_segmentation_rendering_all_views(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model, render_segmentation=True) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                seg = frame.views[name].segmentation
                assert seg is not None, f"{name}: segmentation is None"


# ---------------------------------------------------------------------------
# Synchronization tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraSync:
    """Verify multi-view synchronization contract."""

    def test_views_share_step_index(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=7)
            indices = {frame.views[n].step_index for n in CAMERA_NAMES}
            assert len(indices) == 1, f"Step indices differ across views: {indices}"

    def test_successive_captures_differ(self, alex_model, fresh_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame1 = mv.capture(fresh_data, step_index=0)
            mujoco.mj_step(alex_model, fresh_data)
            frame2 = mv.capture(fresh_data, step_index=1)
            assert frame1.step_index != frame2.step_index


# ---------------------------------------------------------------------------
# Camera movement tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestWristCameraMovement:
    """Verify wrist cameras track arm motion; fixed cameras are static."""

    def test_left_wrist_cam_tracks_arm(self, alex_model, fresh_data) -> None:
        cam_id = resolve_camera_id(alex_model, "left_wrist_cam")
        rest_pos = fresh_data.cam_xpos[cam_id].copy()

        spine_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
        qpos_adr = alex_model.jnt_qposadr[spine_id]
        fresh_data.qpos[qpos_adr] = -0.5
        mujoco.mj_forward(alex_model, fresh_data)

        moved_pos = fresh_data.cam_xpos[cam_id].copy()
        delta = np.linalg.norm(moved_pos - rest_pos)
        assert delta > 0.01, f"left_wrist_cam did not move with arm (delta={delta:.6f})"

    def test_right_wrist_cam_tracks_arm(self, alex_model, fresh_data) -> None:
        cam_id = resolve_camera_id(alex_model, "right_wrist_cam")
        rest_pos = fresh_data.cam_xpos[cam_id].copy()

        jnt_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_y")
        qpos_adr = alex_model.jnt_qposadr[jnt_id]
        fresh_data.qpos[qpos_adr] = -0.5
        mujoco.mj_forward(alex_model, fresh_data)

        moved_pos = fresh_data.cam_xpos[cam_id].copy()
        delta = np.linalg.norm(moved_pos - rest_pos)
        assert delta > 0.01, f"right_wrist_cam did not move with arm (delta={delta:.6f})"

    def test_wrist_cams_are_body_attached(self, alex_model, alex_data) -> None:
        """Wrist cameras should have non-zero body IDs (attached to gripper bodies)."""
        for name in ("left_wrist_cam", "right_wrist_cam"):
            cam_id = resolve_camera_id(alex_model, name)
            assert (
                int(alex_model.cam_bodyid[cam_id]) != 0
            ), f"{name}: cam_bodyid == 0 (camera is in world body, not arm body)"

    def test_overhead_is_static(self, alex_model, fresh_data) -> None:
        cam_id = resolve_camera_id(alex_model, "overhead")
        rest_pos = fresh_data.cam_xpos[cam_id].copy()

        jnt_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
        qpos_adr = alex_model.jnt_qposadr[jnt_id]
        fresh_data.qpos[qpos_adr] = -0.5
        mujoco.mj_forward(alex_model, fresh_data)

        moved_pos = fresh_data.cam_xpos[cam_id].copy()
        delta = np.linalg.norm(moved_pos - rest_pos)
        assert delta < 1e-10, f"overhead camera moved with arm (delta={delta:.6f})"

    def test_third_person_is_static(self, alex_model, fresh_data) -> None:
        cam_id = resolve_camera_id(alex_model, "third_person")
        rest_pos = fresh_data.cam_xpos[cam_id].copy()

        jnt_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
        qpos_adr = alex_model.jnt_qposadr[jnt_id]
        fresh_data.qpos[qpos_adr] = 0.4
        mujoco.mj_forward(alex_model, fresh_data)

        moved_pos = fresh_data.cam_xpos[cam_id].copy()
        delta = np.linalg.norm(moved_pos - rest_pos)
        assert delta < 1e-10, f"third_person camera moved (delta={delta:.6f})"

    def test_wrist_cams_move_with_spine(self, alex_model, fresh_data) -> None:
        """Wrist cams also move when spine_z rotates (children of torso chain)."""
        for name in ("left_wrist_cam", "right_wrist_cam"):
            cam_id = resolve_camera_id(alex_model, name)
            rest_pos = fresh_data.cam_xpos[cam_id].copy()

        jnt_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
        qpos_adr = alex_model.jnt_qposadr[jnt_id]
        fresh_data.qpos[qpos_adr] = 0.4
        mujoco.mj_forward(alex_model, fresh_data)

        for name in ("left_wrist_cam", "right_wrist_cam"):
            cam_id = resolve_camera_id(alex_model, name)
            moved_pos = fresh_data.cam_xpos[cam_id].copy()
            delta = np.linalg.norm(moved_pos - rest_pos)
            assert delta > 0.01, f"{name} did not move with spine_z (delta={delta:.6f})"


# ---------------------------------------------------------------------------
# Legacy camera test (robot_cam still in MJCF but not in contract)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestLegacyCameras:
    """robot_cam stays in the MJCF for debugging but is not in the frozen contract."""

    def test_robot_cam_still_exists_in_model(self, alex_model) -> None:
        cam_id = resolve_camera_id(alex_model, "robot_cam")
        assert cam_id >= 0, "robot_cam was removed from MJCF"

    def test_robot_cam_not_in_frozen_contract(self) -> None:
        assert "robot_cam" not in CAMERA_NAMES


# ---------------------------------------------------------------------------
# CameraIntrinsics tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraIntrinsics:
    """Test pinhole camera intrinsics computation."""

    def test_compute_intrinsics_square(self) -> None:
        intr = compute_intrinsics(60.0, 320, 320)
        # fy = 160 / tan(30°) ≈ 277.13
        assert intr.fy == pytest.approx(277.128, rel=0.01)
        assert intr.fx == pytest.approx(intr.fy)  # square pixels
        assert intr.cx == pytest.approx(160.0)
        assert intr.cy == pytest.approx(160.0)
        assert intr.fovy == 60.0
        # For square resolution, fovx == fovy
        assert intr.fovx == pytest.approx(60.0, rel=0.01)

    def test_compute_intrinsics_nonsquare(self) -> None:
        intr = compute_intrinsics(60.0, 320, 240)
        # fy = 120 / tan(30°) ≈ 207.85
        assert intr.fy == pytest.approx(207.846, rel=0.01)
        assert intr.fx == pytest.approx(intr.fy)  # MuJoCo square pixels
        assert intr.cy == pytest.approx(120.0)
        assert intr.cx == pytest.approx(160.0)
        # fovx > fovy because width > height
        assert intr.fovx > intr.fovy

    def test_intrinsics_matrix_shape(self) -> None:
        intr = compute_intrinsics(75.0, 320, 320)
        K = intr.to_matrix()
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0
        assert K[0, 1] == 0.0  # no skew
        assert K[1, 0] == 0.0

    def test_intrinsics_matrix_values(self) -> None:
        intr = compute_intrinsics(60.0, 320, 320)
        K = intr.to_matrix()
        assert K[0, 0] == pytest.approx(intr.fx)
        assert K[1, 1] == pytest.approx(intr.fy)
        assert K[0, 2] == pytest.approx(intr.cx)
        assert K[1, 2] == pytest.approx(intr.cy)

    def test_intrinsics_to_dict(self) -> None:
        intr = compute_intrinsics(60.0, 320, 320)
        d = intr.to_dict()
        assert set(d.keys()) == {"fx", "fy", "cx", "cy", "fovx", "fovy"}
        assert d["fovy"] == 60.0

    def test_metadata_has_intrinsics(self, alex_model, alex_data) -> None:
        meta = get_camera_metadata(alex_model, alex_data, "overhead")
        assert isinstance(meta.intrinsics, CameraIntrinsics)
        assert meta.intrinsics.fx > 0
        assert meta.intrinsics.fy > 0

    def test_wrist_cam_intrinsics(self, alex_model, alex_data) -> None:
        meta = get_camera_metadata(alex_model, alex_data, "left_wrist_cam")
        assert isinstance(meta.intrinsics, CameraIntrinsics)
        # fovy=75° → fy = 160 / tan(37.5°) ≈ 208.6
        assert meta.intrinsics.fy == pytest.approx(208.6, rel=0.02)

    def test_body_attached_flag_wrist(self, alex_model, alex_data) -> None:
        for name in ("left_wrist_cam", "right_wrist_cam"):
            meta = get_camera_metadata(alex_model, alex_data, name)
            assert meta.is_body_attached is True, f"{name}: expected is_body_attached=True"

    def test_body_attached_flag_fixed(self, alex_model, alex_data) -> None:
        for name in ("overhead", "third_person"):
            meta = get_camera_metadata(alex_model, alex_data, name)
            assert meta.is_body_attached is False, f"{name}: expected is_body_attached=False"


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraMetadata:
    """Test camera intrinsic/extrinsic metadata extraction."""

    def test_all_views_have_positive_fovy(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                assert meta[name].fovy > 0, f"{name} fovy not positive"

    def test_overhead_fovy(self, alex_model, alex_data) -> None:
        # alex_upper_body.xml: overhead fovy="62"
        meta = get_camera_metadata(alex_model, alex_data, "overhead")
        assert meta.fovy == pytest.approx(62.0)

    def test_third_person_fovy(self, alex_model, alex_data) -> None:
        # alex_upper_body.xml: third_person fovy="55"
        meta = get_camera_metadata(alex_model, alex_data, "third_person")
        assert meta.fovy == pytest.approx(55.0)

    def test_wrist_cam_fovy(self, alex_model, alex_data) -> None:
        for name in ("left_wrist_cam", "right_wrist_cam"):
            meta = get_camera_metadata(alex_model, alex_data, name)
            assert meta.fovy == pytest.approx(75.0), f"{name} fovy != 75"

    def test_metadata_pos_finite(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                assert np.all(np.isfinite(meta[name].pos)), f"{name} pos not finite"
                assert np.all(np.isfinite(meta[name].mat)), f"{name} mat not finite"

    def test_metadata_resolution(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                assert meta[name].width == DEFAULT_WIDTH
                assert meta[name].height == DEFAULT_HEIGHT

    def test_metadata_to_dict(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                d = meta[name].to_dict()
                assert isinstance(d, dict)
                assert d["name"] == name
                assert isinstance(d["pos"], list)
                assert len(d["pos"]) == 3
                assert "intrinsics" in d
                assert "is_body_attached" in d

    def test_get_camera_metadata_standalone(self, alex_model, alex_data) -> None:
        meta = get_camera_metadata(alex_model, alex_data, "overhead")
        assert isinstance(meta, CameraMetadata)
        assert meta.name == "overhead"
        assert meta.pos.shape == (3,)
        assert meta.mat.shape == (3, 3)


# ---------------------------------------------------------------------------
# Per-capture metadata (MultiViewFrame.metadata)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestMultiViewFrameMetadata:
    """Verify that MultiViewFrame includes live per-camera metadata."""

    def test_capture_includes_metadata(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert len(frame.metadata) == NUM_VIEWS
            for name in CAMERA_NAMES:
                assert name in frame.metadata
                assert isinstance(frame.metadata[name], CameraMetadata)

    def test_wrist_cam_metadata_updates_with_arm(self, alex_model, fresh_data) -> None:
        """Wrist camera metadata captures live position after arm movement."""
        with MultiViewRenderer(alex_model) as mv:
            frame1 = mv.capture(fresh_data, step_index=0)
            pos1 = frame1.metadata["left_wrist_cam"].pos.copy()

            jnt_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
            qpos_adr = alex_model.jnt_qposadr[jnt_id]
            fresh_data.qpos[qpos_adr] = -0.5
            mujoco.mj_forward(alex_model, fresh_data)

            frame2 = mv.capture(fresh_data, step_index=1)
            pos2 = frame2.metadata["left_wrist_cam"].pos.copy()
            delta = np.linalg.norm(pos2 - pos1)
            assert delta > 0.01, f"Wrist cam metadata not updated (delta={delta:.6f})"

    def test_fixed_cam_metadata_stable(self, alex_model, fresh_data) -> None:
        """Fixed camera metadata doesn't change when robot moves."""
        with MultiViewRenderer(alex_model) as mv:
            frame1 = mv.capture(fresh_data, step_index=0)
            pos1_overhead = frame1.metadata["overhead"].pos.copy()

            jnt_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
            qpos_adr = alex_model.jnt_qposadr[jnt_id]
            fresh_data.qpos[qpos_adr] = -0.5
            mujoco.mj_forward(alex_model, fresh_data)

            frame2 = mv.capture(fresh_data, step_index=1)
            pos2_overhead = frame2.metadata["overhead"].pos.copy()
            delta = np.linalg.norm(pos2_overhead - pos1_overhead)
            assert delta < 1e-10, f"Overhead position changed (delta={delta:.2e})"


# ---------------------------------------------------------------------------
# Context manager / lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestRendererLifecycle:
    """Test renderer creation, context manager, and cleanup."""

    def test_context_manager(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert len(frame.views) == NUM_VIEWS

    def test_closed_renderer_raises(self, alex_model, alex_data) -> None:
        mv = MultiViewRenderer(alex_model)
        mv.close()
        with pytest.raises(RuntimeError, match="closed"):
            mv.capture(alex_data, step_index=0)

    def test_camera_ids_property(self, alex_model) -> None:
        with MultiViewRenderer(alex_model) as mv:
            ids = mv.camera_ids
            assert len(ids) == NUM_VIEWS
            for name in CAMERA_NAMES:
                assert name in ids
                assert isinstance(ids[name], int)
                assert ids[name] >= 0
