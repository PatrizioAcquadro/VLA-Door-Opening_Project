"""Tests for SAKE EZGripper Gen2 integration (Phase 1.1.3).

Run:
    pytest tests/test_ezgripper.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

# Expected EZGripper joint names
EZGRIPPER_JOINTS_LEFT = [
    "left_knuckle_palm_l1_1",
    "left_knuckle_l1_l2_1",
    "left_knuckle_palm_l1_2",
    "left_knuckle_l1_l2_2",
]
EZGRIPPER_JOINTS_RIGHT = [
    "right_knuckle_palm_l1_1",
    "right_knuckle_l1_l2_1",
    "right_knuckle_palm_l1_2",
    "right_knuckle_l1_l2_2",
]


@pytest.fixture(scope="module")
def alex_model():
    """Load the Alex upper-body scene."""
    from sim.asset_loader import load_scene

    return load_scene("alex_upper_body")


@pytest.fixture(scope="module")
def alex_data(alex_model):
    """Create MjData for the Alex model."""
    return mujoco.MjData(alex_model)


# ---------------------------------------------------------------------------
# A. Structural tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestEZGripperStructure:
    """Verify EZGripper MJCF structure is correct."""

    def test_joint_count_updated(self, alex_model) -> None:
        """15 arm + 8 EZGripper = 23 joints."""
        assert alex_model.njnt == 23

    def test_actuator_count_updated(self, alex_model) -> None:
        """15 arm + 2 EZGripper = 17 actuators."""
        assert alex_model.nu == 17

    def test_ezgripper_joints_exist(self, alex_model) -> None:
        for name in EZGRIPPER_JOINTS_LEFT + EZGRIPPER_JOINTS_RIGHT:
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert idx >= 0, f"Joint '{name}' not found"

    def test_ezgripper_actuators_exist(self, alex_model) -> None:
        for name in ["left_ezgripper", "right_ezgripper"]:
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert idx >= 0, f"Actuator '{name}' not found"

    def test_tool_frame_sites_exist(self, alex_model) -> None:
        for name in ["left_tool_frame", "right_tool_frame"]:
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_SITE, name)
            assert idx >= 0, f"Site '{name}' not found"

    def test_equality_constraints_exist(self, alex_model) -> None:
        """Should have 6 joint equality constraints (3 per hand)."""
        assert alex_model.neq >= 6

    def test_ezgripper_joint_ranges(self, alex_model) -> None:
        """All EZGripper joints should have range [0, 1.94]."""
        for name in EZGRIPPER_JOINTS_LEFT + EZGRIPPER_JOINTS_RIGHT:
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = alex_model.jnt_range[idx]
            assert abs(lo - 0.0) < 0.01, f"{name} lower: expected 0, got {lo}"
            assert abs(hi - 1.94) < 0.01, f"{name} upper: expected 1.94, got {hi}"

    def test_palm_collision_geoms_exist(self, alex_model) -> None:
        for name in ["left_palm_collision", "right_palm_collision"]:
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert idx >= 0, f"Geom '{name}' not found"

    def test_finger_pad_collision_geoms_exist(self, alex_model) -> None:
        for name in [
            "left_finger1_pad",
            "left_finger2_pad",
            "right_finger1_pad",
            "right_finger2_pad",
        ]:
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert idx >= 0, f"Geom '{name}' not found"


# ---------------------------------------------------------------------------
# B. Command interface tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestEZGripperCommand:
    """Verify the EZGripperInterface command layer."""

    def test_interface_instantiation(self, alex_model) -> None:
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(alex_model)
        ee_left = EZGripperInterface(alex_model, data, side="left")
        ee_right = EZGripperInterface(alex_model, data, side="right")
        assert ee_left.n_actuated_dof == 1
        assert ee_right.n_actuated_dof == 1

    def test_open_close_cycle(self, alex_model) -> None:
        """Close then open the gripper via the interface."""
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(alex_model)
        ee = EZGripperInterface(alex_model, data, side="left")

        # Close
        ee.set_grasp(0.0)
        for _ in range(500):
            mujoco.mj_step(alex_model, data)
        assert ee.get_grasp_state() < 0.1

        # Open
        ee.set_grasp(1.0)
        for _ in range(500):
            mujoco.mj_step(alex_model, data)
        assert ee.get_grasp_state() > 0.8

    def test_coupling_constraint(self, alex_model) -> None:
        """When primary joint moves, coupled joints follow."""
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(alex_model)
        ee = EZGripperInterface(alex_model, data, side="left")
        ee.set_grasp(0.5)
        for _ in range(500):
            mujoco.mj_step(alex_model, data)

        positions = []
        for name in EZGRIPPER_JOINTS_LEFT:
            jid = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx = alex_model.jnt_qposadr[jid]
            positions.append(data.qpos[qpos_idx])

        # All 4 joints should be within 0.1 rad of each other
        assert np.max(positions) - np.min(positions) < 0.1, f"Coupling mismatch: {positions}"

    def test_multiple_open_close_stable(self, alex_model) -> None:
        """10 open/close cycles without instability."""
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(alex_model)
        ee = EZGripperInterface(alex_model, data, side="left")

        for _ in range(10):
            ee.set_grasp(1.0)
            for _ in range(200):
                mujoco.mj_step(alex_model, data)
            ee.set_grasp(0.0)
            for _ in range(200):
                mujoco.mj_step(alex_model, data)

        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))

    def test_tool_frame_returns_valid_pose(self, alex_model) -> None:
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(alex_model)
        mujoco.mj_forward(alex_model, data)
        ee = EZGripperInterface(alex_model, data, side="left")
        pos, quat = ee.get_tool_frame_pose()
        assert pos.shape == (3,)
        assert quat.shape == (4,)
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(quat))
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-6

    def test_right_gripper_independent(self, alex_model) -> None:
        """Left and right grippers operate independently."""
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(alex_model)
        left = EZGripperInterface(alex_model, data, side="left")
        right = EZGripperInterface(alex_model, data, side="right")

        left.set_grasp(1.0)
        right.set_grasp(0.0)
        for _ in range(500):
            mujoco.mj_step(alex_model, data)

        assert left.get_grasp_state() > 0.7
        assert right.get_grasp_state() < 0.2


# ---------------------------------------------------------------------------
# C. Stability tests with EZGripper
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestEZGripperStability:
    """Verify the model is stable with EZGripper attached."""

    def test_gravity_settle_with_grippers(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        for _ in range(5000):
            mujoco.mj_step(alex_model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.max(np.abs(data.qpos)) < 10.0

    def test_no_penetration_at_home(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        mujoco.mj_forward(alex_model, data)
        for i in range(data.ncon):
            dist = data.contact[i].dist
            assert dist > -0.005, f"Contact {i} penetration {dist:.4f}m"

    def test_energy_bounded_with_grippers(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        for _ in range(2000):
            mujoco.mj_step(alex_model, data)
        total_energy = data.energy[0] + data.energy[1]
        assert np.isfinite(total_energy)
        assert total_energy < 1000.0

    def test_open_grippers_keyframe_stable(self, alex_model) -> None:
        """Load 'open_grippers' keyframe and run; verify stability."""
        data = mujoco.MjData(alex_model)
        key_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_KEY, "open_grippers")
        if key_id < 0:
            pytest.skip("No 'open_grippers' keyframe")
        mujoco.mj_resetDataKeyframe(alex_model, data, key_id)
        data.ctrl[:] = data.ctrl  # keep ctrl from keyframe
        for _ in range(2500):
            mujoco.mj_step(alex_model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))


# ---------------------------------------------------------------------------
# D. Grasp scene tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestEZGripperGraspScene:
    """Verify the grasp test scene loads and is stable."""

    @pytest.fixture
    def grasp_model(self):
        from sim.asset_loader import load_scene

        return load_scene("alex_grasp_test")

    def test_grasp_scene_loads(self, grasp_model) -> None:
        assert grasp_model.nq > 23  # 23 arm joints + 7 cube freejoint

    def test_cube_exists(self, grasp_model) -> None:
        idx = mujoco.mj_name2id(grasp_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube")
        assert idx >= 0

    def test_grasp_scene_stable(self, grasp_model) -> None:
        data = mujoco.MjData(grasp_model)
        for _ in range(2000):
            mujoco.mj_step(grasp_model, data)
        assert np.all(np.isfinite(data.qpos))

    def test_close_gripper_on_cube(self, grasp_model) -> None:
        """Close gripper in grasp scene; verify no explosion."""
        from sim.end_effector import EZGripperInterface

        data = mujoco.MjData(grasp_model)
        # Load pregrasp keyframe if available
        key_id = mujoco.mj_name2id(grasp_model, mujoco.mjtObj.mjOBJ_KEY, "pregrasp")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(grasp_model, data, key_id)

        ee = EZGripperInterface(grasp_model, data, side="left")
        # Close gripper
        ee.set_grasp(0.0)
        for _ in range(1000):
            mujoco.mj_step(grasp_model, data)

        assert np.all(np.isfinite(data.qpos)), "NaN during grasp"
        assert np.all(np.isfinite(data.qvel)), "NaN in qvel during grasp"


# ---------------------------------------------------------------------------
# E. Asset linting
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestEZGripperLintClean:
    """Verify EZGripper-related scenes pass asset linting."""

    def test_alex_model_lint_clean(self) -> None:
        from sim.asset_linter import Severity, lint_mjcf
        from sim.asset_loader import resolve_robot_path

        path = resolve_robot_path("alex")
        issues = lint_mjcf(path)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Lint errors: {errors}"

    def test_grasp_scene_lint_clean(self) -> None:
        from sim.asset_linter import Severity, lint_mjcf
        from sim.asset_loader import SCENES_DIR

        issues = lint_mjcf(SCENES_DIR / "alex_grasp_test.xml")
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Lint errors: {errors}"
