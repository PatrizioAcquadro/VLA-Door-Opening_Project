"""Tests for robot state contract (Phase 1.1.6).

Run:
    pytest tests/test_robot_state.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.action_space import ACTION_DIM, ARM_DIM, GRIPPER_DIM, AlexActionSpace  # noqa: E402
from sim.robot_state import (  # noqa: E402
    EE_POSE_DIM,
    EE_VEL_DIM,
    GRIPPER_SLICE,
    GRIPPER_STATE_DIM,
    LEFT_EE_POS_SLICE,
    LEFT_EE_QUAT_SLICE,
    LEFT_EE_VEL_SLICE,
    Q_DIM,
    Q_DOT_DIM,
    Q_DOT_SLICE,
    Q_SLICE,
    RIGHT_EE_POS_SLICE,
    RIGHT_EE_QUAT_SLICE,
    RIGHT_EE_VEL_SLICE,
    STATE_DIM,
    AlexRobotState,
    RobotState,
)
from sim.sim_runner import SimRunner  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alex_model():
    """Load the Alex upper-body scene."""
    from sim.asset_loader import load_scene

    return load_scene("alex_upper_body")


@pytest.fixture(scope="module")
def alex_data(alex_model):
    """Create MjData for the Alex model (shared, read-only at rest)."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    mujoco.mj_forward(alex_model, data)
    return data


@pytest.fixture()
def fresh_data(alex_model):
    """Fresh MjData (not shared -- tests that mutate state use this)."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    mujoco.mj_forward(alex_model, data)
    return data


@pytest.fixture(scope="module")
def robot_state(alex_model):
    """Shared AlexRobotState instance."""
    return AlexRobotState(alex_model)


@pytest.fixture()
def runner(alex_model, fresh_data):
    """SimRunner with fresh data."""
    return SimRunner(alex_model, fresh_data)


# ===========================================================================
# Contract tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestStateContract:
    """Verify the frozen 52-D state contract."""

    def test_state_dim_is_52(self, robot_state: AlexRobotState) -> None:
        assert robot_state.state_dim == 52

    def test_component_dims_sum(self) -> None:
        assert Q_DIM + Q_DOT_DIM + GRIPPER_STATE_DIM + EE_POSE_DIM + EE_VEL_DIM == STATE_DIM

    def test_slice_coverage(self) -> None:
        """All slices cover exactly [0, 52) with no gaps or overlaps."""
        all_slices = [
            Q_SLICE,
            Q_DOT_SLICE,
            GRIPPER_SLICE,
            LEFT_EE_POS_SLICE,
            LEFT_EE_QUAT_SLICE,
            RIGHT_EE_POS_SLICE,
            RIGHT_EE_QUAT_SLICE,
            LEFT_EE_VEL_SLICE,
            RIGHT_EE_VEL_SLICE,
        ]
        covered = set()
        for sl in all_slices:
            indices = set(range(sl.start, sl.stop))
            assert not covered & indices, f"Overlap at {covered & indices}"
            covered |= indices
        assert covered == set(range(STATE_DIM))

    def test_arm_joint_ids_resolved(self, robot_state: AlexRobotState) -> None:
        assert robot_state._arm_jnt_ids.shape == (ARM_DIM,)
        assert np.all(robot_state._arm_jnt_ids >= 0)

    def test_ee_sites_resolved(self, robot_state: AlexRobotState) -> None:
        assert robot_state._left_ee_site_id >= 0
        assert robot_state._right_ee_site_id >= 0

    def test_norm_ranges_shapes(self, robot_state: AlexRobotState) -> None:
        nr = robot_state.norm_ranges
        assert nr.q_lo.shape == (Q_DIM,)
        assert nr.q_hi.shape == (Q_DIM,)
        assert nr.q_dot_lo.shape == (Q_DOT_DIM,)
        assert nr.q_dot_hi.shape == (Q_DOT_DIM,)
        assert nr.ee_pos_lo.shape == (3,)
        assert nr.ee_pos_hi.shape == (3,)
        assert isinstance(nr.ee_vel_max, float)


# ===========================================================================
# State extraction tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestStateExtraction:
    """Test get_state() correctness at rest."""

    def test_get_state_returns_robot_state(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        assert isinstance(state, RobotState)

    def test_get_flat_state_shape(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        assert flat.shape == (STATE_DIM,)

    def test_get_state_finite(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        assert np.all(np.isfinite(flat))

    def test_q_at_rest_is_zero(self, robot_state: AlexRobotState, alex_data) -> None:
        """Home keyframe has all-zero joint positions."""
        state = robot_state.get_state(alex_data)
        np.testing.assert_allclose(state.q, 0.0, atol=1e-10)

    def test_gripper_at_rest_bounded(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        assert np.all(state.gripper_state >= 0.0)
        assert np.all(state.gripper_state <= 1.0)

    def test_ee_quat_unit_norm(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        for name, quat in [("left", state.left_ee_quat), ("right", state.right_ee_quat)]:
            norm = float(np.linalg.norm(quat))
            np.testing.assert_allclose(norm, 1.0, atol=0.01, err_msg=f"{name}_ee_quat")

    def test_ee_positions_reasonable(self, robot_state: AlexRobotState, alex_data) -> None:
        """EE positions within plausible workspace."""
        state = robot_state.get_state(alex_data)
        for name, pos in [("left", state.left_ee_pos), ("right", state.right_ee_pos)]:
            # Robot base at [0, 0, 1.0], arms ~0.5-0.8m reach
            assert -1.5 < pos[0] < 1.5, f"{name} EE X = {pos[0]}"
            assert -1.5 < pos[1] < 1.5, f"{name} EE Y = {pos[1]}"
            assert 0.0 < pos[2] < 2.5, f"{name} EE Z = {pos[2]}"

    def test_ee_velocity_at_rest_near_zero(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        np.testing.assert_allclose(state.left_ee_vel, 0.0, atol=1e-6)
        np.testing.assert_allclose(state.right_ee_vel, 0.0, atol=1e-6)

    def test_q_dot_at_rest_near_zero(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        np.testing.assert_allclose(state.q_dot, 0.0, atol=1e-6)

    def test_timestamp_at_rest(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        assert state.timestamp == 0.0


# ===========================================================================
# State after motion tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestStateAfterMotion:
    """Verify state changes under actions."""

    def test_q_changes_after_action(self, alex_model, fresh_data) -> None:
        rs = AlexRobotState(alex_model)
        state_before = rs.get_state(fresh_data)

        action_space = AlexActionSpace(alex_model)
        action = np.zeros(ACTION_DIM)
        action[0] = 0.5  # spine delta
        action_space.apply_action(action, fresh_data)
        for _ in range(25):
            mujoco.mj_step(alex_model, fresh_data)

        state_after = rs.get_state(fresh_data)
        assert not np.allclose(state_after.q, state_before.q)

    def test_q_dot_nonzero_during_motion(self, alex_model, fresh_data) -> None:
        rs = AlexRobotState(alex_model)
        action_space = AlexActionSpace(alex_model)

        action = np.zeros(ACTION_DIM)
        action[1] = 0.8  # left shoulder_y
        action_space.apply_action(action, fresh_data)
        for _ in range(25):
            mujoco.mj_step(alex_model, fresh_data)

        state = rs.get_state(fresh_data)
        assert np.max(np.abs(state.q_dot)) > 0.01

    def test_ee_position_changes_after_action(self, alex_model, fresh_data) -> None:
        rs = AlexRobotState(alex_model)
        state_before = rs.get_state(fresh_data)

        runner = SimRunner(alex_model, fresh_data)
        action = np.zeros(ACTION_DIM)
        action[1] = 0.8  # left shoulder_y
        runner.step(action)

        state_after = rs.get_state(fresh_data)
        left_delta = np.linalg.norm(state_after.left_ee_pos - state_before.left_ee_pos)
        assert left_delta > 1e-4, f"Left EE moved only {left_delta} m"

    def test_ee_velocity_nonzero_during_motion(self, alex_model, fresh_data) -> None:
        rs = AlexRobotState(alex_model)
        runner = SimRunner(alex_model, fresh_data)

        action = np.zeros(ACTION_DIM)
        action[1] = 0.8
        runner.step(action)

        state = rs.get_state(fresh_data)
        assert np.linalg.norm(state.left_ee_vel) > 1e-4

    def test_gripper_state_tracks_command(self, alex_model, fresh_data) -> None:
        """Open gripper command leads to gripper_state near 1.0."""
        runner = SimRunner(alex_model, fresh_data)

        action = np.zeros(ACTION_DIM)
        action[ARM_DIM] = 1.0  # open left gripper
        action[ARM_DIM + 1] = 1.0  # open right gripper
        for _ in range(50):
            runner.step(action)

        state = runner.get_state()
        assert state.gripper_state[0] > 0.5, f"Left gripper = {state.gripper_state[0]}"
        assert state.gripper_state[1] > 0.5, f"Right gripper = {state.gripper_state[1]}"


# ===========================================================================
# Normalization tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestNormalization:
    """Normalization / denormalization round-trip and range checks."""

    def test_normalize_denormalize_roundtrip(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        normed = robot_state.normalize(flat)
        recovered = robot_state.denormalize(normed)
        # Quaternions pass through unchanged, so exact round-trip
        np.testing.assert_allclose(recovered, flat, atol=1e-10)

    def test_normalized_q_in_range(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        normed = robot_state.normalize(flat)
        q_normed = normed[Q_SLICE]
        assert np.all(q_normed >= -1.0 - 1e-6)
        assert np.all(q_normed <= 1.0 + 1e-6)

    def test_normalized_q_dot_in_range(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        normed = robot_state.normalize(flat)
        q_dot_normed = normed[Q_DOT_SLICE]
        assert np.all(q_dot_normed >= -1.0 - 1e-6)
        assert np.all(q_dot_normed <= 1.0 + 1e-6)

    def test_normalized_gripper_in_range(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        normed = robot_state.normalize(flat)
        gripper_normed = normed[GRIPPER_SLICE]
        assert np.all(gripper_normed >= -1.0 - 1e-6)
        assert np.all(gripper_normed <= 1.0 + 1e-6)

    def test_normalized_ee_vel_in_range(self, robot_state: AlexRobotState, alex_data) -> None:
        flat = robot_state.get_flat_state(alex_data)
        normed = robot_state.normalize(flat)
        for sl in [LEFT_EE_VEL_SLICE, RIGHT_EE_VEL_SLICE]:
            assert np.all(normed[sl] >= -1.0 - 1e-6)
            assert np.all(normed[sl] <= 1.0 + 1e-6)

    def test_quaternion_unchanged_by_normalization(
        self, robot_state: AlexRobotState, alex_data
    ) -> None:
        flat = robot_state.get_flat_state(alex_data)
        normed = robot_state.normalize(flat)
        np.testing.assert_array_equal(normed[LEFT_EE_QUAT_SLICE], flat[LEFT_EE_QUAT_SLICE])
        np.testing.assert_array_equal(normed[RIGHT_EE_QUAT_SLICE], flat[RIGHT_EE_QUAT_SLICE])

    def test_roundtrip_after_motion(self, alex_model, fresh_data) -> None:
        """Roundtrip works for non-rest states too."""
        rs = AlexRobotState(alex_model)
        runner = SimRunner(alex_model, fresh_data)

        rng = np.random.default_rng(42)
        action = rng.uniform(-0.5, 0.5, size=ACTION_DIM)
        action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
        for _ in range(20):
            runner.step(action)

        flat = rs.get_flat_state(fresh_data)
        normed = rs.normalize(flat)
        recovered = rs.denormalize(normed)
        np.testing.assert_allclose(recovered, flat, atol=1e-8)


# ===========================================================================
# RobotState dataclass tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestRobotStateDataclass:
    """Test RobotState dataclass methods."""

    def test_flat_roundtrip(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        flat = state.to_flat_array()
        recovered = RobotState.from_flat_array(flat, timestamp=state.timestamp)
        np.testing.assert_array_equal(recovered.to_flat_array(), flat)
        assert recovered.timestamp == state.timestamp

    def test_validate_at_rest(self, robot_state: AlexRobotState, alex_data) -> None:
        state = robot_state.get_state(alex_data)
        warnings = state.validate()
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    def test_validate_catches_nan(self) -> None:
        state = RobotState(
            q=np.full(Q_DIM, np.nan),
            q_dot=np.zeros(Q_DOT_DIM),
            gripper_state=np.zeros(GRIPPER_STATE_DIM),
            left_ee_pos=np.zeros(3),
            left_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            right_ee_pos=np.zeros(3),
            right_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            left_ee_vel=np.zeros(3),
            right_ee_vel=np.zeros(3),
        )
        warnings = state.validate()
        assert any("NaN" in w for w in warnings)

    def test_validate_catches_bad_quat(self) -> None:
        state = RobotState(
            q=np.zeros(Q_DIM),
            q_dot=np.zeros(Q_DOT_DIM),
            gripper_state=np.zeros(GRIPPER_STATE_DIM),
            left_ee_pos=np.zeros(3),
            left_ee_quat=np.array([0.0, 0.0, 0.0, 0.0]),  # zero norm
            right_ee_pos=np.zeros(3),
            right_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            left_ee_vel=np.zeros(3),
            right_ee_vel=np.zeros(3),
        )
        warnings = state.validate()
        assert any("left_ee_quat" in w for w in warnings)

    def test_from_flat_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected shape"):
            RobotState.from_flat_array(np.zeros(10))


# ===========================================================================
# SimRunner integration tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestSimRunnerIntegration:
    """Test SimRunner state return and get_state()."""

    def test_step_returns_robot_state(self, runner: SimRunner) -> None:
        state = runner.step(np.zeros(ACTION_DIM))
        assert isinstance(state, RobotState)

    def test_step_sequence_returns_states(self, runner: SimRunner) -> None:
        actions = np.zeros((5, ACTION_DIM))
        states = runner.step_sequence(actions)
        assert len(states) == 5
        assert all(isinstance(s, RobotState) for s in states)

    def test_get_state_method(self, runner: SimRunner) -> None:
        runner.step(np.zeros(ACTION_DIM))
        state = runner.get_state()
        assert isinstance(state, RobotState)

    def test_state_timestamps_increase(self, runner: SimRunner) -> None:
        states = runner.step_sequence(np.zeros((10, ACTION_DIM)))
        timestamps = [s.timestamp for s in states]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_state_q_consistent_with_action_space(self, alex_model, fresh_data) -> None:
        """q from get_state() matches action_space.get_current_state()[:15]."""
        runner = SimRunner(alex_model, fresh_data)
        runner.step(np.zeros(ACTION_DIM))

        robot_state = runner.get_state()
        action_state = runner.action_space.get_current_state(fresh_data)

        np.testing.assert_allclose(robot_state.q, action_state[:ARM_DIM], atol=1e-10)

    def test_state_gripper_consistent_with_action_space(self, alex_model, fresh_data) -> None:
        """gripper_state from get_state() matches action_space gripper output."""
        runner = SimRunner(alex_model, fresh_data)
        action = np.zeros(ACTION_DIM)
        action[ARM_DIM] = 0.7  # open left gripper partially
        for _ in range(20):
            runner.step(action)

        robot_state = runner.get_state()
        action_state = runner.action_space.get_current_state(fresh_data)

        np.testing.assert_allclose(robot_state.gripper_state, action_state[ARM_DIM:], atol=1e-10)


# ===========================================================================
# Determinism tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestDeterminism:
    """Verify deterministic state extraction."""

    def test_same_actions_same_state(self, alex_model) -> None:
        """Same action sequence produces identical states."""
        rng = np.random.default_rng(99)
        actions = rng.uniform(-0.5, 0.5, size=(50, ACTION_DIM))
        actions[:, ARM_DIM:] = rng.uniform(0, 1, size=(50, GRIPPER_DIM))

        final_states = []
        for _ in range(2):
            data = mujoco.MjData(alex_model)
            mujoco.mj_resetData(alex_model, data)
            runner = SimRunner(alex_model, data)
            states = runner.step_sequence(actions)
            final_states.append(states[-1].to_flat_array())

        np.testing.assert_array_equal(final_states[0], final_states[1])
