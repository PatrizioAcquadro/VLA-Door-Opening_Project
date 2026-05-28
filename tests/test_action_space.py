"""Tests for action space and sim runner (Phase 1.1.5).

Run:
    pytest tests/test_action_space.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.action_space import (  # noqa: E402
    _ACTUATOR_TO_JOINT,
    ACTION_DIM,
    ARM_ACTUATOR_NAMES,
    ARM_DIM,
    DEFAULT_CONTROL_HZ,
    DEFAULT_RATE_LIMIT_FACTOR,
    GRIPPER_ACTUATOR_NAMES,
    GRIPPER_DIM,
    AlexActionSpace,
)
from sim.control import JOINT_VELOCITY_LIMITS  # noqa: E402
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
    """Create MjData for the Alex model."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    return data


@pytest.fixture()
def fresh_data(alex_model):
    """Fresh MjData (not shared — tests that mutate state use this)."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    return data


@pytest.fixture(scope="module")
def action_space(alex_model):
    """Shared AlexActionSpace instance."""
    return AlexActionSpace(alex_model)


@pytest.fixture()
def runner(alex_model, fresh_data):
    """SimRunner with fresh data."""
    return SimRunner(alex_model, fresh_data)


# ===========================================================================
# Contract tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestActionSpaceContract:
    """Verify the frozen 17-D action contract."""

    def test_action_dim_is_17(self, action_space: AlexActionSpace) -> None:
        assert action_space.action_dim == 17

    def test_arm_dim_is_15(self, action_space: AlexActionSpace) -> None:
        assert action_space.arm_dim == 15

    def test_gripper_dim_is_2(self, action_space: AlexActionSpace) -> None:
        assert action_space.gripper_dim == 2

    def test_arm_joint_ordering(self, alex_model) -> None:
        """Arm joint ordering matches the frozen canonical list."""
        for name in ARM_ACTUATOR_NAMES:
            act_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert act_id >= 0, f"Actuator '{name}' not found"

    def test_gripper_actuators_exist(self, alex_model) -> None:
        for name in GRIPPER_ACTUATOR_NAMES:
            act_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert act_id >= 0, f"Actuator '{name}' not found"

    def test_delta_q_max_positive(self, action_space: AlexActionSpace) -> None:
        assert np.all(action_space.delta_q_max > 0)

    def test_delta_q_max_values(self, action_space: AlexActionSpace) -> None:
        """Verify delta_q_max matches vel_limit * control_dt * rate_limit."""
        dqm = action_space.delta_q_max
        control_dt = 1.0 / DEFAULT_CONTROL_HZ
        for i, act_name in enumerate(ARM_ACTUATOR_NAMES):
            jnt_name = _ACTUATOR_TO_JOINT.get(act_name, act_name)
            vel = JOINT_VELOCITY_LIMITS.get(jnt_name, 25.0)
            expected = vel * control_dt * DEFAULT_RATE_LIMIT_FACTOR
            np.testing.assert_allclose(
                dqm[i],
                expected,
                atol=1e-10,
                err_msg=f"delta_q_max mismatch for {act_name}",
            )

    def test_delta_q_max_shape(self, action_space: AlexActionSpace) -> None:
        assert action_space.delta_q_max.shape == (ARM_DIM,)

    def test_control_hz(self, action_space: AlexActionSpace) -> None:
        assert action_space.control_hz == DEFAULT_CONTROL_HZ

    def test_control_dt(self, action_space: AlexActionSpace) -> None:
        np.testing.assert_allclose(action_space.control_dt, 0.05)


# ===========================================================================
# Normalization tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestNormalization:
    """Normalization / denormalization round-trip and semantics."""

    def test_normalize_denormalize_roundtrip(self, action_space: AlexActionSpace) -> None:
        rng = np.random.default_rng(42)
        delta_q = rng.uniform(-0.1, 0.1, size=ARM_DIM)
        gripper = rng.uniform(0, 1, size=GRIPPER_DIM)

        action = action_space.normalize(delta_q, gripper)
        delta_q_out, gripper_out = action_space.denormalize(action)

        np.testing.assert_allclose(delta_q_out, delta_q, atol=1e-10)
        np.testing.assert_allclose(gripper_out, gripper, atol=1e-10)

    def test_denormalize_clips_arm(self, action_space: AlexActionSpace) -> None:
        """Actions outside [-1, 1] are clipped before denormalizing."""
        action = np.full(ACTION_DIM, 2.0)
        delta_q, _ = action_space.denormalize(action)
        np.testing.assert_allclose(delta_q, action_space.delta_q_max)

    def test_denormalize_clips_gripper(self, action_space: AlexActionSpace) -> None:
        action = np.full(ACTION_DIM, -1.0)
        _, gripper = action_space.denormalize(action)
        np.testing.assert_allclose(gripper, [0.0, 0.0])

    def test_zero_action_zero_delta(self, action_space: AlexActionSpace) -> None:
        action = np.zeros(ACTION_DIM)
        delta_q, gripper = action_space.denormalize(action)
        np.testing.assert_allclose(delta_q, np.zeros(ARM_DIM))
        np.testing.assert_allclose(gripper, [0.0, 0.0])

    def test_unit_action_equals_delta_q_max(self, action_space: AlexActionSpace) -> None:
        action = np.ones(ACTION_DIM)
        delta_q, gripper = action_space.denormalize(action)
        np.testing.assert_allclose(delta_q, action_space.delta_q_max)
        np.testing.assert_allclose(gripper, [1.0, 1.0])

    def test_negative_unit_action(self, action_space: AlexActionSpace) -> None:
        action = -np.ones(ACTION_DIM)
        action[ARM_DIM:] = 0.0  # grippers clip to 0
        delta_q, gripper = action_space.denormalize(action)
        np.testing.assert_allclose(delta_q, -action_space.delta_q_max)
        np.testing.assert_allclose(gripper, [0.0, 0.0])


# ===========================================================================
# Apply action tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestApplyAction:
    """Test action application pipeline."""

    def test_apply_zero_action_stable(
        self, alex_model, fresh_data, action_space: AlexActionSpace
    ) -> None:
        """Zero action should keep ctrl near current qpos."""
        mujoco.mj_forward(alex_model, fresh_data)
        qpos_before = fresh_data.qpos.copy()

        action = np.zeros(ACTION_DIM)
        action_space.apply_action(action, fresh_data)

        # ctrl should be close to current qpos (zero delta)
        for i in range(alex_model.nu):
            jnt_id = alex_model.actuator_trnid[i, 0]
            np.testing.assert_allclose(fresh_data.ctrl[i], qpos_before[jnt_id], atol=1e-8)

    def test_apply_action_changes_ctrl(
        self, alex_model, fresh_data, action_space: AlexActionSpace
    ) -> None:
        """Non-zero action changes ctrl values."""
        mujoco.mj_forward(alex_model, fresh_data)

        action = np.zeros(ACTION_DIM)
        action[0] = 0.5  # half-max delta on spine_z
        action_space.apply_action(action, fresh_data)

        # spine_z actuator (index 0) should have moved
        spine_act_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "spine_z")
        assert fresh_data.ctrl[spine_act_id] != 0.0

    def test_apply_action_respects_joint_limits(self, alex_model, fresh_data) -> None:
        """Target position stays within joint limits even with large deltas."""
        action_space = AlexActionSpace(alex_model)
        mujoco.mj_forward(alex_model, fresh_data)

        # Saturate in positive direction for many steps
        action = np.ones(ACTION_DIM)
        for _ in range(500):
            action_space.apply_action(action, fresh_data)
            # Step physics to update qpos
            for _ in range(25):
                mujoco.mj_step(alex_model, fresh_data)

        # All ctrl values must be within joint limits
        for i in range(alex_model.nu):
            jnt_id = alex_model.actuator_trnid[i, 0]
            lo, hi = alex_model.jnt_range[jnt_id]
            assert (
                fresh_data.ctrl[i] >= lo - 1e-6
            ), f"ctrl[{i}] = {fresh_data.ctrl[i]} below limit {lo}"
            assert (
                fresh_data.ctrl[i] <= hi + 1e-6
            ), f"ctrl[{i}] = {fresh_data.ctrl[i]} above limit {hi}"

    def test_gripper_action_independent(
        self, alex_model, fresh_data, action_space: AlexActionSpace
    ) -> None:
        """Gripper actions don't affect arm ctrl and vice versa."""
        mujoco.mj_forward(alex_model, fresh_data)

        # Apply arm-only action
        action_arm = np.zeros(ACTION_DIM)
        action_arm[0] = 0.5
        action_space.apply_action(action_arm, fresh_data)
        fresh_data.ctrl.copy()  # verify ctrl was written

        # Reset
        mujoco.mj_resetData(alex_model, fresh_data)
        mujoco.mj_forward(alex_model, fresh_data)

        # Apply gripper-only action
        action_grip = np.zeros(ACTION_DIM)
        action_grip[ARM_DIM] = 1.0  # open left gripper
        action_space.apply_action(action_grip, fresh_data)
        ctrl_grip_only = fresh_data.ctrl.copy()

        # Arm actuators should be unchanged by gripper action
        for name in ARM_ACTUATOR_NAMES:
            act_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            np.testing.assert_allclose(
                ctrl_grip_only[act_id],
                0.0,
                atol=1e-8,
                err_msg=f"Gripper action affected arm actuator {name}",
            )

    def test_get_current_state_shape(
        self, alex_model, fresh_data, action_space: AlexActionSpace
    ) -> None:
        state = action_space.get_current_state(fresh_data)
        assert state.shape == (ACTION_DIM,)

    def test_get_current_state_grippers_bounded(
        self, alex_model, fresh_data, action_space: AlexActionSpace
    ) -> None:
        state = action_space.get_current_state(fresh_data)
        assert np.all(state[ARM_DIM:] >= 0.0)
        assert np.all(state[ARM_DIM:] <= 1.0)


# ===========================================================================
# SimRunner tests
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestSimRunner:
    """Test fixed-rate simulation runner."""

    def test_substep_count(self, runner: SimRunner) -> None:
        """20 Hz with 0.002s timestep = 25 substeps."""
        assert runner.substeps == 25

    def test_control_period(self, runner: SimRunner) -> None:
        np.testing.assert_allclose(runner.control_dt, 0.05)

    def test_step_advances_time(self, runner: SimRunner) -> None:
        t0 = runner.sim_time
        runner.step(np.zeros(ACTION_DIM))
        np.testing.assert_allclose(runner.sim_time - t0, 0.05, atol=1e-10)

    def test_step_sequence_advances_time(self, runner: SimRunner) -> None:
        t0 = runner.sim_time
        actions = np.zeros((10, ACTION_DIM))
        runner.step_sequence(actions)
        np.testing.assert_allclose(runner.sim_time - t0, 0.5, atol=1e-8)

    def test_step_with_zero_action_stable(self, runner: SimRunner) -> None:
        """100 zero-action steps — no NaN, no explosion."""
        for _ in range(100):
            runner.step(np.zeros(ACTION_DIM))

        assert np.all(np.isfinite(runner._data.qpos))
        assert np.all(np.isfinite(runner._data.qvel))
        energy = runner._data.energy[0] + runner._data.energy[1]
        assert energy < 1000.0, f"Energy exploded: {energy}"

    def test_random_actions_stable(self, alex_model, fresh_data) -> None:
        """200 random actions in [-1, 1] — no NaN, bounded energy."""
        runner = SimRunner(alex_model, fresh_data)
        rng = np.random.default_rng(123)

        for _ in range(200):
            action = rng.uniform(-1, 1, size=ACTION_DIM)
            action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
            runner.step(action)

        assert np.all(np.isfinite(fresh_data.qpos)), "NaN in qpos"
        assert np.all(np.isfinite(fresh_data.qvel)), "NaN in qvel"

        # Check no massive energy
        energy = fresh_data.energy[0] + fresh_data.energy[1]
        assert energy < 1000.0, f"Energy exploded: {energy}"

        # Check no deep penetration
        for i in range(fresh_data.ncon):
            dist = fresh_data.contact[i].dist
            assert dist > -0.05, f"Deep penetration: {dist} m at contact {i}"

    def test_deterministic_replay(self, alex_model) -> None:
        """Same action sequence → identical final state."""
        rng = np.random.default_rng(99)
        actions = rng.uniform(-0.5, 0.5, size=(50, ACTION_DIM))
        actions[:, ARM_DIM:] = rng.uniform(0, 1, size=(50, GRIPPER_DIM))

        final_states = []
        for _ in range(2):
            data = mujoco.MjData(alex_model)
            mujoco.mj_resetData(alex_model, data)
            runner = SimRunner(alex_model, data)
            runner.step_sequence(actions)
            final_states.append(data.qpos.copy())

        np.testing.assert_array_equal(final_states[0], final_states[1])

    def test_invalid_control_rate_raises(self, alex_model) -> None:
        """Non-integer substep count should raise ValueError."""
        data = mujoco.MjData(alex_model)
        with pytest.raises(ValueError, match="not an integer multiple"):
            SimRunner(alex_model, data, control_hz=13.0)  # 0.002 * 13 ≠ integer


# ===========================================================================
# Action chunk tests (h=16)
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestActionChunks:
    """Stability under repeated action chunks (horizon h=16)."""

    def test_smooth_sinusoidal_chunk(self, alex_model) -> None:
        """Smooth sinusoidal trajectory over one chunk — bounded motion."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        chunk_size = 16
        t = np.linspace(0, 2 * np.pi, chunk_size)
        actions = np.zeros((chunk_size, ACTION_DIM))
        # Gentle sinusoidal motion on a few joints
        actions[:, 0] = 0.3 * np.sin(t)  # spine
        actions[:, 1] = 0.2 * np.sin(t + 0.5)  # left_shoulder_y
        actions[:, 8] = 0.2 * np.sin(t + 0.5)  # right_shoulder_y

        runner.step_sequence(actions)

        assert np.all(np.isfinite(data.qpos))
        # Velocity should stay reasonable
        max_vel = np.max(np.abs(data.qvel))
        assert max_vel < 50.0, f"Excessive velocity: {max_vel} rad/s"

    def test_repeated_chunks_stable(self, alex_model) -> None:
        """10 repeated random chunks (h=16) — stable after 160 control steps."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        rng = np.random.default_rng(42)
        chunk = rng.uniform(-0.3, 0.3, size=(16, ACTION_DIM))
        chunk[:, ARM_DIM:] = rng.uniform(0.2, 0.8, size=(16, GRIPPER_DIM))

        for _ in range(10):
            runner.step_sequence(chunk)

        assert np.all(np.isfinite(data.qpos)), "NaN after repeated chunks"
        energy = data.energy[0] + data.energy[1]
        assert energy < 1000.0, f"Energy after 10 chunks: {energy}"

    def test_alternating_chunks_no_explosion(self, alex_model) -> None:
        """Alternate between two different random chunks — no instability."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        rng = np.random.default_rng(77)
        chunk_a = rng.uniform(-0.5, 0.5, size=(16, ACTION_DIM))
        chunk_a[:, ARM_DIM:] = rng.uniform(0, 1, size=(16, GRIPPER_DIM))
        chunk_b = rng.uniform(-0.5, 0.5, size=(16, ACTION_DIM))
        chunk_b[:, ARM_DIM:] = rng.uniform(0, 1, size=(16, GRIPPER_DIM))

        for i in range(10):
            runner.step_sequence(chunk_a if i % 2 == 0 else chunk_b)

        assert np.all(np.isfinite(data.qpos)), "NaN after alternating chunks"
        assert np.all(np.isfinite(data.qvel)), "NaN vel after alternating chunks"


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.mujoco
@pytest.mark.assets
class TestEdgeCases:
    """Edge cases: saturated actions, rapid reversals."""

    def test_max_action_all_ones(self, alex_model) -> None:
        """Saturated positive action for 50 steps — hits limits, no explosion."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        action = np.ones(ACTION_DIM)
        for _ in range(50):
            runner.step(action)

        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))

    def test_min_action_all_negative(self, alex_model) -> None:
        """Saturated negative action for 50 steps."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        action = -np.ones(ACTION_DIM)
        action[ARM_DIM:] = 0.0  # grippers closed
        for _ in range(50):
            runner.step(action)

        assert np.all(np.isfinite(data.qpos))

    def test_rapid_direction_reversal(self, alex_model) -> None:
        """Alternate +1/-1 actions every step for 100 steps — PD absorbs."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        for i in range(100):
            sign = 1.0 if i % 2 == 0 else -1.0
            action = np.full(ACTION_DIM, sign * 0.8)
            action[ARM_DIM:] = 0.5  # grippers neutral
            runner.step(action)

        assert np.all(np.isfinite(data.qpos)), "NaN after rapid reversals"
        assert np.all(np.isfinite(data.qvel)), "NaN vel after rapid reversals"
        energy = data.energy[0] + data.energy[1]
        assert energy < 2000.0, f"High energy after reversals: {energy}"

    def test_single_action_via_step_sequence(self, alex_model) -> None:
        """step_sequence with a 1-D action (not 2-D) works."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_resetData(alex_model, data)
        runner = SimRunner(alex_model, data)

        action = np.zeros(ACTION_DIM)
        runner.step_sequence(action)  # 1-D, should be handled
        assert np.all(np.isfinite(data.qpos))
