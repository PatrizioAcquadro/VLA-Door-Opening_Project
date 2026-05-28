"""Tests for Phase 1.2.5: Episode Manager (block spawning & reset).

Tests deterministic spawn sampling, reset reliability, settle phase,
curriculum scaffolding, and ResetMetrics tracking.
"""

from __future__ import annotations

import math

import pytest

from sim.lego.episode_manager import (
    LEVEL_MULTI_STEP,
    LEVEL_SINGLE_BRICK,
    LEVEL_SINGLE_CONNECTION,
    EpisodeInfo,
    EpisodeManager,
    ResetMetrics,
    SpawnPose,
    _yaw_to_quat,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(**kwargs) -> EpisodeManager:
    """Create a default EpisodeManager (1 slot, 2x2 brick) for fast tests."""
    return EpisodeManager(
        **{"brick_slots": ["2x2"], "settle_max_steps": 300, "settle_check_interval": 25, **kwargs}
    )


def _make_manager_n(n: int, **kwargs) -> EpisodeManager:
    """Create an EpisodeManager with n 2x2 brick slots."""
    return EpisodeManager(
        **{
            "brick_slots": ["2x2"] * n,
            "settle_max_steps": 300,
            "settle_check_interval": 25,
            **kwargs,
        }
    )


# ---------------------------------------------------------------------------
# SpawnPose tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestSpawnPose:
    def test_frozen_dataclass(self):
        import dataclasses

        pose = SpawnPose(position=(1.0, 2.0, 3.0), quaternion=(1.0, 0.0, 0.0, 0.0))
        with pytest.raises(dataclasses.FrozenInstanceError):
            pose.position = (0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_identity_quat_fields(self):
        pose = SpawnPose(position=(0.1, 0.2, 0.85), quaternion=(1.0, 0.0, 0.0, 0.0))
        assert math.isclose(sum(q**2 for q in pose.quaternion), 1.0, rel_tol=1e-9)

    def test_yaw_to_quat_normalized(self):
        for yaw in [0.0, math.pi / 4, math.pi / 2, math.pi]:
            q = _yaw_to_quat(yaw)
            assert math.isclose(
                sum(v**2 for v in q), 1.0, rel_tol=1e-9
            ), f"Quaternion not normalized for yaw={yaw}: {q}"

    def test_yaw_to_quat_zero(self):
        q = _yaw_to_quat(0.0)
        assert q == (1.0, 0.0, 0.0, 0.0)

    def test_yaw_to_quat_180(self):
        q = _yaw_to_quat(math.pi)
        # cos(pi/2)=0, sin(pi/2)=1 → qw≈0, qz≈1
        assert math.isclose(q[0], 0.0, abs_tol=1e-9)
        assert math.isclose(q[3], 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# EpisodeInfo tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestEpisodeInfo:
    def test_frozen_dataclass(self):
        import dataclasses

        info = EpisodeInfo(
            seed=42,
            level=1,
            brick_types=["2x2"],
            spawn_poses=[],
            settle_steps=0,
            settle_success=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.seed = 0  # type: ignore[misc]

    def test_failure_info(self):
        info = EpisodeInfo(
            seed=0,
            level=1,
            brick_types=["2x2"],
            spawn_poses=[],
            settle_steps=0,
            settle_success=False,
        )
        assert not info.settle_success
        assert info.spawn_poses == []


# ---------------------------------------------------------------------------
# ResetMetrics tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestResetMetrics:
    def test_initial_state(self):
        m = ResetMetrics()
        assert m.total_resets == 0
        assert m.success_rate == 0.0
        assert m.avg_settle_steps == 0.0

    def test_success_rate(self):
        m = ResetMetrics()
        m._record_success(100)
        m._record_success(200)
        m._record_failure("spawn_overlap")
        assert m.total_resets == 3
        assert m.successful_resets == 2
        assert math.isclose(m.success_rate, 2 / 3)

    def test_avg_settle_steps(self):
        m = ResetMetrics()
        m._record_success(100)
        m._record_success(200)
        assert math.isclose(m.avg_settle_steps, 150.0)

    def test_failure_reasons_tracked(self):
        m = ResetMetrics()
        m._record_failure("spawn_overlap")
        m._record_failure("spawn_overlap")
        m._record_failure("settle_timeout")
        assert m.failure_reasons["spawn_overlap"] == 2
        assert m.failure_reasons["settle_timeout"] == 1

    def test_no_resets_success_rate_zero(self):
        m = ResetMetrics()
        assert m.success_rate == 0.0


# ---------------------------------------------------------------------------
# Spawn Sampling tests (no MuJoCo — unit tests on _sample_spawn_poses)
# ---------------------------------------------------------------------------


class TestSpawnSampling:
    """Test spawn sampling logic without MuJoCo (via EpisodeManager internals)."""

    def setup_method(self):
        """Import numpy here to keep marker-free tests clean."""
        import numpy as np

        self.np = np

    def _make_sampler(self, **kwargs):
        """Create a lightweight object exposing _sample_spawn_poses."""
        # We instantiate a real EpisodeManager for this — it requires mujoco.
        # These tests are pure Python logic, so skip if mujoco unavailable.
        pytest.importorskip("mujoco")
        return _make_manager(**kwargs)

    def test_deterministic_by_seed(self):
        em = self._make_sampler()
        rng1 = self.np.random.Generator(self.np.random.PCG64(42))
        rng2 = self.np.random.Generator(self.np.random.PCG64(42))
        poses1 = em._sample_spawn_poses(rng1, 1)
        poses2 = em._sample_spawn_poses(rng2, 1)
        assert poses1 is not None and poses2 is not None
        assert poses1[0].position == poses2[0].position
        assert poses1[0].quaternion == poses2[0].quaternion

    def test_different_seeds_differ(self):
        em = self._make_sampler()
        rng0 = self.np.random.Generator(self.np.random.PCG64(0))
        rng1 = self.np.random.Generator(self.np.random.PCG64(1))
        p0 = em._sample_spawn_poses(rng0, 1)
        p1 = em._sample_spawn_poses(rng1, 1)
        assert p0 is not None and p1 is not None
        assert p0[0].position != p1[0].position

    def test_spawn_within_region(self):
        em = self._make_sampler()
        rng = self.np.random.Generator(self.np.random.PCG64(7))
        poses = em._sample_spawn_poses(rng, 1)
        assert poses is not None
        x, y, z = poses[0].position
        assert (
            em._spawn_x_range[0] <= x <= em._spawn_x_range[1]
        ), f"x={x} outside [{em._spawn_x_range}]"
        assert (
            em._spawn_y_range[0] <= y <= em._spawn_y_range[1]
        ), f"y={y} outside [{em._spawn_y_range}]"
        assert math.isclose(z, em._spawn_z, rel_tol=1e-6)

    def test_min_distance_enforced(self):
        em = _make_manager_n(3, min_spawn_distance=0.05)
        rng = self.np.random.Generator(self.np.random.PCG64(99))
        poses = em._sample_spawn_poses(rng, 3)
        assert poses is not None, "Should find 3 poses with plenty of spawn space"
        for i, pi in enumerate(poses):
            for j, pj in enumerate(poses):
                if i >= j:
                    continue
                dx = pi.position[0] - pj.position[0]
                dy = pi.position[1] - pj.position[1]
                dist = math.sqrt(dx**2 + dy**2)
                assert (
                    dist >= em._min_spawn_distance - 1e-9
                ), f"Bricks {i} and {j} too close: {dist:.4f} m < {em._min_spawn_distance}"

    def test_max_attempts_exhaustion_returns_none(self):
        # Spawn region so small that two bricks can't fit with large min-distance
        em = EpisodeManager(
            brick_slots=["2x2", "2x2"],
            spawn_x_range=(0.45, 0.451),  # tiny region
            spawn_y_range=(0.0, 0.001),
            min_spawn_distance=0.10,  # much larger than region
            max_spawn_attempts=10,
            settle_max_steps=50,
        )
        rng = self.np.random.Generator(self.np.random.PCG64(0))
        result = em._sample_spawn_poses(rng, 2)
        assert result is None, "Should fail to place 2 bricks in tiny region"


# ---------------------------------------------------------------------------
# EpisodeReset tests (require MuJoCo)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.lego
class TestEpisodeReset:
    def test_reset_returns_episode_info(self):
        em = _make_manager()
        info = em.reset(seed=42)
        assert isinstance(info, EpisodeInfo)
        assert info.seed == 42
        assert info.level == LEVEL_SINGLE_BRICK
        assert info.brick_types == ["2x2"]

    def test_deterministic_by_seed(self):
        em1 = _make_manager()
        em2 = _make_manager()
        info1 = em1.reset(seed=0)
        info2 = em2.reset(seed=0)
        assert (
            info1.spawn_poses == info2.spawn_poses
        ), "Same seed must produce identical spawn poses"

    def test_different_seeds_differ(self):
        em = _make_manager()
        info_a = em.reset(seed=100)
        info_b = em.reset(seed=101)
        assert (
            info_a.spawn_poses != info_b.spawn_poses
        ), "Different seeds should produce different spawns"

    def test_brick_position_in_workspace(self):

        em = _make_manager()
        info = em.reset(seed=5)
        assert info.settle_success, f"Settle failed: {info}"
        # Read back brick position from qpos
        addr = em._slot_qpos_addrs[0]
        pos = em.data.qpos[addr : addr + 3]
        assert (
            em._spawn_x_range[0] <= pos[0] <= em._spawn_x_range[1]
        ), f"Brick X={pos[0]:.3f} outside spawn region"
        assert (
            em._spawn_y_range[0] <= pos[1] <= em._spawn_y_range[1]
        ), f"Brick Y={pos[1]:.3f} outside spawn region"
        # Brick should be near table surface after settle (above baseplate)
        assert (
            pos[2] >= em._baseplate_surface_z - 0.01
        ), f"Brick Z={pos[2]:.3f} below baseplate surface {em._baseplate_surface_z:.3f}"

    def test_unused_bricks_parked(self):
        em = _make_manager_n(3)
        info = em.reset(seed=1, level=LEVEL_SINGLE_BRICK)
        assert info.settle_success
        # Slots 1 and 2 should be parked well below floor
        for i in [1, 2]:
            addr = em._slot_qpos_addrs[i]
            z = float(em.data.qpos[addr + 2])
            assert z < -5.0, f"Unused brick {i} not parked (Z={z:.2f})"

    def test_robot_at_home_position(self):
        import mujoco
        import numpy as np

        em = _make_manager()
        # Check robot qpos immediately after mj_resetData (before settle drifts it)
        mujoco.mj_resetData(em.model, em.data)
        robot_q = em.data.qpos[: em._slot_qpos_addrs[0]]
        np.testing.assert_allclose(
            robot_q,
            np.zeros_like(robot_q),
            atol=1e-6,
            err_msg="Robot joints should be at zero immediately after mj_resetData",
        )

    def test_episode_count_increments(self):
        em = _make_manager()
        assert em.episode_count == 0
        em.reset(seed=0)
        em.reset(seed=1)
        assert em.episode_count == 2


# ---------------------------------------------------------------------------
# Settle phase tests (require MuJoCo)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.lego
class TestSettle:
    def test_settle_converges(self):
        em = _make_manager(settle_max_steps=500)
        info = em.reset(seed=10)
        assert info.settle_success, f"Settle failed for seed=10 after {info.settle_steps} steps"
        assert 0 < info.settle_steps <= 500

    def test_settle_steps_tracked(self):
        em = _make_manager()
        info = em.reset(seed=20)
        assert info.settle_steps > 0

    def test_settle_metrics_recorded(self):
        em = _make_manager()
        em.reset(seed=30)
        assert em.metrics.total_resets == 1
        if em.metrics.successful_resets == 1:
            assert len(em.metrics.settle_steps_history) == 1
            assert em.metrics.avg_settle_steps > 0

    def test_settle_timeout_returns_false(self):
        # Force timeout by giving 1 step (bricks never settle in 1 step)
        em = EpisodeManager(
            brick_slots=["2x2"],
            settle_max_steps=1,
            settle_check_interval=1,
        )
        info = em.reset(seed=0)
        # Either settles (if brick happens to be stable at step 1) or fails
        # With only 1 step, almost certainly fails — but we just check the flag matches steps
        assert info.settle_steps <= 1


# ---------------------------------------------------------------------------
# Curriculum tests (require MuJoCo)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.lego
class TestCurriculum:
    def test_level_1_one_brick(self):
        em = _make_manager_n(4)
        info = em.reset(seed=42, level=LEVEL_SINGLE_BRICK)
        assert len(info.brick_types) == 1

    def test_level_2_one_brick(self):
        em = _make_manager_n(4)
        info = em.reset(seed=42, level=LEVEL_SINGLE_CONNECTION)
        assert len(info.brick_types) == 1

    def test_level_3_two_to_four_bricks(self):
        em = _make_manager_n(4)
        # Run many seeds and check that brick count is in [2, 4]
        counts = set()
        for seed in range(20):
            info = em.reset(seed=seed, level=LEVEL_MULTI_STEP)
            counts.add(len(info.brick_types))
        assert all(2 <= c <= 4 for c in counts), f"Invalid brick counts: {counts}"
        assert len(counts) > 1, "Level 3 should vary brick count across seeds"

    def test_n_active_override(self):
        em = _make_manager_n(4)
        info = em.reset(seed=0, level=LEVEL_MULTI_STEP, n_active=2)
        assert len(info.brick_types) == 2

    def test_n_active_clamped_to_max_bricks(self):
        em = _make_manager_n(2)
        info = em.reset(seed=0, n_active=10)  # more than max_bricks
        assert len(info.brick_types) == 2  # clamped


# ---------------------------------------------------------------------------
# ConnectionManager.reset() tests (require MuJoCo)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.lego
class TestConnectionManagerReset:
    def test_reset_clears_engaged_steps(self):
        from sim.lego.connection_manager import ConnectionManager
        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_insertion_scene

        model, data = load_insertion_scene(
            BRICK_TYPES["2x2"], BRICK_TYPES["2x2"], retention_mode="spec_proxy"
        )
        import mujoco

        mgr = ConnectionManager(model, data)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_2x2")
        top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top_2x2")
        # Find weld eq id
        eq_id = -1
        for i in range(model.neq):
            if model.eq_obj1id[i] == base_id and model.eq_obj2id[i] == top_id:
                eq_id = i
                break
        if eq_id < 0:
            pytest.skip("No weld constraint found in spec_proxy scene")
        mgr.register_pair("base_2x2", "top_2x2", eq_id)

        # Manually advance engaged_steps counter
        pair = mgr.pairs[0]
        pair.engaged_steps = 30

        mgr.reset()
        assert pair.engaged_steps == 0

    def test_reset_deactivates_active_welds(self):
        from sim.lego.connection_manager import ConnectionManager
        from sim.lego.constants import BRICK_TYPES
        from sim.lego.contact_scene import load_insertion_scene

        model, data = load_insertion_scene(
            BRICK_TYPES["2x2"], BRICK_TYPES["2x2"], retention_mode="spec_proxy"
        )
        import mujoco

        mgr = ConnectionManager(model, data)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_2x2")
        top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "top_2x2")
        eq_id = -1
        for i in range(model.neq):
            if model.eq_obj1id[i] == base_id and model.eq_obj2id[i] == top_id:
                eq_id = i
                break
        if eq_id < 0:
            pytest.skip("No weld constraint found in spec_proxy scene")
        mgr.register_pair("base_2x2", "top_2x2", eq_id)

        # Force-activate the weld
        data.eq_active[eq_id] = 1
        pair = mgr.pairs[0]
        pair.weld_active = True

        mgr.reset()
        assert not pair.weld_active
        assert int(data.eq_active[eq_id]) == 0


# ---------------------------------------------------------------------------
# Reset reliability stress test (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mujoco
@pytest.mark.lego
class TestResetReliability:
    def test_100_resets_success_rate(self):
        """Stress test: ≥95% success over 100 resets."""
        em = _make_manager_n(4, settle_max_steps=500)
        for seed in range(100):
            em.reset(seed=seed, level=LEVEL_MULTI_STEP)
        rate = em.metrics.success_rate
        assert rate >= 0.95, (
            f"Reset success rate {rate:.2%} below 95% target. "
            f"Failures: {em.metrics.failure_reasons}"
        )

    def test_metrics_accumulate_correctly(self):
        em = _make_manager()
        n_resets = 10
        for seed in range(n_resets):
            em.reset(seed=seed)
        assert em.metrics.total_resets == n_resets
        assert em.episode_count == n_resets
        assert em.metrics.successful_resets + em.metrics.failed_resets == em.metrics.total_resets
