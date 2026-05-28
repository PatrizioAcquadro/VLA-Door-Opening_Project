"""Tests for Phase 1.2.6: MVP-3 Task (multi-step assembly).

Tests task specification, success detection, scripted assembly execution,
and evaluation metrics for multi-step LEGO assembly.
"""

from __future__ import annotations

import math

import pytest

from sim.lego.constants import (
    BASEPLATE_THICKNESS,
    BASEPLATE_TYPES,
    BRICK_HEIGHT,
    BRICK_TYPES,
    STUD_PITCH,
)
from sim.lego.task import (
    AssemblyGoal,
    AssemblyResult,
    PlacementResult,
    PlacementTarget,
    _brick_footprints_overlap,
    compute_target_position,
    generate_assembly_goal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default baseplate + workspace positions matching EpisodeManager defaults
_BP_TYPE = BASEPLATE_TYPES["8x8"]
_TABLE_POS = (0.45, 0.0, 0.75)
_TABLE_HALF_Z = 0.02
_BP_WORLD_POS = (_TABLE_POS[0], _TABLE_POS[1], _TABLE_POS[2] + _TABLE_HALF_Z)

_MJ = pytest.importorskip("mujoco", reason="mujoco not available")

lego = pytest.mark.lego
mujoco_mark = pytest.mark.mujoco
slow = pytest.mark.slow


def _make_episode_manager(**kwargs):
    from sim.lego.episode_manager import EpisodeManager

    return EpisodeManager(
        **{"brick_slots": ["2x2"], "settle_max_steps": 300, "settle_check_interval": 25, **kwargs}
    )


def _make_episode_info(brick_types=None, level=2, seed=42):
    from sim.lego.episode_manager import EpisodeInfo, SpawnPose

    if brick_types is None:
        brick_types = ["2x2"]
    return EpisodeInfo(
        seed=seed,
        level=level,
        brick_types=brick_types,
        spawn_poses=[SpawnPose(position=(0.45, 0.0, 0.85), quaternion=(1, 0, 0, 0))]
        * len(brick_types),
        settle_steps=100,
        settle_success=True,
    )


# ---------------------------------------------------------------------------
# PlacementTarget tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestPlacementTarget:
    def test_frozen_dataclass(self):
        import dataclasses

        t = PlacementTarget(
            slot_index=0,
            brick_type="2x2",
            target_position=(0.45, 0.0, 0.79),
            target_quaternion=(1, 0, 0, 0),
            base_body_name="baseplate_8x8",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.slot_index = 1  # type: ignore[misc]

    def test_fields(self):
        t = PlacementTarget(
            slot_index=2,
            brick_type="2x4",
            target_position=(0.5, 0.1, 0.8),
            target_quaternion=(1, 0, 0, 0),
            base_body_name="baseplate_8x8",
        )
        assert t.slot_index == 2
        assert t.brick_type == "2x4"
        assert t.base_body_name == "baseplate_8x8"


# ---------------------------------------------------------------------------
# AssemblyGoal tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestAssemblyGoal:
    def test_frozen_dataclass(self):
        import dataclasses

        goal = AssemblyGoal(targets=(), seed=42, level=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            goal.seed = 0  # type: ignore[misc]

    def test_targets_ordering(self):
        t1 = PlacementTarget(0, "2x2", (0, 0, 0), (1, 0, 0, 0), "baseplate_8x8")
        t2 = PlacementTarget(1, "2x4", (0.1, 0, 0), (1, 0, 0, 0), "baseplate_8x8")
        goal = AssemblyGoal(targets=(t1, t2), seed=0, level=3)
        assert goal.targets[0].slot_index == 0
        assert goal.targets[1].slot_index == 1
        assert len(goal.targets) == 2


# ---------------------------------------------------------------------------
# PlacementResult / AssemblyResult tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestPlacementResult:
    def test_mutable_dataclass(self):
        t = PlacementTarget(0, "2x2", (0, 0, 0), (1, 0, 0, 0), "baseplate_8x8")
        r = PlacementResult(
            target=t,
            success=True,
            position_error_m=0.0005,
            z_engaged=True,
            stable=True,
            insertion_steps=500,
            final_position=(0.45, 0.0, 0.79),
        )
        assert r.success is True
        r.success = False
        assert r.success is False


class TestAssemblyResult:
    def test_default_fields(self):
        goal = AssemblyGoal(targets=(), seed=0, level=1)
        r = AssemblyResult(goal=goal)
        assert r.n_successful == 0
        assert r.all_placed is False
        assert r.placements == []


# ---------------------------------------------------------------------------
# Target position computation tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestTargetPositionCompute:
    def test_center_stud(self):
        """Brick 2x2 at center of 8x8 baseplate (stud 3,3)."""
        pos = compute_target_position(_BP_WORLD_POS, _BP_TYPE, BRICK_TYPES["2x2"], 3, 3)
        # Baseplate stud (3,3) is at local offset from center
        # Grid origin: -STUD_PITCH * 3.5 = -0.028
        # Stud (3,3): -0.028 + 3*0.008 = -0.004
        # Brick 2x2 center covers studs (3,3) and (4,4), center at (3.5, 3.5)
        # local_x = -0.028 + 3.5*0.008 = 0.0
        # local_y = -0.028 + 3.5*0.008 = 0.0
        expected_x = _BP_WORLD_POS[0] + 0.0
        expected_y = _BP_WORLD_POS[1] + 0.0
        expected_z = _BP_WORLD_POS[2] + BASEPLATE_THICKNESS
        assert math.isclose(pos[0], expected_x, abs_tol=1e-6)
        assert math.isclose(pos[1], expected_y, abs_tol=1e-6)
        assert math.isclose(pos[2], expected_z, abs_tol=1e-6)

    def test_corner_stud(self):
        """Brick 2x2 at corner (stud 0,0) of 8x8 baseplate."""
        pos = compute_target_position(_BP_WORLD_POS, _BP_TYPE, BRICK_TYPES["2x2"], 0, 0)
        # Center of brick footprint at (0.5, 0.5) in stud grid
        grid_origin = -STUD_PITCH * 3.5
        expected_local_x = grid_origin + 0.5 * STUD_PITCH
        expected_local_y = grid_origin + 0.5 * STUD_PITCH
        assert math.isclose(pos[0], _BP_WORLD_POS[0] + expected_local_x, abs_tol=1e-6)
        assert math.isclose(pos[1], _BP_WORLD_POS[1] + expected_local_y, abs_tol=1e-6)

    def test_2x4_brick(self):
        """Brick 2x4 placed at stud (1,2) of 8x8 baseplate."""
        bt = BRICK_TYPES["2x4"]
        pos = compute_target_position(_BP_WORLD_POS, _BP_TYPE, bt, 1, 2)
        # Center of 2x4 footprint: (1 + 0.5, 2 + 1.5) = (1.5, 3.5)
        grid_origin_x = -STUD_PITCH * 3.5
        grid_origin_y = -STUD_PITCH * 3.5
        expected_local_x = grid_origin_x + 1.5 * STUD_PITCH
        expected_local_y = grid_origin_y + 3.5 * STUD_PITCH
        assert math.isclose(pos[0], _BP_WORLD_POS[0] + expected_local_x, abs_tol=1e-6)
        assert math.isclose(pos[1], _BP_WORLD_POS[1] + expected_local_y, abs_tol=1e-6)

    def test_z_height(self):
        """Z should be baseplate top + thickness."""
        pos = compute_target_position(_BP_WORLD_POS, _BP_TYPE, BRICK_TYPES["2x2"], 1, 1)
        assert math.isclose(pos[2], _BP_WORLD_POS[2] + BASEPLATE_THICKNESS, abs_tol=1e-6)

    def test_out_of_bounds_x(self):
        """Brick extending beyond baseplate X should raise ValueError."""
        with pytest.raises(ValueError, match="extends beyond"):
            compute_target_position(_BP_WORLD_POS, _BP_TYPE, BRICK_TYPES["2x2"], 7, 0)

    def test_out_of_bounds_y(self):
        """Brick extending beyond baseplate Y should raise ValueError."""
        with pytest.raises(ValueError, match="extends beyond"):
            compute_target_position(_BP_WORLD_POS, _BP_TYPE, BRICK_TYPES["2x6"], 0, 5)

    def test_negative_stud_index(self):
        with pytest.raises(ValueError, match="extends beyond"):
            compute_target_position(_BP_WORLD_POS, _BP_TYPE, BRICK_TYPES["2x2"], -1, 0)


# ---------------------------------------------------------------------------
# Footprint overlap tests (no MuJoCo)
# ---------------------------------------------------------------------------


class TestFootprintOverlap:
    def test_overlapping(self):
        bt = BRICK_TYPES["2x2"]
        assert _brick_footprints_overlap((0.0, 0.0), bt, (0.005, 0.005), bt) is True

    def test_non_overlapping(self):
        bt = BRICK_TYPES["2x2"]
        assert _brick_footprints_overlap((0.0, 0.0), bt, (0.1, 0.0), bt) is False

    def test_edge_touching(self):
        bt = BRICK_TYPES["2x2"]
        separation = bt.shell_half_x * 2 + 0.001
        assert _brick_footprints_overlap((0.0, 0.0), bt, (separation, 0.0), bt) is False


# ---------------------------------------------------------------------------
# Goal generation tests (needs MuJoCo for EpisodeInfo)
# ---------------------------------------------------------------------------


@lego
class TestGoalGeneration:
    def test_single_brick_goal(self):
        info = _make_episode_info(brick_types=["2x2"], level=2)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)
        assert len(goal.targets) == 1
        assert goal.targets[0].slot_index == 0
        assert goal.targets[0].brick_type == "2x2"
        assert goal.targets[0].base_body_name == "baseplate_8x8"

    def test_multi_brick_no_overlap(self):
        info = _make_episode_info(brick_types=["2x2", "2x4", "2x2"], level=3)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)
        assert len(goal.targets) == 3
        # Check no overlapping footprints
        for i in range(len(goal.targets)):
            for j in range(i + 1, len(goal.targets)):
                ti, tj = goal.targets[i], goal.targets[j]
                # Skip brick-on-brick targets
                if not ti.base_body_name.startswith("baseplate_"):
                    continue
                if not tj.base_body_name.startswith("baseplate_"):
                    continue
                bi = BRICK_TYPES[ti.brick_type]
                bj = BRICK_TYPES[tj.brick_type]
                overlap = _brick_footprints_overlap(
                    (ti.target_position[0], ti.target_position[1]),
                    bi,
                    (tj.target_position[0], tj.target_position[1]),
                    bj,
                )
                assert not overlap, f"Targets {i} and {j} overlap"

    def test_deterministic(self):
        info = _make_episode_info(brick_types=["2x2", "2x4"], level=3)
        goal1 = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=99)
        goal2 = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=99)
        assert goal1.targets == goal2.targets

    def test_different_seeds_differ(self):
        info = _make_episode_info(brick_types=["2x2", "2x4", "2x6"], level=3)
        goal1 = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=1)
        goal2 = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=2)
        # Very unlikely to be identical with different seeds
        positions_differ = any(
            goal1.targets[i].target_position != goal2.targets[i].target_position
            for i in range(len(goal1.targets))
        )
        assert positions_differ

    def test_stacking_goal(self):
        info = _make_episode_info(brick_types=["2x2", "2x2"], level=3)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42, stacking=True)
        assert len(goal.targets) == 2
        # First target on baseplate
        assert goal.targets[0].base_body_name == "baseplate_8x8"
        # Second target on first brick
        assert goal.targets[1].base_body_name.startswith("brick_")
        # Stacked Z should be one brick height above baseplate placement
        z_diff = goal.targets[1].target_position[2] - goal.targets[0].target_position[2]
        assert math.isclose(z_diff, BRICK_HEIGHT, abs_tol=1e-6)

    def test_goal_level_preserved(self):
        info = _make_episode_info(level=3)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=0)
        assert goal.level == 3
        assert goal.seed == 0


# ---------------------------------------------------------------------------
# Placement check tests (MuJoCo)
# ---------------------------------------------------------------------------


@lego
@mujoco_mark
class TestPlacementCheck:
    def test_correct_placement(self):
        """Brick at target position should pass check."""
        from sim.lego.task import check_placement

        em = _make_episode_manager()
        info = em.reset(seed=42)

        # Generate a goal and teleport brick to target
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)
        target = goal.targets[0]

        bt = BRICK_TYPES[target.brick_type]
        joint_name = f"brick_{target.slot_index}_{bt.name}_joint"
        jnt_id = _MJ.mj_name2id(em.model, _MJ.mjtObj.mjOBJ_JOINT, joint_name)
        addr = int(em.model.jnt_qposadr[jnt_id])

        # Place brick exactly at target
        em.data.qpos[addr : addr + 3] = target.target_position
        em.data.qpos[addr + 3 : addr + 7] = target.target_quaternion
        _MJ.mj_forward(em.model, em.data)

        success, xy_err = check_placement(em.model, em.data, target)
        assert success is True
        assert xy_err < 0.001

    def test_misaligned_placement(self):
        """Brick offset from target should fail check."""
        from sim.lego.task import check_placement

        em = _make_episode_manager()
        info = em.reset(seed=42)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)
        target = goal.targets[0]

        bt = BRICK_TYPES[target.brick_type]
        joint_name = f"brick_{target.slot_index}_{bt.name}_joint"
        jnt_id = _MJ.mj_name2id(em.model, _MJ.mjtObj.mjOBJ_JOINT, joint_name)
        addr = int(em.model.jnt_qposadr[jnt_id])

        # Place 5mm off in X
        offset_pos = (target.target_position[0] + 0.005, *target.target_position[1:])
        em.data.qpos[addr : addr + 3] = offset_pos
        em.data.qpos[addr + 3 : addr + 7] = target.target_quaternion
        _MJ.mj_forward(em.model, em.data)

        success, xy_err = check_placement(em.model, em.data, target)
        assert success is False
        assert xy_err > 0.004


# ---------------------------------------------------------------------------
# Scripted assembly tests (MuJoCo)
# ---------------------------------------------------------------------------


@lego
@mujoco_mark
class TestScriptedPlacement:
    def test_single_brick_placement(self):
        """Scripted assembler should successfully place one brick on baseplate."""
        from sim.lego.scripted_assembly import ScriptedAssembler

        em = _make_episode_manager()
        info = em.reset(seed=42, level=2)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)

        assembler = ScriptedAssembler(em.model, em.data)
        result = assembler.execute_placement(goal.targets[0])

        assert result.z_engaged is True, f"Insertion failed, final_pos={result.final_position}"
        assert result.position_error_m < 0.002, f"XY error too large: {result.position_error_m}"

    def test_single_brick_assembly_result(self):
        """Full assembly with one brick should produce valid AssemblyResult."""
        from sim.lego.scripted_assembly import ScriptedAssembler

        em = _make_episode_manager()
        info = em.reset(seed=42, level=2)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)

        assembler = ScriptedAssembler(em.model, em.data)
        result = assembler.execute_assembly(goal, hold_duration_s=0.5)

        assert result.n_total == 1
        assert result.n_successful >= 0
        assert result.total_physics_steps > 0
        assert len(result.placements) == 1


@lego
@mujoco_mark
@slow
class TestScriptedAssembly:
    def test_multi_brick_baseplate(self):
        """Place 2 bricks side-by-side on baseplate."""
        from sim.lego.scripted_assembly import ScriptedAssembler

        em = _make_episode_manager(brick_slots=["2x2", "2x2"])
        info = em.reset(seed=42, level=3, n_active=2)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)

        assembler = ScriptedAssembler(em.model, em.data)
        result = assembler.execute_assembly(goal, hold_duration_s=0.5)

        assert result.n_total == 2
        assert len(result.placements) == 2
        # At least first brick should engage
        assert result.placements[0].z_engaged is True

    def test_three_brick_assembly(self):
        """Place 3 bricks on baseplate."""
        from sim.lego.scripted_assembly import ScriptedAssembler

        em = _make_episode_manager(brick_slots=["2x2", "2x4", "2x2"])
        info = em.reset(seed=10, level=3, n_active=3)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=10)

        assembler = ScriptedAssembler(em.model, em.data)
        result = assembler.execute_assembly(goal, hold_duration_s=0.5)

        assert result.n_total == 3
        assert len(result.placements) == 3


@lego
@mujoco_mark
@slow
class TestScriptedStacking:
    def test_brick_on_brick(self):
        """Stack a 2x2 on top of another 2x2 on the baseplate."""
        from sim.lego.scripted_assembly import ScriptedAssembler

        em = _make_episode_manager(brick_slots=["2x2", "2x2"])
        info = em.reset(seed=42, level=3, n_active=2)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42, stacking=True)

        # Verify goal structure
        assert goal.targets[0].base_body_name == "baseplate_8x8"
        assert goal.targets[1].base_body_name.startswith("brick_")

        assembler = ScriptedAssembler(em.model, em.data)
        result = assembler.execute_assembly(goal, hold_duration_s=0.5)

        assert result.n_total == 2
        # First brick should engage on baseplate
        assert result.placements[0].z_engaged is True


# ---------------------------------------------------------------------------
# Failure detection tests (MuJoCo)
# ---------------------------------------------------------------------------


@lego
@mujoco_mark
class TestFailureDetection:
    def test_misaligned_insertion_fails(self):
        """Brick placed far from any stud position should not pass check."""
        from sim.lego.task import check_placement

        em = _make_episode_manager()
        info = em.reset(seed=42)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)
        target = goal.targets[0]

        # Create a bogus target 10cm off
        bad_target = PlacementTarget(
            slot_index=target.slot_index,
            brick_type=target.brick_type,
            target_position=(
                target.target_position[0] + 0.1,
                target.target_position[1],
                target.target_position[2],
            ),
            target_quaternion=target.target_quaternion,
            base_body_name=target.base_body_name,
        )
        success, xy_err = check_placement(em.model, em.data, bad_target)
        assert success is False
        assert xy_err > 0.01  # brick is not at the bogus target


# ---------------------------------------------------------------------------
# Evaluation metrics tests (MuJoCo)
# ---------------------------------------------------------------------------


@lego
@mujoco_mark
class TestAssemblyMetrics:
    def test_evaluate_assembly_fields(self):
        """evaluate_assembly should produce complete metrics."""
        from sim.lego.task import evaluate_assembly

        em = _make_episode_manager()
        info = em.reset(seed=42)
        goal = generate_assembly_goal(info, _BP_TYPE, _BP_WORLD_POS, seed=42)

        # Fake a successful placement result
        t = goal.targets[0]
        placement = PlacementResult(
            target=t,
            success=True,
            position_error_m=0.0003,
            z_engaged=True,
            stable=True,
            insertion_steps=500,
            final_position=t.target_position,
        )

        result = evaluate_assembly(em.model, em.data, goal, [placement], hold_duration_s=0.2)
        assert result.n_successful == 1
        assert result.n_total == 1
        assert result.all_placed is True
        assert result.total_physics_steps > 0


# ---------------------------------------------------------------------------
# Deterministic replay tests (MuJoCo)
# ---------------------------------------------------------------------------


@lego
@mujoco_mark
class TestDeterministicReplay:
    def test_same_seed_same_goal(self):
        """Same seed should produce identical goals."""
        em = _make_episode_manager(brick_slots=["2x2", "2x4"])
        info1 = em.reset(seed=42, level=3, n_active=2)
        goal1 = generate_assembly_goal(info1, _BP_TYPE, _BP_WORLD_POS, seed=42)

        info2 = em.reset(seed=42, level=3, n_active=2)
        goal2 = generate_assembly_goal(info2, _BP_TYPE, _BP_WORLD_POS, seed=42)

        for t1, t2 in zip(goal1.targets, goal2.targets, strict=True):
            assert t1.target_position == t2.target_position
            assert t1.brick_type == t2.brick_type

    def test_same_seed_same_assembly(self):
        """Same seed should produce same assembly results."""
        from sim.lego.scripted_assembly import ScriptedAssembler

        em = _make_episode_manager()
        info1 = em.reset(seed=42, level=2)
        goal1 = generate_assembly_goal(info1, _BP_TYPE, _BP_WORLD_POS, seed=42)
        a1 = ScriptedAssembler(em.model, em.data)
        r1 = a1.execute_assembly(goal1, hold_duration_s=0.2)

        info2 = em.reset(seed=42, level=2)
        goal2 = generate_assembly_goal(info2, _BP_TYPE, _BP_WORLD_POS, seed=42)
        a2 = ScriptedAssembler(em.model, em.data)
        r2 = a2.execute_assembly(goal2, hold_duration_s=0.2)

        assert r1.n_successful == r2.n_successful
        for p1, p2 in zip(r1.placements, r2.placements, strict=True):
            assert p1.z_engaged == p2.z_engaged
            assert math.isclose(p1.position_error_m, p2.position_error_m, abs_tol=1e-6)
