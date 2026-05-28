"""MVP-3 task specification, success detection, and evaluation (Phase 1.2.6).

Provides parametric assembly goal definitions, per-placement and episode-level
success checks, and evaluation metrics for multi-step LEGO assembly tasks.

Usage::

    from sim.lego.task import (
        AssemblyGoal, PlacementTarget, generate_assembly_goal,
        check_placement, evaluate_assembly,
    )

    em = EpisodeManager(brick_slots=["2x2", "2x4"])
    info = em.reset(seed=42, level=LEVEL_SINGLE_CONNECTION)
    goal = generate_assembly_goal(info, bp_type, bp_world_pos, seed=42)

    # After scripted/policy placement...
    success, xy_err = check_placement(em.model, em.data, goal.targets[0])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim.lego.constants import (
    BRICK_HEIGHT,
    BRICK_TYPES,
    STUD_PITCH,
    BaseplateType,
    BrickType,
)
from sim.lego.episode_manager import EpisodeInfo

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementTarget:
    """Where one brick should be placed.

    Attributes:
        slot_index: Which brick slot (0..max_bricks-1) in the episode scene.
        brick_type: Brick type name, e.g., "2x2".
        target_position: World-frame XYZ of brick body origin when correctly placed.
        target_quaternion: MuJoCo quaternion [w, x, y, z] for target orientation.
        base_body_name: Body to place on ("baseplate_8x8" or "brick_0_2x2").
    """

    slot_index: int
    brick_type: str
    target_position: tuple[float, float, float]
    target_quaternion: tuple[float, float, float, float]
    base_body_name: str


@dataclass(frozen=True)
class AssemblyGoal:
    """Ordered sequence of placement targets for one episode.

    Attributes:
        targets: Placement targets ordered by assembly step.
        seed: Seed used to generate this goal.
        level: Curriculum level (1/2/3).
    """

    targets: tuple[PlacementTarget, ...]
    seed: int
    level: int


@dataclass
class PlacementResult:
    """Result of placing one brick.

    Attributes:
        target: The target this result corresponds to.
        success: True if placement meets XY + Z criteria.
        position_error_m: Euclidean XY error from target (meters).
        z_engaged: True if brick Z is within engagement margin of target Z.
        stable: True if brick held for stability duration.
        insertion_steps: Physics steps used for insertion.
        final_position: World-frame XYZ after placement.
    """

    target: PlacementTarget
    success: bool
    position_error_m: float
    z_engaged: bool
    stable: bool
    insertion_steps: int
    final_position: tuple[float, float, float]


@dataclass
class AssemblyResult:
    """Full assembly evaluation.

    Attributes:
        goal: The assembly goal that was attempted.
        placements: Per-brick placement results.
        n_successful: Number of successfully placed bricks.
        n_total: Total number of placement targets.
        all_placed: True if every brick was placed successfully.
        structure_stable: True if final stability hold passed.
        stability_hold_steps: Physics steps used for final stability check.
        total_physics_steps: Total physics steps across all placements + holds.
        max_penetration_m: Max contact penetration observed (meters).
        max_energy_J: Max total energy observed (Joules).
    """

    goal: AssemblyGoal
    placements: list[PlacementResult] = field(default_factory=list)
    n_successful: int = 0
    n_total: int = 0
    all_placed: bool = False
    structure_stable: bool = False
    stability_hold_steps: int = 0
    total_physics_steps: int = 0
    max_penetration_m: float = 0.0
    max_energy_J: float = 0.0


# ---------------------------------------------------------------------------
# Target position computation
# ---------------------------------------------------------------------------


def compute_target_position(
    baseplate_world_pos: tuple[float, float, float],
    baseplate_type: BaseplateType,
    brick_type: BrickType,
    stud_ix: int,
    stud_iy: int,
) -> tuple[float, float, float]:
    """Compute world position for a brick placed at baseplate stud (ix, iy).

    The brick body origin sits at the center of the brick's bottom face.
    When placed on the baseplate, the bottom face rests on top of the
    baseplate studs, so Z = baseplate_world_Z + baseplate_thickness.

    The XY position aligns the brick's center with the midpoint of the
    stud sub-grid it covers on the baseplate.

    Args:
        baseplate_world_pos: World-frame XYZ of the baseplate body origin.
        baseplate_type: BaseplateType for stud grid computation.
        brick_type: BrickType being placed.
        stud_ix: X index of the brick's "anchor" stud (lower-left corner
            of the brick footprint on the baseplate grid).
        stud_iy: Y index of the brick's "anchor" stud.

    Returns:
        World-frame (x, y, z) for the brick body origin.

    Raises:
        ValueError: If the brick footprint extends beyond the baseplate grid.
    """
    # Validate that brick fits on baseplate at the given anchor
    if stud_ix < 0 or stud_ix + brick_type.nx > baseplate_type.nx_studs:
        raise ValueError(
            f"Brick {brick_type.name} at stud_ix={stud_ix} extends beyond "
            f"baseplate X range [0, {baseplate_type.nx_studs})"
        )
    if stud_iy < 0 or stud_iy + brick_type.ny > baseplate_type.ny_studs:
        raise ValueError(
            f"Brick {brick_type.name} at stud_iy={stud_iy} extends beyond "
            f"baseplate Y range [0, {baseplate_type.ny_studs})"
        )

    # Baseplate stud grid origin (first stud at ix=0, iy=0) relative to
    # baseplate body origin (center of bottom face)
    bp_grid_origin_x = -STUD_PITCH * (baseplate_type.nx_studs - 1) / 2.0
    bp_grid_origin_y = -STUD_PITCH * (baseplate_type.ny_studs - 1) / 2.0

    # Center of the brick footprint on the baseplate stud grid (local frame)
    # The brick covers studs [stud_ix, stud_ix + nx) × [stud_iy, stud_iy + ny)
    # Center of that sub-grid:
    center_ix = stud_ix + (brick_type.nx - 1) / 2.0
    center_iy = stud_iy + (brick_type.ny - 1) / 2.0

    local_x = bp_grid_origin_x + center_ix * STUD_PITCH
    local_y = bp_grid_origin_y + center_iy * STUD_PITCH

    # World position: baseplate origin + local offset
    world_x = baseplate_world_pos[0] + local_x
    world_y = baseplate_world_pos[1] + local_y
    # Brick bottom sits on top of baseplate surface (thickness from bp origin bottom)
    world_z = baseplate_world_pos[2] + baseplate_type.thickness

    return (world_x, world_y, world_z)


def compute_brick_on_brick_target(
    model,
    data,
    base_body_name: str,
    base_brick_type: BrickType,
    top_brick_type: BrickType,
    stud_ix: int = 0,
    stud_iy: int = 0,
) -> tuple[float, float, float]:
    """Compute world position for stacking a brick on top of another brick.

    The top brick sits on the base brick's studs. Its body origin Z is at
    base_body_Z + BRICK_HEIGHT (the base brick's top surface).

    XY is aligned so the top brick's center is at the midpoint of the stud
    sub-grid it covers on the base brick.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (must be forward-computed).
        base_body_name: Name of the base brick body (e.g., "brick_0_2x2").
        base_brick_type: BrickType of the base brick.
        top_brick_type: BrickType of the top brick.
        stud_ix: X stud index on the base brick's grid (0-indexed).
        stud_iy: Y stud index on the base brick's grid (0-indexed).

    Returns:
        World-frame (x, y, z) for the top brick body origin.

    Raises:
        ValueError: If the top brick footprint extends beyond the base brick grid.
    """
    if stud_ix < 0 or stud_ix + top_brick_type.nx > base_brick_type.nx:
        raise ValueError(
            f"Top brick {top_brick_type.name} at stud_ix={stud_ix} extends beyond "
            f"base brick {base_brick_type.name} X range [0, {base_brick_type.nx})"
        )
    if stud_iy < 0 or stud_iy + top_brick_type.ny > base_brick_type.ny:
        raise ValueError(
            f"Top brick {top_brick_type.name} at stud_iy={stud_iy} extends beyond "
            f"base brick {base_brick_type.name} Y range [0, {base_brick_type.ny})"
        )

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    base_pos = data.xpos[body_id]

    # Base brick stud grid origin relative to body origin
    grid_origin_x = -STUD_PITCH * (base_brick_type.nx - 1) / 2.0
    grid_origin_y = -STUD_PITCH * (base_brick_type.ny - 1) / 2.0

    center_ix = stud_ix + (top_brick_type.nx - 1) / 2.0
    center_iy = stud_iy + (top_brick_type.ny - 1) / 2.0

    # Note: base brick may be rotated; for now assume identity orientation
    # (the scripted assembler places bricks with identity quaternion)
    local_x = grid_origin_x + center_ix * STUD_PITCH
    local_y = grid_origin_y + center_iy * STUD_PITCH

    world_x = float(base_pos[0]) + local_x
    world_y = float(base_pos[1]) + local_y
    world_z = float(base_pos[2]) + BRICK_HEIGHT

    return (world_x, world_y, world_z)


# ---------------------------------------------------------------------------
# Goal generation
# ---------------------------------------------------------------------------

# Identity quaternion (no rotation)
_IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _brick_footprints_overlap(
    pos_a: tuple[float, float],
    brick_a: BrickType,
    pos_b: tuple[float, float],
    brick_b: BrickType,
) -> bool:
    """Check if two brick footprints overlap on the XY plane (AABB test)."""
    a_hx, a_hy = brick_a.shell_half_x, brick_a.shell_half_y
    b_hx, b_hy = brick_b.shell_half_x, brick_b.shell_half_y

    if abs(pos_a[0] - pos_b[0]) < a_hx + b_hx and abs(pos_a[1] - pos_b[1]) < a_hy + b_hy:
        return True
    return False


def generate_assembly_goal(
    episode_info: EpisodeInfo,
    baseplate_type: BaseplateType,
    baseplate_world_pos: tuple[float, float, float],
    seed: int,
    stacking: bool = False,
    stud_margin: int = 1,
) -> AssemblyGoal:
    """Generate a random valid assembly goal for the episode's active bricks.

    Deterministic from seed. Picks non-overlapping baseplate stud positions
    for each brick. If ``stacking=True`` and there are 2+ bricks, the
    second brick is placed on top of the first (brick-on-brick).

    Args:
        episode_info: EpisodeInfo returned by ``EpisodeManager.reset()``.
        baseplate_type: BaseplateType for target computation.
        baseplate_world_pos: World-frame XYZ of the baseplate body origin.
        seed: Seed for deterministic goal sampling.
        stacking: If True and n_bricks >= 2, place second brick on first.
        stud_margin: Minimum stud distance from baseplate edge for placement.

    Returns:
        AssemblyGoal with ordered placement targets.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n_bricks = len(episode_info.brick_types)
    targets: list[PlacementTarget] = []
    placed: list[tuple[tuple[float, float], BrickType]] = []  # (xy, brick_type) of placed bricks

    bp_body_name = f"baseplate_{baseplate_type.name}"

    for step_idx in range(n_bricks):
        bt_name = episode_info.brick_types[step_idx]
        bt = BRICK_TYPES[bt_name]

        # If stacking and this is the second brick, place on top of first
        if stacking and step_idx == 1 and len(targets) > 0:
            first_target = targets[0]
            first_bt = BRICK_TYPES[first_target.brick_type]
            # Stack at stud (0, 0) of the first brick — top brick must fit
            can_stack = bt.nx <= first_bt.nx and bt.ny <= first_bt.ny
            if can_stack:
                # Use a placeholder position — actual position computed at execution
                # time from base brick's live position. Store the target Z offset.
                first_pos = first_target.target_position
                target_pos = (
                    first_pos[0],
                    first_pos[1],
                    first_pos[2] + BRICK_HEIGHT,
                )
                targets.append(
                    PlacementTarget(
                        slot_index=step_idx,
                        brick_type=bt_name,
                        target_position=target_pos,
                        target_quaternion=_IDENTITY_QUAT,
                        base_body_name=f"brick_{first_target.slot_index}_{first_target.brick_type}",
                    )
                )
                continue

        # Place on baseplate: sample a valid anchor stud position
        max_ix = baseplate_type.nx_studs - bt.nx - stud_margin
        max_iy = baseplate_type.ny_studs - bt.ny - stud_margin
        min_ix = stud_margin
        min_iy = stud_margin

        if max_ix < min_ix or max_iy < min_iy:
            # Brick too large for baseplate with given margin; use edge
            min_ix = max(0, baseplate_type.nx_studs - bt.nx)
            min_iy = max(0, baseplate_type.ny_studs - bt.ny)
            max_ix = baseplate_type.nx_studs - bt.nx
            max_iy = baseplate_type.ny_studs - bt.ny

        # Try to find non-overlapping position
        found = False
        for _ in range(100):
            stud_ix = int(rng.integers(min_ix, max_ix + 1))
            stud_iy = int(rng.integers(min_iy, max_iy + 1))

            target_pos = compute_target_position(
                baseplate_world_pos, baseplate_type, bt, stud_ix, stud_iy
            )

            # Check overlap with already-placed bricks
            overlap = False
            for prev_xy, prev_bt in placed:
                if _brick_footprints_overlap((target_pos[0], target_pos[1]), bt, prev_xy, prev_bt):
                    overlap = True
                    break

            if not overlap:
                found = True
                break

        if not found:
            # Use last sampled position as fallback
            target_pos = compute_target_position(
                baseplate_world_pos, baseplate_type, bt, stud_ix, stud_iy
            )

        targets.append(
            PlacementTarget(
                slot_index=step_idx,
                brick_type=bt_name,
                target_position=target_pos,
                target_quaternion=_IDENTITY_QUAT,
                base_body_name=bp_body_name,
            )
        )
        placed.append(((target_pos[0], target_pos[1]), bt))

    return AssemblyGoal(
        targets=tuple(targets),
        seed=seed,
        level=episode_info.level,
    )


# ---------------------------------------------------------------------------
# Success detection
# ---------------------------------------------------------------------------


def check_placement(
    model,
    data,
    target: PlacementTarget,
    xy_tol_m: float = 0.001,
    z_margin_m: float = 0.0005,
) -> tuple[bool, float]:
    """Check if a brick is correctly placed at its target.

    Success requires:
    - XY distance from target within ``xy_tol_m``
    - Z position within ``z_margin_m`` of target Z

    Args:
        model: MuJoCo model.
        data: MuJoCo data (must be forward-computed).
        target: PlacementTarget to check against.
        xy_tol_m: Maximum XY distance for success (meters).
        z_margin_m: Maximum absolute Z deviation (meters).

    Returns:
        (success, xy_error_m) tuple.
    """
    bt = BRICK_TYPES[target.brick_type]
    body_name = f"brick_{target.slot_index}_{bt.name}"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    actual_pos = data.xpos[body_id]
    target_pos = target.target_position

    dx = float(actual_pos[0]) - target_pos[0]
    dy = float(actual_pos[1]) - target_pos[1]
    dz = float(actual_pos[2]) - target_pos[2]

    xy_error = math.sqrt(dx * dx + dy * dy)
    z_ok = abs(dz) < z_margin_m
    xy_ok = xy_error < xy_tol_m

    return (xy_ok and z_ok, xy_error)


def check_stability(
    model,
    data,
    body_names: list[str],
    hold_duration_s: float = 2.0,
    velocity_threshold: float = 0.001,
    check_interval: int = 50,
) -> tuple[bool, int]:
    """Step physics and verify all bodies remain stable.

    Runs physics for ``hold_duration_s``, checking every ``check_interval``
    steps that all body velocities are below threshold.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_names: Names of bodies to monitor.
        hold_duration_s: Hold duration in seconds.
        velocity_threshold: Max linear velocity (m/s) for "stable".
        check_interval: Steps between checks.

    Returns:
        (stable, steps_taken) — stable=True if all bodies stayed below
        threshold for the full duration.
    """
    dt = model.opt.timestep
    total_steps = int(hold_duration_s / dt)

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]

    for step in range(total_steps):
        mujoco.mj_step(model, data)

        if (step + 1) % check_interval != 0:
            continue

        for body_id in body_ids:
            lin_speed = float(np.linalg.norm(data.cvel[body_id, 3:]))
            if lin_speed > velocity_threshold:
                return False, step + 1

    return True, total_steps


# ---------------------------------------------------------------------------
# Assembly evaluation
# ---------------------------------------------------------------------------


def evaluate_assembly(
    model,
    data,
    goal: AssemblyGoal,
    placements: list[PlacementResult],
    hold_duration_s: float = 2.0,
    velocity_threshold: float = 0.001,
) -> AssemblyResult:
    """Evaluate the full assembly after all placements.

    Runs a final stability hold and computes aggregate metrics.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (post-assembly state).
        goal: The assembly goal that was attempted.
        placements: Per-brick PlacementResults.
        hold_duration_s: Duration of final stability hold (seconds).
        velocity_threshold: Velocity threshold for stability (m/s).

    Returns:
        AssemblyResult with aggregate metrics.
    """
    n_total = len(goal.targets)
    n_successful = sum(1 for p in placements if p.success)
    all_placed = n_successful == n_total
    total_steps = sum(p.insertion_steps for p in placements)

    # Measure max penetration and energy from current state
    max_pen = 0.0
    for k in range(data.ncon):
        dist = float(data.contact[k].dist)
        if dist < 0.0:
            max_pen = max(max_pen, -dist)

    max_energy = float(abs(data.energy[0] + data.energy[1])) if hasattr(data, "energy") else 0.0

    # Run final stability hold on all placed bricks
    placed_body_names = []
    for p in placements:
        if p.success:
            bt = BRICK_TYPES[p.target.brick_type]
            placed_body_names.append(f"brick_{p.target.slot_index}_{bt.name}")

    structure_stable = False
    hold_steps = 0
    if placed_body_names:
        structure_stable, hold_steps = check_stability(
            model, data, placed_body_names, hold_duration_s, velocity_threshold
        )
        total_steps += hold_steps

    return AssemblyResult(
        goal=goal,
        placements=placements,
        n_successful=n_successful,
        n_total=n_total,
        all_placed=all_placed,
        structure_stable=structure_stable,
        stability_hold_steps=hold_steps,
        total_physics_steps=total_steps,
        max_penetration_m=max_pen,
        max_energy_J=max_energy,
    )
