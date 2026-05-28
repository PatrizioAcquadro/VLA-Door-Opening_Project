"""Episode manager for deterministic, reproducible LEGO assembly episodes (Phase 1.2.5).

Implements a high-reliability episode manager using the template model approach:
compile MJCF once with all brick slots, then use ``mj_resetData()`` + qpos
manipulation for fast per-episode resets.

Features:
- Deterministic spawn sampling from integer seed (np.random.PCG64)
- Constraint-based spawn placement (min-distance, region bounds, random yaw)
- Settle phase: step physics until brick velocities converge
- Three curriculum levels (single brick → single connection → multi-step)
- Cumulative reset reliability metrics

Usage::

    from sim.lego.episode_manager import EpisodeManager, LEVEL_SINGLE_BRICK

    em = EpisodeManager()               # default: 4 slots, 2x2/2x4/2x6 bricks
    info = em.reset(seed=42)            # Level 1: 1 brick spawned
    info = em.reset(seed=0, level=3)    # Level 3: 2-4 bricks

    print(em.metrics.success_rate)      # fraction of successful resets
    print(info.settle_steps)            # physics steps to stabilize
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.lego.constants import BASEPLATE_TYPES, BRICK_TYPES, BaseplateType, BrickType
from sim.lego.contact_scene import generate_episode_scene

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Curriculum level constants
# ---------------------------------------------------------------------------

LEVEL_SINGLE_BRICK: int = 1
"""Level 1: 1 brick spawned on table surface (pick-and-place task)."""

LEVEL_SINGLE_CONNECTION: int = 2
"""Level 2: 1 brick spawned, goal is to connect it to the baseplate."""

LEVEL_MULTI_STEP: int = 3
"""Level 3: 2-4 bricks spawned, multi-step assembly task."""

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PARK_POS: tuple[float, float, float] = (0.0, 0.0, -10.0)
_IDENTITY_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Failure reason keys for ResetMetrics.failure_reasons
_FAIL_SPAWN_OVERLAP = "spawn_overlap"
_FAIL_SETTLE_TIMEOUT = "settle_timeout"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpawnPose:
    """Position and orientation for one brick at spawn time.

    Args:
        position: World-frame XYZ center of brick body origin.
        quaternion: MuJoCo quaternion [w, x, y, z].
    """

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class EpisodeInfo:
    """Information returned after ``EpisodeManager.reset()``.

    Args:
        seed: Episode seed used for spawn sampling.
        level: Curriculum level (1/2/3).
        brick_types: Ordered list of brick type names for active slots.
        spawn_poses: Spawn pose per active brick (same order as brick_types).
        settle_steps: Number of physics steps taken during settle phase.
        settle_success: True if bricks stabilized within settle budget.
    """

    seed: int
    level: int
    brick_types: list[str]
    spawn_poses: list[SpawnPose]
    settle_steps: int
    settle_success: bool


@dataclass
class ResetMetrics:
    """Cumulative reset reliability statistics.

    Tracks success rate, average settle time, and failure breakdown
    across all ``EpisodeManager.reset()`` calls.
    """

    total_resets: int = 0
    successful_resets: int = 0
    failed_resets: int = 0
    settle_steps_history: list[int] = field(default_factory=list)
    failure_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of resets that completed successfully (0.0 if no resets)."""
        if self.total_resets == 0:
            return 0.0
        return self.successful_resets / self.total_resets

    @property
    def avg_settle_steps(self) -> float:
        """Average physics steps for settle phase (successful resets only)."""
        if not self.settle_steps_history:
            return 0.0
        return float(np.mean(self.settle_steps_history))

    def _record_success(self, settle_steps: int) -> None:
        self.total_resets += 1
        self.successful_resets += 1
        self.settle_steps_history.append(settle_steps)

    def _record_failure(self, reason: str) -> None:
        self.total_resets += 1
        self.failed_resets += 1
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """Convert yaw (rotation around world Z-axis) to MuJoCo quaternion [w, x, y, z]."""
    half = yaw_rad / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# ---------------------------------------------------------------------------
# EpisodeManager
# ---------------------------------------------------------------------------


class EpisodeManager:
    """High-reliability episode manager for LEGO assembly tasks.

    Compiles a template MJCF scene once at init with all brick slots declared.
    At each ``reset()`` call, uses ``mj_resetData()`` + qpos writes for fast
    re-initialization without XML recompilation.

    Args:
        baseplate: Baseplate type for the workspace.
        brick_slots: Brick type name per slot; length sets max_bricks.
            Cycles through ``brick_set`` if None.
        max_bricks: Number of slots to pre-declare (used only when
            ``brick_slots`` is None).
        brick_set: Available brick type names (used when brick_slots is None).
        retention_mode: ``"physics"`` or ``"spec_proxy"`` (hybrid weld).
        table_pos: XYZ center of table surface in world frame.
        table_size: Half-extents of table box.
        baseplate_offset: XY offset of baseplate center on table.
        spawn_x_range: X bounds for brick spawning (world frame, meters).
        spawn_y_range: Y bounds for brick spawning (world frame, meters).
        spawn_z_above: Height above baseplate surface to spawn bricks (meters).
        settle_max_steps: Max physics steps for settle phase (default 500 = 1.0 s).
        settle_velocity_threshold: Max body speed (m/s) for "settled" state.
        settle_max_penetration: Max contact penetration (m) accepted during settle.
        settle_check_interval: Steps between stability checks during settle.
        min_spawn_distance: Min XY distance between brick centers (meters).
        max_spawn_attempts: Max sampling attempts per brick before failing.
        random_yaw: Randomize brick yaw (Z-rotation) at spawn.
    """

    def __init__(
        self,
        baseplate: BaseplateType = BASEPLATE_TYPES["8x8"],
        brick_slots: list[str] | None = None,
        max_bricks: int = 4,
        brick_set: tuple[str, ...] = ("2x2", "2x4", "2x6"),
        retention_mode: str = "physics",
        table_pos: tuple[float, float, float] = (0.45, 0.0, 0.75),
        table_size: tuple[float, float, float] = (0.25, 0.30, 0.02),
        baseplate_offset: tuple[float, float] = (0.0, 0.0),
        spawn_x_range: tuple[float, float] = (0.25, 0.65),
        spawn_y_range: tuple[float, float] = (-0.20, 0.20),
        spawn_z_above: float = 0.05,
        settle_max_steps: int = 500,
        settle_velocity_threshold: float = 0.001,
        settle_max_penetration: float = 0.005,
        settle_check_interval: int = 50,
        min_spawn_distance: float = 0.03,
        max_spawn_attempts: int = 50,
        random_yaw: bool = True,
    ) -> None:
        # Resolve brick slots
        if brick_slots is None:
            brick_slots = [brick_set[i % len(brick_set)] for i in range(max_bricks)]
        if not brick_slots:
            raise ValueError("brick_slots must not be empty")
        for name in brick_slots:
            if name not in BRICK_TYPES:
                raise ValueError(f"Unknown brick type '{name}'. Available: {list(BRICK_TYPES)}")

        self._baseplate = baseplate
        self._brick_slots = list(brick_slots)
        self._retention_mode = retention_mode

        # Spawn region
        self._spawn_x_range = spawn_x_range
        self._spawn_y_range = spawn_y_range

        # Compute spawn Z: table_top + baseplate_thickness + above_offset
        bp_z = table_pos[2] + table_size[2]
        self._baseplate_surface_z: float = bp_z + baseplate.thickness
        self._spawn_z: float = self._baseplate_surface_z + spawn_z_above

        # Settle config
        self._settle_max_steps = settle_max_steps
        self._settle_velocity_threshold = settle_velocity_threshold
        self._settle_max_penetration = settle_max_penetration
        self._settle_check_interval = settle_check_interval

        # Spawn config
        self._min_spawn_distance = min_spawn_distance
        self._max_spawn_attempts = max_spawn_attempts
        self._random_yaw = random_yaw

        self._template_path: Path | None = None
        self._model = None
        self._data = None
        self._slot_qpos_addrs: list[int] = []
        self._slot_vel_addrs: list[int] = []
        self._slot_body_ids: list[int] = []
        self._slot_geom_ids: list[list[int]] = []
        self._slot_geom_ctypes: list[list[int]] = []
        self._slot_geom_caffinities: list[list[int]] = []
        self._metrics = ResetMetrics()
        self._episode_count: int = 0

        # MuJoCo is optional for spawn-sampling unit tests. Runtime simulation
        # state is only initialized when the package is available.
        if mujoco is None:  # pragma: no cover
            return

        # Compile template scene once.
        # Write to file because <include file="...alex.xml"> requires path resolution.
        from sim.asset_loader import SCENES_DIR

        brick_type_objects: list[BrickType] = [BRICK_TYPES[t] for t in self._brick_slots]
        xml = generate_episode_scene(
            baseplate=baseplate,
            brick_types=brick_type_objects,
            table_pos=table_pos,
            table_size=table_size,
            baseplate_offset=baseplate_offset,
            retention_mode=retention_mode,
        )
        self._template_path = SCENES_DIR / "_episode_template.xml"
        self._template_path.write_text(xml)
        self._model = mujoco.MjModel.from_xml_path(str(self._template_path))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        # Resolve freejoint qpos and velocity addresses + body IDs for each slot
        for i, bt_name in enumerate(self._brick_slots):
            bt = BRICK_TYPES[bt_name]
            joint_name = f"brick_{i}_{bt.name}_joint"
            body_name = f"brick_{i}_{bt.name}"

            jnt_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if jnt_id < 0:
                raise RuntimeError(f"Joint '{joint_name}' not found in template scene")
            if body_id < 0:
                raise RuntimeError(f"Body '{body_name}' not found in template scene")

            self._slot_qpos_addrs.append(int(self._model.jnt_qposadr[jnt_id]))
            self._slot_vel_addrs.append(int(self._model.jnt_dofadr[jnt_id]))
            self._slot_body_ids.append(body_id)

            # Collect all geom IDs for this body slot
            geom_ids = [
                g for g in range(self._model.ngeom) if self._model.geom_bodyid[g] == body_id
            ]
            self._slot_geom_ids.append(geom_ids)
            self._slot_geom_ctypes.append([int(self._model.geom_contype[g]) for g in geom_ids])
            self._slot_geom_caffinities.append(
                [int(self._model.geom_conaffinity[g]) for g in geom_ids]
            )

        # Start with all slots' contacts disabled (all parked, no physics interaction)
        for i in range(self.max_bricks):
            self._disable_slot_contacts(i)

    def _require_simulation(self) -> None:
        """Ensure MuJoCo-backed simulation state is available."""
        if mujoco is None or self._model is None or self._data is None:
            raise RuntimeError("mujoco package is required for EpisodeManager")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def model(self) -> mujoco.MjModel:  # type: ignore[name-defined]
        """Compiled MuJoCo template model."""
        self._require_simulation()
        assert self._model is not None
        return self._model

    @property
    def data(self) -> mujoco.MjData:  # type: ignore[name-defined]
        """MuJoCo simulation data (current episode state)."""
        self._require_simulation()
        assert self._data is not None
        return self._data

    @property
    def metrics(self) -> ResetMetrics:
        """Cumulative reset reliability metrics across all episodes."""
        return self._metrics

    @property
    def episode_count(self) -> int:
        """Total number of ``reset()`` calls made."""
        return self._episode_count

    @property
    def max_bricks(self) -> int:
        """Number of pre-declared brick slots in the template scene."""
        return len(self._brick_slots)

    @property
    def baseplate_surface_z(self) -> float:
        """World-frame Z height of the top surface of the baseplate (meters)."""
        return self._baseplate_surface_z

    @property
    def brick_slots(self) -> list[str]:
        """Brick type name for each slot (index = slot index)."""
        return list(self._brick_slots)

    # -----------------------------------------------------------------------
    # Main reset entry point
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: int,
        level: int = LEVEL_SINGLE_BRICK,
        n_active: int | None = None,
    ) -> EpisodeInfo:
        """Reset episode deterministically from seed.

        Resets physics state, samples collision-free spawn poses, applies
        them to ``data.qpos``, and runs a settle phase. Returns metadata
        about the episode regardless of settle success.

        Args:
            seed: Integer seed for deterministic spawn sampling.
            level: Curriculum level (1=single_brick, 2=single_connection,
                3=multi_step). Controls how many bricks to activate.
            n_active: Override number of active brick slots. If None,
                determined by ``level``.

        Returns:
            EpisodeInfo with seed, level, brick types, spawn poses, and
            settle metrics. ``settle_success=False`` on failure.
        """
        self._require_simulation()
        self._episode_count += 1
        rng = np.random.Generator(np.random.PCG64(seed))

        # Determine active slot count
        if n_active is None:
            n_active = self._n_active_for_level(level, rng)
        n_active = max(1, min(n_active, self.max_bricks))

        active_brick_types = self._brick_slots[:n_active]

        # Sample collision-free spawn positions
        spawn_poses = self._sample_spawn_poses(rng, n_active)
        if spawn_poses is None:
            self._metrics._record_failure(_FAIL_SPAWN_OVERLAP)
            return EpisodeInfo(
                seed=seed,
                level=level,
                brick_types=active_brick_types,
                spawn_poses=[],
                settle_steps=0,
                settle_success=False,
            )

        # Reset all physics state to initial reference configuration
        model = self.model
        data = self.data
        mujoco.mj_resetData(model, data)

        # Write spawn poses into qpos and park unused slots
        self._apply_spawn_poses(spawn_poses, n_active)
        mujoco.mj_forward(model, data)

        # Settle phase: let bricks fall and stabilize
        settle_success, settle_steps = self._settle(n_active)

        if not settle_success:
            self._metrics._record_failure(_FAIL_SETTLE_TIMEOUT)
            return EpisodeInfo(
                seed=seed,
                level=level,
                brick_types=active_brick_types,
                spawn_poses=spawn_poses,
                settle_steps=settle_steps,
                settle_success=False,
            )

        self._metrics._record_success(settle_steps)
        return EpisodeInfo(
            seed=seed,
            level=level,
            brick_types=active_brick_types,
            spawn_poses=spawn_poses,
            settle_steps=settle_steps,
            settle_success=True,
        )

    # -----------------------------------------------------------------------
    # Curriculum
    # -----------------------------------------------------------------------

    def _n_active_for_level(self, level: int, rng: np.random.Generator) -> int:
        """Return number of active brick slots for a given curriculum level."""
        if level == LEVEL_SINGLE_BRICK:
            return 1
        elif level == LEVEL_SINGLE_CONNECTION:
            return 1
        else:  # LEVEL_MULTI_STEP or any higher level
            n_max = min(self.max_bricks, 4)
            return int(rng.integers(2, n_max + 1))

    # -----------------------------------------------------------------------
    # Spawn sampling
    # -----------------------------------------------------------------------

    def _sample_spawn_poses(
        self, rng: np.random.Generator, n_bricks: int
    ) -> list[SpawnPose] | None:
        """Sample constraint-satisfying spawn poses for n_bricks bricks.

        Enforces:
        - Position within spawn_x_range x spawn_y_range
        - Min-distance between all brick centers
        - Optional random yaw rotation

        Returns:
            List of SpawnPose (one per brick), or None if max_attempts
            is exhausted for any brick.
        """
        poses: list[SpawnPose] = []
        placed_xy: list[tuple[float, float]] = []

        for _ in range(n_bricks):
            placed = False
            for _ in range(self._max_spawn_attempts):
                x = float(rng.uniform(*self._spawn_x_range))
                y = float(rng.uniform(*self._spawn_y_range))

                # Min-distance check against already-placed bricks
                too_close = any(
                    math.sqrt((x - px) ** 2 + (y - py) ** 2) < self._min_spawn_distance
                    for px, py in placed_xy
                )
                if too_close:
                    continue

                quat = (
                    _yaw_to_quat(float(rng.uniform(0.0, 2.0 * math.pi)))
                    if self._random_yaw
                    else _IDENTITY_QUAT
                )
                poses.append(SpawnPose(position=(x, y, self._spawn_z), quaternion=quat))
                placed_xy.append((x, y))
                placed = True
                break

            if not placed:
                return None  # exceeded max_attempts for this brick

        return poses

    def _enable_slot_contacts(self, slot_idx: int) -> None:
        """Restore original contact bits for a brick slot (makes it physically active)."""
        model = self.model
        for j, g in enumerate(self._slot_geom_ids[slot_idx]):
            model.geom_contype[g] = self._slot_geom_ctypes[slot_idx][j]
            model.geom_conaffinity[g] = self._slot_geom_caffinities[slot_idx][j]

    def _disable_slot_contacts(self, slot_idx: int) -> None:
        """Zero contact bits for a brick slot (makes it transparent to physics)."""
        model = self.model
        for g in self._slot_geom_ids[slot_idx]:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0

    def _apply_spawn_poses(self, poses: list[SpawnPose], n_active: int) -> None:
        """Write spawn poses into data.qpos and update contact flags.

        Enables contacts for active slots and disables them for unused slots.
        Parked slots are left at PARK_POS with contacts disabled so they do
        not interact with the floor or workspace during the settle phase.

        Args:
            poses: Spawn poses for active slots (length == n_active).
            n_active: Number of active slots.
        """
        data = self.data

        # Enable contacts for active slots and write their spawn poses
        for i, pose in enumerate(poses):
            self._enable_slot_contacts(i)
            addr = self._slot_qpos_addrs[i]
            data.qpos[addr : addr + 3] = pose.position
            data.qpos[addr + 3 : addr + 7] = pose.quaternion

        # Disable contacts for unused slots and park their positions
        for i in range(n_active, self.max_bricks):
            self._disable_slot_contacts(i)
            addr = self._slot_qpos_addrs[i]
            data.qpos[addr : addr + 3] = _PARK_POS
            data.qpos[addr + 3 : addr + 7] = _IDENTITY_QUAT

    # -----------------------------------------------------------------------
    # Settle phase
    # -----------------------------------------------------------------------

    def _settle(self, n_active: int) -> tuple[bool, int]:
        """Step physics until active bricks stabilize.

        Checks velocity and penetration every ``settle_check_interval`` steps.
        Returns early if both thresholds are met.

        Args:
            n_active: Number of active brick slots to monitor.

        Returns:
            (success, steps_taken) — success=True if settled within budget.
        """
        model = self.model
        data = self.data
        active_body_ids = self._slot_body_ids[:n_active]

        for step in range(self._settle_max_steps):
            mujoco.mj_step(model, data)

            if (step + 1) % self._settle_check_interval != 0:
                continue

            # Velocity check: all active brick bodies below threshold
            max_vel = 0.0
            for body_id in active_body_ids:
                # data.cvel[body_id] = [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
                lin_speed = float(np.linalg.norm(data.cvel[body_id, 3:]))
                if lin_speed > max_vel:
                    max_vel = lin_speed

            # Penetration check: max negative contact distance
            max_pen = 0.0
            for k in range(data.ncon):
                dist = float(data.contact[k].dist)
                if dist < 0.0 and -dist > max_pen:
                    max_pen = -dist

            if max_vel < self._settle_velocity_threshold and max_pen < self._settle_max_penetration:
                return True, step + 1

        return False, self._settle_max_steps
