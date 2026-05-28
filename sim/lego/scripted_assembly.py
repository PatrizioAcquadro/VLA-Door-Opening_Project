"""Force-based scripted assembly executor for MVP-3 tasks (Phase 1.2.6).

Executes assembly goals by kinematically positioning bricks above targets
and using ``xfrc_applied`` forces for physics-based press-fit insertion.
This proves the assembly pipeline works end-to-end with real contact physics.

Usage::

    from sim.lego.scripted_assembly import ScriptedAssembler

    em = EpisodeManager(brick_slots=["2x2", "2x4"])
    info = em.reset(seed=42, level=LEVEL_SINGLE_CONNECTION)
    goal = generate_assembly_goal(info, bp_type, bp_world_pos, seed=42)

    assembler = ScriptedAssembler(em.model, em.data)
    result = assembler.execute_assembly(goal)
    print(f"Placed {result.n_successful}/{result.n_total} bricks")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.lego.constants import BRICK_TYPES, STUD_HALF_HEIGHT
from sim.lego.task import (
    AssemblyGoal,
    AssemblyResult,
    PlacementResult,
    PlacementTarget,
    check_placement,
    compute_brick_on_brick_target,
    evaluate_assembly,
)

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AssemblyStepLog:
    """Log entry for one placement step.

    Attributes:
        step_index: Ordinal of this placement in the assembly sequence.
        target: The placement target.
        approach_position: World-frame XYZ of approach waypoint.
        insertion_steps: Physics steps consumed during insertion.
        engaged: True if Z engagement was detected.
        settle_steps: Physics steps consumed during settle.
        stable: True if brick was stable after settle.
    """

    step_index: int
    target: PlacementTarget
    approach_position: tuple[float, float, float]
    insertion_steps: int
    engaged: bool
    settle_steps: int
    stable: bool


# ---------------------------------------------------------------------------
# ScriptedAssembler
# ---------------------------------------------------------------------------


class ScriptedAssembler:
    """Force-based scripted assembly executor.

    Executes an ``AssemblyGoal`` by sequentially placing each brick:
    1. Kinematic positioning: write brick qpos to approach waypoint (above target)
    2. Physics insertion: apply downward force via ``xfrc_applied``
    3. Settle: run physics with no external forces until stable

    This uses real soft press-fit contact physics — connections form through
    stud/tube geometric interlock and friction, not weld constraints.

    Args:
        model: MuJoCo model (from EpisodeManager).
        data: MuJoCo data (from EpisodeManager).
        approach_height_m: Height above target for approach position.
        insertion_force_factor: Gravity multiplier for insertion force.
        max_insertion_steps: Max physics steps per insertion attempt.
        settle_steps: Physics steps for post-insertion settle.
        settle_velocity_threshold: Max velocity (m/s) for "settled".
        xy_tol_m: XY tolerance for placement success check.
        z_margin_m: Z margin for engagement check.
    """

    def __init__(
        self,
        model,
        data,
        approach_height_m: float = 0.02,
        insertion_force_factor: float = 5.0,
        max_insertion_steps: int = 2000,
        settle_steps: int = 500,
        settle_velocity_threshold: float = 0.001,
        xy_tol_m: float = 0.001,
        z_margin_m: float = 0.0005,
    ) -> None:
        self._model = model
        self._data = data
        self._approach_height = approach_height_m
        self._force_factor = insertion_force_factor
        self._max_insertion_steps = max_insertion_steps
        self._settle_steps = settle_steps
        self._settle_vel_thresh = settle_velocity_threshold
        self._xy_tol = xy_tol_m
        self._z_margin = z_margin_m
        self._step_logs: list[AssemblyStepLog] = []

    @property
    def step_logs(self) -> list[AssemblyStepLog]:
        """Log entries for each placement step (populated after execute_assembly)."""
        return list(self._step_logs)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_body_and_joint(self, target: PlacementTarget) -> tuple[int, int, int]:
        """Resolve body ID, joint ID, and qpos address for a target's brick slot."""
        bt = BRICK_TYPES[target.brick_type]
        body_name = f"brick_{target.slot_index}_{bt.name}"
        joint_name = f"brick_{target.slot_index}_{bt.name}_joint"

        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_addr = int(self._model.jnt_qposadr[joint_id])

        return body_id, joint_id, qpos_addr

    def _compute_engage_z(self, target: PlacementTarget) -> float:
        """Compute the Z threshold for engagement detection.

        A brick is "engaged" when its body Z drops to or below the target Z
        plus a small margin accounting for stud height.
        """
        return target.target_position[2] + STUD_HALF_HEIGHT

    def _position_brick_at_approach(
        self, target: PlacementTarget, body_id: int, qpos_addr: int
    ) -> tuple[float, float, float]:
        """Write brick qpos to approach position above target.

        Sets position, identity quaternion, and zeros all velocities.
        Calls mj_forward to update kinematics.

        Returns:
            The approach position (x, y, z).
        """
        # Compute approach position: target XY, elevated Z
        approach_pos = (
            target.target_position[0],
            target.target_position[1],
            target.target_position[2] + self._approach_height,
        )

        # Write position + identity quaternion into qpos
        self._data.qpos[qpos_addr : qpos_addr + 3] = approach_pos
        self._data.qpos[qpos_addr + 3 : qpos_addr + 7] = target.target_quaternion

        # Zero velocities for this body's DOFs
        vel_addr = int(
            self._model.jnt_dofadr[
                mujoco.mj_name2id(
                    self._model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"brick_{target.slot_index}_{BRICK_TYPES[target.brick_type].name}_joint",
                )
            ]
        )
        self._data.qvel[vel_addr : vel_addr + 6] = 0.0

        # Clear any residual applied forces
        self._data.xfrc_applied[body_id, :] = 0.0

        # Update kinematics
        mujoco.mj_forward(self._model, self._data)

        return approach_pos

    def _run_insertion(self, body_id: int, engage_z: float) -> tuple[bool, int]:
        """Apply downward force and step until engagement or timeout.

        Args:
            body_id: MuJoCo body ID of the brick being inserted.
            engage_z: Z threshold for engagement detection.

        Returns:
            (engaged, steps) — whether engagement was detected and steps consumed.
        """
        brick_mass = float(self._model.body_mass[body_id])
        insertion_force = brick_mass * 9.81 * self._force_factor

        engaged = False
        steps = 0

        for step in range(self._max_insertion_steps):
            if not engaged:
                self._data.xfrc_applied[body_id, 2] = -insertion_force

            mujoco.mj_step(self._model, self._data)
            steps += 1

            # NaN check
            if np.any(np.isnan(self._data.qpos)) or np.any(np.isnan(self._data.qvel)):
                self._data.xfrc_applied[body_id, :] = 0.0
                return False, steps

            # Check engagement every 10 steps
            if step % 10 == 0:
                body_z = float(self._data.xpos[body_id][2])
                if not engaged and body_z <= engage_z:
                    engaged = True
                    self._data.xfrc_applied[body_id, :] = 0.0

        # Final check
        if not engaged:
            body_z = float(self._data.xpos[body_id][2])
            if body_z <= engage_z:
                engaged = True

        # Ensure force is cleared
        self._data.xfrc_applied[body_id, :] = 0.0

        return engaged, steps

    def _run_settle(self, body_id: int) -> tuple[bool, int]:
        """Run settle phase with no external forces.

        Args:
            body_id: Body ID to monitor for stability.

        Returns:
            (stable, steps) — whether the brick stabilized.
        """
        for step in range(self._settle_steps):
            mujoco.mj_step(self._model, self._data)

            if (step + 1) % 50 == 0:
                lin_speed = float(np.linalg.norm(self._data.cvel[body_id, 3:]))
                if lin_speed < self._settle_vel_thresh:
                    return True, step + 1

        return False, self._settle_steps

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def execute_placement(self, target: PlacementTarget) -> PlacementResult:
        """Execute one brick placement.

        1. Position brick at approach waypoint (above target)
        2. Apply downward force for physics-based insertion
        3. Settle with no external forces
        4. Check placement success (XY + Z)

        Args:
            target: PlacementTarget specifying where to place the brick.

        Returns:
            PlacementResult with success status and metrics.
        """
        body_id, joint_id, qpos_addr = self._resolve_body_and_joint(target)
        engage_z = self._compute_engage_z(target)

        # 1. Position at approach
        approach_pos = self._position_brick_at_approach(target, body_id, qpos_addr)

        # 2. Insert via force
        engaged, insertion_steps = self._run_insertion(body_id, engage_z)

        # 3. Settle
        stable, settle_steps = self._run_settle(body_id)

        # 4. Check placement
        success, xy_error = check_placement(
            self._model, self._data, target, self._xy_tol, self._z_margin
        )

        final_pos = tuple(float(v) for v in self._data.xpos[body_id])

        # Log
        self._step_logs.append(
            AssemblyStepLog(
                step_index=len(self._step_logs),
                target=target,
                approach_position=approach_pos,
                insertion_steps=insertion_steps,
                engaged=engaged,
                settle_steps=settle_steps,
                stable=stable,
            )
        )

        return PlacementResult(
            target=target,
            success=success,
            position_error_m=xy_error,
            z_engaged=engaged,
            stable=stable,
            insertion_steps=insertion_steps + settle_steps,
            final_position=final_pos,  # type: ignore[arg-type]
        )

    def execute_assembly(
        self,
        goal: AssemblyGoal,
        hold_duration_s: float = 2.0,
    ) -> AssemblyResult:
        """Execute a full assembly goal sequentially.

        Places each brick in order. For brick-on-brick targets, recomputes
        the target position from the base brick's live position at execution
        time.

        After all placements, runs a final stability hold.

        Args:
            goal: AssemblyGoal with ordered placement targets.
            hold_duration_s: Duration of final stability hold (seconds).

        Returns:
            AssemblyResult with per-brick results and aggregate metrics.
        """
        self._step_logs.clear()
        placements: list[PlacementResult] = []

        for target in goal.targets:
            # For brick-on-brick targets, recompute position from live base body
            actual_target = target
            if not target.base_body_name.startswith("baseplate_"):
                # This is a brick-on-brick placement — recompute from live position
                # Parse base brick type from body name: "brick_{idx}_{type}"
                parts = target.base_body_name.split("_", 2)
                base_bt = BRICK_TYPES[parts[2]]
                top_bt = BRICK_TYPES[target.brick_type]

                live_target_pos = compute_brick_on_brick_target(
                    self._model,
                    self._data,
                    target.base_body_name,
                    base_bt,
                    top_bt,
                    stud_ix=0,
                    stud_iy=0,
                )
                actual_target = PlacementTarget(
                    slot_index=target.slot_index,
                    brick_type=target.brick_type,
                    target_position=live_target_pos,
                    target_quaternion=target.target_quaternion,
                    base_body_name=target.base_body_name,
                )

            result = self.execute_placement(actual_target)
            placements.append(result)

        # Final evaluation with stability hold
        return evaluate_assembly(
            self._model,
            self._data,
            goal,
            placements,
            hold_duration_s=hold_duration_s,
        )
