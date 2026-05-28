"""Measurement utilities for LEGO contact physics tests (Phase 1.2.2).

Provides helpers for running insertions, applying force ramps, and
measuring jitter/drift/penetration/energy during contact experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sim.lego.constants import BRICK_HEIGHT, STUD_HALF_HEIGHT, STUD_HEIGHT

if TYPE_CHECKING:
    from sim.lego.connection_manager import ConnectionManager


@dataclass
class InsertionResult:
    """Result of running an insertion experiment."""

    success: bool
    engagement_fraction: float  # 1.0 = fully engaged
    time_to_engage_s: float  # seconds to engagement (inf if not engaged)
    max_penetration_m: float  # max penetration depth observed
    max_energy_J: float  # max total energy observed
    final_top_z: float  # final Z position of top brick body
    max_contact_count: int = 0  # max simultaneous active contacts
    position_trace: list[float] = field(default_factory=list)  # Z trace


def get_top_body_id(model, top_name: str = "top", brick_name: str = "2x2") -> int:
    """Get body ID for the top brick."""
    import mujoco

    body_name = f"{top_name}_{brick_name}"
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))


def get_top_joint_id(model, top_name: str = "top", brick_name: str = "2x2") -> int:
    """Get joint ID for the top brick's freejoint."""
    import mujoco

    joint_name = f"{top_name}_{brick_name}_joint"
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))


def get_max_penetration(data) -> float:
    """Return max penetration depth across all active contacts (meters).

    MuJoCo stores contact distance in ``data.contact[i].dist``.
    Negative values indicate penetration.
    """
    if data.ncon == 0:
        return 0.0
    dists = np.array([data.contact[i].dist for i in range(data.ncon)])
    neg = dists[dists < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.abs(neg).max())


def get_total_energy(model, data) -> float:
    """Return total energy (potential + kinetic).

    Requires ``option.flag.energy = True`` in the MJCF.
    ``data.energy`` is a 2-element array: [potential, kinetic].
    """
    return float(data.energy[0] + data.energy[1])


def get_active_contact_count(data) -> int:
    """Return number of active contacts."""
    return int(data.ncon)


def run_insertion(
    model,
    data,
    base_brick_name: str = "2x2",
    top_name: str = "top",
    base_height: float = 0.05,
    approach_velocity: float = 0.001,
    max_time_s: float = 2.0,
    check_interval_steps: int = 10,
    connection_manager: ConnectionManager | None = None,
    base_surface_height: float | None = None,
) -> InsertionResult:
    """Run a controlled insertion by setting downward velocity on the top brick.

    Sets the top brick's freejoint linear velocity to ``(0, 0, -approach_velocity)``
    and steps physics until engagement or timeout.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        base_brick_name: Name of brick type (e.g., "2x2").
        top_name: Name prefix of top brick body.
        base_height: Z of base brick origin.
        approach_velocity: Downward approach speed (m/s, positive = downward).
        max_time_s: Maximum simulation time before timeout.
        check_interval_steps: Steps between engagement checks.
        connection_manager: Optional ConnectionManager for hybrid retention.
        base_surface_height: Height of base surface (meters). Defaults to
            ``BRICK_HEIGHT`` for brick-on-brick; use ``baseplate.thickness``
            for brick-on-baseplate scenes.

    Returns:
        InsertionResult with engagement status and measurements.
    """
    import mujoco

    if base_surface_height is None:
        base_surface_height = BRICK_HEIGHT

    body_id = get_top_body_id(model, top_name, base_brick_name)

    dt = model.opt.timestep
    max_steps = int(max_time_s / dt)
    # Engaged when top brick body Z drops below the base stud center.
    engage_z_threshold = base_height + base_surface_height + STUD_HALF_HEIGHT

    max_pen = 0.0
    max_energy = 0.0
    max_contacts = 0
    z_trace: list[float] = []
    engaged = False
    engage_time = float("inf")

    # Apply a downward force to drive insertion. The force is proportional
    # to brick mass to ensure sufficient push through the capsule ring friction.
    # F = mass * g * force_multiplier provides controllable insertion pressure.
    brick_mass = model.body_mass[body_id]
    insertion_force = brick_mass * 9.81 * 5.0  # 5x gravity force

    for step in range(max_steps):
        # Apply downward force during approach; stop once engaged
        if not engaged:
            data.xfrc_applied[body_id, 2] = -insertion_force

        mujoco.mj_step(model, data)

        if connection_manager is not None:
            connection_manager.update()

        # Check for NaN
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            return InsertionResult(
                success=False,
                engagement_fraction=0.0,
                time_to_engage_s=float("inf"),
                max_penetration_m=max_pen,
                max_energy_J=max_energy,
                final_top_z=float("nan"),
                max_contact_count=max_contacts,
                position_trace=z_trace,
            )

        # Track measurements
        pen = get_max_penetration(data)
        max_pen = max(max_pen, pen)
        energy = get_total_energy(model, data)
        max_energy = max(max_energy, abs(energy))
        max_contacts = max(max_contacts, data.ncon)

        if step % check_interval_steps == 0:
            top_z = data.xpos[body_id][2]
            z_trace.append(top_z)

            if not engaged and top_z <= engage_z_threshold:
                engaged = True
                engage_time = step * dt
                # Clear insertion force once engaged
                data.xfrc_applied[body_id, :] = 0.0

    final_z = data.xpos[body_id][2]
    # Final engagement check
    if not engaged and final_z <= engage_z_threshold:
        engaged = True
        engage_time = max_steps * dt

    # Ensure force is cleared
    data.xfrc_applied[body_id, :] = 0.0

    return InsertionResult(
        success=engaged,
        engagement_fraction=1.0 if engaged else 0.0,
        time_to_engage_s=engage_time,
        max_penetration_m=max_pen,
        max_energy_J=max_energy,
        final_top_z=final_z,
        max_contact_count=max_contacts,
        position_trace=z_trace,
    )


def run_insertion_gravity_only(
    model,
    data,
    base_brick_name: str = "2x2",
    top_name: str = "top",
    base_height: float = 0.05,
    max_time_s: float = 2.0,
    check_interval_steps: int = 10,
    base_surface_height: float | None = None,
) -> InsertionResult:
    """Run insertion driven only by gravity (no velocity override).

    Useful for testing if gravity alone can seat a brick.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        base_brick_name: Name of brick type.
        top_name: Name prefix of top brick body.
        base_height: Z of base brick origin.
        max_time_s: Maximum simulation time.
        check_interval_steps: Steps between checks.
        base_surface_height: Height of base surface (meters). Defaults to
            ``BRICK_HEIGHT``; use ``baseplate.thickness`` for baseplates.

    Returns:
        InsertionResult.
    """
    import mujoco

    if base_surface_height is None:
        base_surface_height = BRICK_HEIGHT

    body_id = get_top_body_id(model, top_name, base_brick_name)
    dt = model.opt.timestep
    max_steps = int(max_time_s / dt)
    engage_z_threshold = base_height + base_surface_height + STUD_HALF_HEIGHT

    max_pen = 0.0
    max_energy = 0.0
    max_contacts = 0
    z_trace: list[float] = []
    engaged = False
    engage_time = float("inf")

    for step in range(max_steps):
        mujoco.mj_step(model, data)

        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            return InsertionResult(
                success=False,
                engagement_fraction=0.0,
                time_to_engage_s=float("inf"),
                max_penetration_m=max_pen,
                max_energy_J=max_energy,
                final_top_z=float("nan"),
                max_contact_count=max_contacts,
                position_trace=z_trace,
            )

        pen = get_max_penetration(data)
        max_pen = max(max_pen, pen)
        energy = get_total_energy(model, data)
        max_energy = max(max_energy, abs(energy))
        max_contacts = max(max_contacts, data.ncon)

        if step % check_interval_steps == 0:
            top_z = data.xpos[body_id][2]
            z_trace.append(top_z)

            if not engaged and top_z <= engage_z_threshold:
                engaged = True
                engage_time = step * dt

    final_z = data.xpos[body_id][2]
    if not engaged and final_z <= engage_z_threshold:
        engaged = True
        engage_time = max_steps * dt

    return InsertionResult(
        success=engaged,
        engagement_fraction=1.0 if engaged else 0.0,
        time_to_engage_s=engage_time,
        max_penetration_m=max_pen,
        max_energy_J=max_energy,
        final_top_z=final_z,
        max_contact_count=max_contacts,
        position_trace=z_trace,
    )


def apply_force_ramp(
    model,
    data,
    body_name: str,
    direction: np.ndarray,
    force_rate: float = 0.5,
    max_force: float = 5.0,
    displacement_threshold: float | None = None,
    connection_manager: ConnectionManager | None = None,
) -> float:
    """Apply a linearly ramping external force and return detachment force.

    Applies force via ``data.xfrc_applied`` on the specified body.
    Force increases at ``force_rate`` N/s. The body is considered detached
    when it displaces by more than ``displacement_threshold`` from its
    initial position.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (should be in post-insertion state).
        body_name: Name of the body to apply force to.
        direction: Unit vector for force direction (world frame).
        force_rate: Force increase rate in N/s.
        max_force: Maximum force before giving up.
        displacement_threshold: Distance threshold for detachment (meters).
            Defaults to STUD_HEIGHT / 2.

    Returns:
        Force magnitude (N) at which detachment occurred.
        Returns ``max_force`` if body did not detach.
    """
    import mujoco

    if displacement_threshold is None:
        displacement_threshold = STUD_HEIGHT / 2.0

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    initial_pos = data.xpos[body_id].copy()

    dt = float(model.opt.timestep)
    current_force = 0.0

    while current_force < max_force:
        current_force += force_rate * dt

        # Apply force in world frame
        data.xfrc_applied[body_id, :3] = direction * current_force
        mujoco.mj_step(model, data)

        if connection_manager is not None:
            connection_manager.update()

        if np.any(np.isnan(data.qpos)):
            break

        displacement = np.linalg.norm(data.xpos[body_id] - initial_pos)
        if displacement > displacement_threshold:
            # Clear applied force
            data.xfrc_applied[body_id, :] = 0
            return float(current_force)

    # Clear applied force
    data.xfrc_applied[body_id, :] = 0
    return max_force


def measure_position_jitter(
    model,
    data,
    body_name: str,
    duration_s: float = 1.0,
    connection_manager: ConnectionManager | None = None,
) -> float:
    """Measure RMS position deviation over a duration.

    Simulates for ``duration_s`` with no external forces and measures
    the RMS deviation from the mean position.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_name: Name of the body to measure.
        duration_s: Measurement duration in seconds.

    Returns:
        RMS position deviation in meters.
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    dt = float(model.opt.timestep)
    n_steps = int(duration_s / dt)

    positions: list[np.ndarray] = []
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        if connection_manager is not None:
            connection_manager.update()
        positions.append(data.xpos[body_id].copy())

    if len(positions) < 2:
        return 0.0

    pos_array = np.array(positions)
    mean_pos = pos_array.mean(axis=0)
    deviations = np.linalg.norm(pos_array - mean_pos, axis=1)
    return float(np.sqrt(np.mean(deviations**2)))


def measure_position_drift(
    model,
    data,
    body_name: str,
    duration_s: float = 5.0,
    connection_manager: ConnectionManager | None = None,
) -> float:
    """Measure maximum position drift over a duration.

    Simulates for ``duration_s`` with no external forces and returns
    the maximum displacement from the initial position.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        body_name: Name of the body to measure.
        duration_s: Measurement duration in seconds.

    Returns:
        Maximum drift in meters.
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    initial_pos = data.xpos[body_id].copy()

    dt = model.opt.timestep
    n_steps = int(duration_s / dt)

    max_drift = 0.0
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        if connection_manager is not None:
            connection_manager.update()
        drift = float(np.linalg.norm(data.xpos[body_id] - initial_pos))
        max_drift = max(max_drift, drift)

    return float(max_drift)


def perform_insertion_then_measure(
    model,
    data,
    base_brick_name: str = "2x2",
    top_name: str = "top",
    base_height: float = 0.05,
    approach_velocity: float = 0.001,
    settle_time_s: float = 0.5,
    connection_manager: ConnectionManager | None = None,
    base_surface_height: float | None = None,
) -> InsertionResult:
    """Run insertion then let the system settle.

    Performs a controlled insertion followed by a settling period with
    no applied velocity, allowing contacts to stabilize.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        base_brick_name: Brick type name.
        top_name: Name prefix of top brick.
        base_height: Z of base brick origin.
        approach_velocity: Downward speed during insertion.
        settle_time_s: Time to simulate after engagement for settling.
        connection_manager: Optional ConnectionManager for hybrid retention.
        base_surface_height: Height of base surface (meters). Defaults to
            ``BRICK_HEIGHT``; use ``baseplate.thickness`` for baseplates.

    Returns:
        InsertionResult after insertion + settling.
    """
    import mujoco

    # Phase 1: Insert (physics-only, no weld activation during insertion)
    result = run_insertion(
        model,
        data,
        base_brick_name=base_brick_name,
        top_name=top_name,
        base_height=base_height,
        approach_velocity=approach_velocity,
        max_time_s=2.0,
        base_surface_height=base_surface_height,
    )

    if not result.success:
        return result

    # Phase 2: Settle (no applied forces, let physics stabilize)
    body_id = get_top_body_id(model, top_name, base_brick_name)
    joint_id = get_top_joint_id(model, top_name, base_brick_name)
    qvel_adr = model.jnt_dofadr[joint_id]

    # Clear forces and velocity
    data.xfrc_applied[body_id, :] = 0.0
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    dt = model.opt.timestep
    settle_steps = int(settle_time_s / dt)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if connection_manager is not None:
            connection_manager.update()
        pen = get_max_penetration(data)
        result.max_penetration_m = max(result.max_penetration_m, pen)

    result.final_top_z = data.xpos[body_id][2]
    return result
