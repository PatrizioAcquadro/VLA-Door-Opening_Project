"""Mass computation for LEGO bricks and baseplates from geometry (Phase 1.2.1+).

Implements spec Section 3.4 formula:
    mass = density * (V_outer - V_inner + V_studs + V_tubes)

All volumes computed in m^3, mass in kg.
"""

from __future__ import annotations

import math

from sim.lego.constants import (
    DENSITY_ABS,
    INTER_BRICK_GAP,
    STUD_HEIGHT,
    STUD_PITCH,
    STUD_VISUAL_RADIUS,
    TOP_THICKNESS,
    TUBE_CAPSULE_HALF_HEIGHT,
    TUBE_RING_RADIUS,
    WALL_THICKNESS,
    BaseplateType,
    BrickType,
)


def compute_brick_mass(brick: BrickType) -> float:
    """Compute brick mass in kg from geometry and ABS density.

    Args:
        brick: BrickType definition.

    Returns:
        Mass in kilograms.
    """
    outer_x = brick.nx * STUD_PITCH - INTER_BRICK_GAP
    outer_y = brick.ny * STUD_PITCH - INTER_BRICK_GAP
    outer_z = brick.height

    v_outer = outer_x * outer_y * outer_z

    inner_x = outer_x - 2 * WALL_THICKNESS
    inner_y = outer_y - 2 * WALL_THICKNESS
    inner_z = outer_z - TOP_THICKNESS
    v_inner = max(inner_x, 0.0) * max(inner_y, 0.0) * max(inner_z, 0.0)

    v_studs = brick.n_studs * math.pi * STUD_VISUAL_RADIUS**2 * STUD_HEIGHT

    # Tube volume approximation: hollow cylinder per spec Section 3.4
    tube_outer_r = TUBE_RING_RADIUS + 0.000855  # ~3.255 mm
    tube_inner_r = TUBE_RING_RADIUS  # 2.4 mm
    tube_h = TUBE_CAPSULE_HALF_HEIGHT * 2
    v_tubes = brick.n_tubes * math.pi * (tube_outer_r**2 - tube_inner_r**2) * tube_h

    volume_solid = v_outer - v_inner + v_studs + v_tubes
    return DENSITY_ABS * volume_solid


def compute_baseplate_mass(baseplate: BaseplateType) -> float:
    """Compute baseplate mass in kg from geometry and ABS density.

    Solid thin plate (no hollow interior) plus studs on top.

    Args:
        baseplate: BaseplateType definition.

    Returns:
        Mass in kilograms.
    """
    plate_x = baseplate.nx_studs * STUD_PITCH - INTER_BRICK_GAP
    plate_y = baseplate.ny_studs * STUD_PITCH - INTER_BRICK_GAP
    v_plate = plate_x * plate_y * baseplate.thickness
    v_studs = baseplate.n_studs * math.pi * STUD_VISUAL_RADIUS**2 * STUD_HEIGHT
    return DENSITY_ABS * (v_plate + v_studs)
