"""Frozen geometry and physics constants for LEGO simulation (Phase 1.2.1).

All dimensions in meters (MuJoCo convention). Values frozen from
press-fit spec (docs/press-fit-spec.md, Sections 3 & 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Stud geometry (frozen) ---
STUD_PITCH: float = 0.008  # 8.0 mm center-to-center
STUD_COLLISION_RADIUS: float = 0.00235  # 2.35 mm (50 um undersize for clearance)
STUD_VISUAL_RADIUS: float = 0.0024  # 2.4 mm (true LEGO dimension)
STUD_HALF_HEIGHT: float = 0.00085  # half of 1.7 mm
STUD_HEIGHT: float = 0.0017  # 1.7 mm above brick top

# --- Tube geometry (tuned in Phase 1.2.2) ---
# Ring radius set so outer extent (ring_radius + capsule_radius = 3.5 mm)
# reaches surrounding stud faces at 3.31 mm from tube center, creating
# ~0.19 mm interference for press-fit retention.
# 8 capsules at 45° intervals align with the stud diagonal positions
# (studs at ±4mm, ±4mm → 45° from tube center).
TUBE_RING_RADIUS: float = 0.003  # 3.0 mm (capsule centers)
TUBE_CAPSULE_RADIUS: float = 0.00055  # 0.55 mm (tuned Phase 1.2.2)
TUBE_CAPSULE_HALF_HEIGHT: float = 0.00085  # half of 1.7 mm (matches stud height)
TUBE_CAPSULE_COUNT: int = 8
TUBE_CAPSULE_ANGULAR_SPACING: float = 2.0 * math.pi / TUBE_CAPSULE_COUNT  # 45 deg

# --- Brick shell geometry (frozen) ---
BRICK_HEIGHT: float = 0.0096  # 9.6 mm body only
WALL_THICKNESS: float = 0.0012  # 1.2 mm
TOP_THICKNESS: float = 0.001  # 1.0 mm top wall
INTER_BRICK_GAP: float = 0.0002  # 0.2 mm total (0.1 per side)

# --- Material ---
DENSITY_ABS: float = 1050.0  # kg/m^3 (ABS plastic)

# --- Contact isolation (tuned Phase 1.2.2) ---
# bit 0 = robot/environment, bit 1 = LEGO surface, bit 2 = stud-tube internal
LEGO_CONTYPE_SURFACE: int = 2  # brick surface: bit 1
LEGO_CONAFFINITY_SURFACE: int = 3  # brick surface responds to robot(0) + LEGO(1)
LEGO_CONTYPE_STUD: int = 6  # stud: bits 1+2 (contacts surfaces and tubes)
LEGO_CONAFFINITY_STUD: int = 7  # stud responds to robot(0) + LEGO(1) + stud-tube(2)
LEGO_CONTYPE_TUBE: int = 4  # tube: bit 2 only (contacts studs, NOT surfaces)
LEGO_CONAFFINITY_TUBE: int = 4  # tube responds to stud-tube(2) only

# --- Geom groups ---
COLLISION_GROUP: int = 3  # matches Alex robot collision geoms
VISUAL_GROUP: int = 0  # visual-only geoms

# --- Default brick color (LEGO red, RGBA) ---
DEFAULT_BRICK_COLOR: tuple[float, float, float, float] = (0.78, 0.09, 0.09, 1.0)


@dataclass(frozen=True)
class BrickType:
    """Parametric definition of a LEGO brick type."""

    name: str  # e.g., "2x2"
    nx: int  # studs along X
    ny: int  # studs along Y
    height: float = BRICK_HEIGHT

    @property
    def n_studs(self) -> int:
        return self.nx * self.ny

    @property
    def n_tubes(self) -> int:
        """For 2-wide bricks: ny - 1 tubes along centerline."""
        return self.ny - 1

    @property
    def shell_half_x(self) -> float:
        return (self.nx * STUD_PITCH - INTER_BRICK_GAP) / 2.0

    @property
    def shell_half_y(self) -> float:
        return (self.ny * STUD_PITCH - INTER_BRICK_GAP) / 2.0

    @property
    def shell_half_z(self) -> float:
        return self.height / 2.0


BRICK_TYPES: dict[str, BrickType] = {
    "2x2": BrickType(name="2x2", nx=2, ny=2),
    "2x4": BrickType(name="2x4", nx=2, ny=4),
    "2x6": BrickType(name="2x6", nx=2, ny=6),
}

# --- Baseplate geometry (Phase 1.2.3) ---
BASEPLATE_THICKNESS: float = 0.0032  # 3.2 mm (standard LEGO plate height)
DEFAULT_BASEPLATE_COLOR: tuple[float, float, float, float] = (0.16, 0.50, 0.16, 1.0)


@dataclass(frozen=True)
class BaseplateType:
    """Parametric definition of a LEGO baseplate."""

    name: str  # e.g., "8x8"
    nx_studs: int  # studs along X
    ny_studs: int  # studs along Y
    thickness: float = BASEPLATE_THICKNESS

    @property
    def n_studs(self) -> int:
        return self.nx_studs * self.ny_studs

    @property
    def half_x(self) -> float:
        return (self.nx_studs * STUD_PITCH - INTER_BRICK_GAP) / 2.0

    @property
    def half_y(self) -> float:
        return (self.ny_studs * STUD_PITCH - INTER_BRICK_GAP) / 2.0

    @property
    def half_z(self) -> float:
        return self.thickness / 2.0


BASEPLATE_TYPES: dict[str, BaseplateType] = {
    "8x8": BaseplateType(name="8x8", nx_studs=8, ny_studs=8),
}
