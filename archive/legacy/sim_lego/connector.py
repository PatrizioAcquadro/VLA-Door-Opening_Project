"""Connector metadata for LEGO bricks and baseplates (Phase 1.2.1+).

Provides stable IDs and positions for studs and tubes, used for
assembly verification, reward computation, and alignment checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.lego.constants import (
    STUD_COLLISION_RADIUS,
    STUD_HALF_HEIGHT,
    STUD_PITCH,
    TUBE_CAPSULE_HALF_HEIGHT,
    TUBE_RING_RADIUS,
    BaseplateType,
    BrickType,
)


@dataclass(frozen=True)
class ConnectorPoint:
    """A single connector (stud or tube) on a brick.

    Attributes:
        id: Stable identifier, e.g., "stud_0_1" or "tube_2".
        kind: "stud" or "tube".
        position: (x, y, z) relative to brick origin (center of bottom face).
        axis: Unit vector for insertion axis.
        radius: Collision radius in meters.
    """

    id: str
    kind: str
    position: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0


@dataclass(frozen=True)
class BrickConnectors:
    """Complete connector metadata for a single brick type.

    Attributes:
        brick_name: e.g., "2x4".
        studs: Tuple of ConnectorPoint for all studs.
        tubes: Tuple of ConnectorPoint for all tubes.
    """

    brick_name: str
    studs: tuple[ConnectorPoint, ...]
    tubes: tuple[ConnectorPoint, ...]

    @property
    def n_studs(self) -> int:
        return len(self.studs)

    @property
    def n_tubes(self) -> int:
        return len(self.tubes)

    def stud_positions_array(self) -> np.ndarray:
        """Return (n_studs, 3) array of stud positions."""
        return np.array([s.position for s in self.studs], dtype=np.float64)

    def tube_positions_array(self) -> np.ndarray:
        """Return (n_tubes, 3) array of tube positions."""
        return np.array([t.position for t in self.tubes], dtype=np.float64)


def get_brick_connectors(brick: BrickType) -> BrickConnectors:
    """Compute connector metadata for a brick type.

    Stud grid origin: first stud at (-pitch*(nx-1)/2, -pitch*(ny-1)/2)
    relative to brick origin (center of bottom face), per spec Section 3.1.

    Tube positions: centerline X=0 for 2-wide bricks, spaced at stud pitch.

    Args:
        brick: BrickType definition.

    Returns:
        BrickConnectors with all stud and tube positions.
    """
    studs: list[ConnectorPoint] = []
    for ix in range(brick.nx):
        for iy in range(brick.ny):
            x = -STUD_PITCH * (brick.nx - 1) / 2.0 + ix * STUD_PITCH
            y = -STUD_PITCH * (brick.ny - 1) / 2.0 + iy * STUD_PITCH
            z = brick.height + STUD_HALF_HEIGHT
            studs.append(
                ConnectorPoint(
                    id=f"stud_{ix}_{iy}",
                    kind="stud",
                    position=(x, y, z),
                    radius=STUD_COLLISION_RADIUS,
                )
            )

    tubes: list[ConnectorPoint] = []
    for it in range(brick.n_tubes):
        x = 0.0  # centerline for 2-wide bricks
        y = -STUD_PITCH * (brick.ny - 2) / 2.0 + it * STUD_PITCH
        z = TUBE_CAPSULE_HALF_HEIGHT  # inside brick cavity, above bottom face
        tubes.append(
            ConnectorPoint(
                id=f"tube_{it}",
                kind="tube",
                position=(x, y, z),
                radius=TUBE_RING_RADIUS,
            )
        )

    return BrickConnectors(
        brick_name=brick.name,
        studs=tuple(studs),
        tubes=tuple(tubes),
    )


@dataclass(frozen=True)
class BaseplateConnectors:
    """Connector metadata for a baseplate (studs only, no tubes).

    Attributes:
        baseplate_name: e.g., "8x8".
        studs: Tuple of ConnectorPoint for all studs.
    """

    baseplate_name: str
    studs: tuple[ConnectorPoint, ...]

    @property
    def n_studs(self) -> int:
        return len(self.studs)

    def stud_positions_array(self) -> np.ndarray:
        """Return (n_studs, 3) array of stud positions."""
        return np.array([s.position for s in self.studs], dtype=np.float64)


def get_baseplate_connectors(baseplate: BaseplateType) -> BaseplateConnectors:
    """Compute connector metadata for a baseplate.

    Stud grid centered on baseplate origin (center of bottom face).
    Studs at Z = baseplate.thickness + STUD_HALF_HEIGHT (on top surface).

    Args:
        baseplate: BaseplateType definition.

    Returns:
        BaseplateConnectors with all stud positions.
    """
    studs: list[ConnectorPoint] = []
    for ix in range(baseplate.nx_studs):
        for iy in range(baseplate.ny_studs):
            x = -STUD_PITCH * (baseplate.nx_studs - 1) / 2.0 + ix * STUD_PITCH
            y = -STUD_PITCH * (baseplate.ny_studs - 1) / 2.0 + iy * STUD_PITCH
            z = baseplate.thickness + STUD_HALF_HEIGHT
            studs.append(
                ConnectorPoint(
                    id=f"stud_{ix}_{iy}",
                    kind="stud",
                    position=(x, y, z),
                    radius=STUD_COLLISION_RADIUS,
                )
            )

    return BaseplateConnectors(
        baseplate_name=baseplate.name,
        studs=tuple(studs),
    )
