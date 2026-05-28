"""Procedural MJCF generation for LEGO bricks (Phase 1.2.1).

Generates MuJoCo-compatible XML for parametric LEGO bricks with:
- Accurate collision geometry (box shell, cylinder studs, capsule-ring tubes)
- Visual geometry (MuJoCo primitives, slightly larger studs)
- Correct contact classes from defaults.xml
- Computed mass from ABS density

Usage:
    from sim.lego.brick_generator import generate_brick_mjcf, write_brick_assets
    from sim.lego.constants import BRICK_TYPES

    xml = generate_brick_mjcf(BRICK_TYPES["2x4"])
    write_brick_assets()  # writes all 3 brick files to sim/assets/lego/bricks/
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from sim.lego.connector import get_brick_connectors
from sim.lego.constants import (
    BRICK_TYPES,
    COLLISION_GROUP,
    DEFAULT_BRICK_COLOR,
    STUD_COLLISION_RADIUS,
    STUD_HALF_HEIGHT,
    STUD_VISUAL_RADIUS,
    TOP_THICKNESS,
    TUBE_CAPSULE_COUNT,
    TUBE_CAPSULE_HALF_HEIGHT,
    TUBE_CAPSULE_RADIUS,
    TUBE_RING_RADIUS,
    VISUAL_GROUP,
    WALL_THICKNESS,
    BrickType,
)
from sim.lego.mass import compute_brick_mass

# Path to the defaults.xml relative to bricks/ directory
_DEFAULTS_REL_PATH = "../defaults.xml"

# Path to the LEGO assets directory
_LEGO_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "lego"
_BRICKS_DIR = _LEGO_ASSETS_DIR / "bricks"


def _fmt(v: float, precision: int = 6) -> str:
    """Format a float, stripping trailing zeros."""
    return f"{v:.{precision}f}".rstrip("0").rstrip(".")


def _rgba_str(color: tuple[float, float, float, float]) -> str:
    return " ".join(_fmt(c, 3) for c in color)


def _pos_str(x: float, y: float, z: float) -> str:
    return f"{_fmt(x)} {_fmt(y)} {_fmt(z)}"


def generate_brick_body_xml(
    brick: BrickType,
    name_prefix: str = "brick",
    color: tuple[float, float, float, float] = DEFAULT_BRICK_COLOR,
    include_freejoint: bool = True,
) -> str:
    """Generate an MJCF ``<body>`` element for a LEGO brick.

    The body includes collision and visual geoms for the shell, studs, and
    tubes. Brick origin is at the center of the bottom face (Z=0).

    Args:
        brick: BrickType specification.
        name_prefix: Prefix for body and geom names (e.g., "brick" -> "brick_2x4").
        color: RGBA tuple for visual geoms.
        include_freejoint: Whether to add a freejoint (True for free bodies).

    Returns:
        XML string for the ``<body>`` element.
    """
    body_name = f"{name_prefix}_{brick.name}"
    mass = compute_brick_mass(brick)
    connectors = get_brick_connectors(brick)

    body = ET.Element("body", name=body_name)

    if include_freejoint:
        ET.SubElement(body, "freejoint", name=f"{body_name}_joint")

    # Explicit inertial — mass from formula, pos at geometric center
    ET.SubElement(
        body,
        "inertial",
        pos=_pos_str(0, 0, brick.shell_half_z),
        mass=_fmt(mass),
        diaginertia=_compute_box_inertia_str(mass, brick),
    )

    # --- Shell collision geoms (hollow: 4 walls + top plate) ---
    # Real LEGO bricks are open on the bottom; studs from below enter through
    # the open bottom to engage tubes inside. Using 5 thin box geoms instead
    # of a single solid box preserves this critical property.
    hx, hy, hz = brick.shell_half_x, brick.shell_half_y, brick.shell_half_z
    wt = WALL_THICKNESS
    tt = TOP_THICKNESS
    hw = wt / 2.0  # half wall thickness
    ht = tt / 2.0  # half top thickness

    # Top plate
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_shell_top",
        type="box",
        attrib={"class": "lego/brick_surface"},
        size=_pos_str(hx, hy, ht),
        pos=_pos_str(0, 0, brick.height - ht),
        group=str(COLLISION_GROUP),
    )
    # Left wall (−X)
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_shell_lx",
        type="box",
        attrib={"class": "lego/brick_surface"},
        size=_pos_str(hw, hy, hz),
        pos=_pos_str(-hx + hw, 0, hz),
        group=str(COLLISION_GROUP),
    )
    # Right wall (+X)
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_shell_rx",
        type="box",
        attrib={"class": "lego/brick_surface"},
        size=_pos_str(hw, hy, hz),
        pos=_pos_str(hx - hw, 0, hz),
        group=str(COLLISION_GROUP),
    )
    # Front wall (+Y) — inner extent excludes corners (already covered by LX/RX)
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_shell_fy",
        type="box",
        attrib={"class": "lego/brick_surface"},
        size=_pos_str(hx - wt, hw, hz),
        pos=_pos_str(0, hy - hw, hz),
        group=str(COLLISION_GROUP),
    )
    # Back wall (−Y)
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_shell_by",
        type="box",
        attrib={"class": "lego/brick_surface"},
        size=_pos_str(hx - wt, hw, hz),
        pos=_pos_str(0, -hy + hw, hz),
        group=str(COLLISION_GROUP),
    )

    # --- Shell visual geom ---
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_shell_vis",
        type="box",
        size=_pos_str(brick.shell_half_x, brick.shell_half_y, brick.shell_half_z),
        pos=_pos_str(0, 0, brick.shell_half_z),
        group=str(VISUAL_GROUP),
        contype="0",
        conaffinity="0",
        rgba=_rgba_str(color),
    )

    # --- Stud collision + visual geoms ---
    for cp in connectors.studs:
        ET.SubElement(
            body,
            "geom",
            name=f"{body_name}_{cp.id}",
            type="cylinder",
            attrib={"class": "lego/stud"},
            size=f"{_fmt(STUD_COLLISION_RADIUS)} {_fmt(STUD_HALF_HEIGHT)}",
            pos=_pos_str(*cp.position),
            group=str(COLLISION_GROUP),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{body_name}_{cp.id}_vis",
            type="cylinder",
            size=f"{_fmt(STUD_VISUAL_RADIUS)} {_fmt(STUD_HALF_HEIGHT)}",
            pos=_pos_str(*cp.position),
            group=str(VISUAL_GROUP),
            contype="0",
            conaffinity="0",
            rgba=_rgba_str(color),
        )

    # --- Tube capsule geoms (collision only) ---
    for cp in connectors.tubes:
        tx, ty, tz = cp.position
        for ci in range(TUBE_CAPSULE_COUNT):
            angle = ci * (2.0 * math.pi / TUBE_CAPSULE_COUNT)
            cx = tx + TUBE_RING_RADIUS * math.cos(angle)
            cy = ty + TUBE_RING_RADIUS * math.sin(angle)
            # Capsule is vertical: fromto defines the two endpoints
            z_lo = tz - TUBE_CAPSULE_HALF_HEIGHT
            z_hi = tz + TUBE_CAPSULE_HALF_HEIGHT
            ET.SubElement(
                body,
                "geom",
                name=f"{body_name}_{cp.id}_c{ci}",
                type="capsule",
                attrib={"class": "lego/tube"},
                size=_fmt(TUBE_CAPSULE_RADIUS),
                fromto=f"{_fmt(cx)} {_fmt(cy)} {_fmt(z_lo)} {_fmt(cx)} {_fmt(cy)} {_fmt(z_hi)}",
                group=str(COLLISION_GROUP),
            )

    return _indent_xml(body)


def generate_brick_mjcf(
    brick: BrickType,
    name_prefix: str = "brick",
    color: tuple[float, float, float, float] = DEFAULT_BRICK_COLOR,
    include_freejoint: bool = True,
) -> str:
    """Generate a complete standalone MJCF document for a LEGO brick.

    Includes compiler settings, default classes, and the brick body in a
    worldbody. Suitable for independent loading and testing.

    Args:
        brick: BrickType specification.
        name_prefix: Prefix for body and geom names.
        color: RGBA tuple for visual geoms.
        include_freejoint: Whether to add a freejoint.

    Returns:
        Complete MJCF XML string.
    """
    root = ET.Element("mujoco", model=f"{name_prefix}_{brick.name}")

    # Compiler
    ET.SubElement(root, "compiler", inertiafromgeom="auto", angle="radian")

    # Option: solver settings from spec
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="pyramidal",
        solver="Newton",
        iterations="80",
        ls_iterations="10",
    )

    # Inline default classes (so from_xml_string works without file context)
    add_lego_defaults(root)

    # Worldbody with floor and brick
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        diffuse="0.8 0.8 0.8",
        pos="0 0 1",
        dir="0 0 -1",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="1 1 0.1",
        rgba="0.9 0.9 0.9 1",
        contype="1",
        conaffinity="1",
    )

    # Parse and append the brick body element
    body_xml = generate_brick_body_xml(brick, name_prefix, color, include_freejoint)
    brick_body = ET.fromstring(body_xml)
    # Position the brick above the floor
    brick_body.set("pos", _pos_str(0, 0, 0.05))
    worldbody.append(brick_body)

    return _indent_xml(root)


def write_brick_assets(
    output_dir: Path | None = None,
    brick_types: dict[str, BrickType] | None = None,
) -> list[Path]:
    """Write MJCF files for all brick types to disk.

    Args:
        output_dir: Directory to write brick files. Defaults to
            ``sim/assets/lego/bricks/``.
        brick_types: Dict of brick types to generate. Defaults to all
            standard types from ``BRICK_TYPES``.

    Returns:
        List of paths to generated files.
    """
    if output_dir is None:
        output_dir = _BRICKS_DIR
    if brick_types is None:
        brick_types = BRICK_TYPES

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for name, brick in brick_types.items():
        xml = generate_brick_mjcf(brick)
        path = output_dir / f"brick_{name}.xml"
        path.write_text(xml, encoding="utf-8")
        paths.append(path)

    return paths


def add_lego_defaults(root: ET.Element) -> None:
    """Add the LEGO contact material default classes inline."""
    default = ET.SubElement(root, "default")

    stud = ET.SubElement(default, "default", attrib={"class": "lego/stud"})
    ET.SubElement(
        stud,
        "geom",
        contype="6",
        conaffinity="7",
        solref="0.003 1.0",
        solimp="0.97 0.995 0.001 0.5 4",
        friction="0.65 0.005 0.005",
        condim="4",
    )

    tube = ET.SubElement(default, "default", attrib={"class": "lego/tube"})
    ET.SubElement(
        tube,
        "geom",
        contype="4",
        conaffinity="4",
        solref="0.003 1.0",
        solimp="0.97 0.995 0.001 0.5 4",
        friction="0.65 0.005 0.005",
        condim="4",
    )

    bs = ET.SubElement(default, "default", attrib={"class": "lego/brick_surface"})
    ET.SubElement(
        bs,
        "geom",
        contype="2",
        conaffinity="3",
        solref="0.005 1.0",
        solimp="0.9 0.95 0.001 0.5 2",
        friction="0.4 0.005 0.002",
        condim="3",
    )

    bp = ET.SubElement(default, "default", attrib={"class": "lego/baseplate"})
    ET.SubElement(
        bp,
        "geom",
        contype="2",
        conaffinity="3",
        solref="0.005 1.0",
        solimp="0.9 0.95 0.001 0.5 2",
        friction="0.6 0.01 0.005",
        condim="3",
    )


def _compute_box_inertia_str(mass: float, brick: BrickType) -> str:
    """Compute diagonal inertia of a solid box (shell approximation)."""
    hx, hy, hz = brick.shell_half_x, brick.shell_half_y, brick.shell_half_z
    sx, sy, sz = 2 * hx, 2 * hy, 2 * hz
    ixx = mass / 12.0 * (sy**2 + sz**2)
    iyy = mass / 12.0 * (sx**2 + sz**2)
    izz = mass / 12.0 * (sx**2 + sy**2)
    return f"{_fmt(ixx, 10)} {_fmt(iyy, 10)} {_fmt(izz, 10)}"


def _indent_xml(elem: ET.Element, level: int = 0) -> str:
    """Pretty-print an ElementTree element to a string."""
    ET.indent(elem, space="    ", level=level)
    return ET.tostring(elem, encoding="unicode", xml_declaration=False)
