"""Procedural MJCF generation for LEGO baseplates (Phase 1.2.3).

Generates MuJoCo-compatible XML for parametric LEGO baseplates with:
- Solid plate surface (lego/baseplate contact class)
- Stud grid on top (lego/stud contact class, same geometry as brick studs)
- Visual geometry (MuJoCo primitives)
- No tubes (baseplates are solid thin plates)

Usage:
    from sim.lego.baseplate_generator import generate_baseplate_mjcf, write_baseplate_assets
    from sim.lego.constants import BASEPLATE_TYPES

    xml = generate_baseplate_mjcf(BASEPLATE_TYPES["8x8"])
    write_baseplate_assets()  # writes to sim/assets/lego/baseplates/
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from sim.lego.brick_generator import add_lego_defaults
from sim.lego.connector import get_baseplate_connectors
from sim.lego.constants import (
    BASEPLATE_TYPES,
    COLLISION_GROUP,
    DEFAULT_BASEPLATE_COLOR,
    STUD_COLLISION_RADIUS,
    STUD_HALF_HEIGHT,
    STUD_VISUAL_RADIUS,
    VISUAL_GROUP,
    BaseplateType,
)
from sim.lego.mass import compute_baseplate_mass

_LEGO_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "lego"
_BASEPLATES_DIR = _LEGO_ASSETS_DIR / "baseplates"


def _fmt(v: float, precision: int = 6) -> str:
    """Format a float, stripping trailing zeros."""
    return f"{v:.{precision}f}".rstrip("0").rstrip(".")


def _rgba_str(color: tuple[float, float, float, float]) -> str:
    return " ".join(_fmt(c, 3) for c in color)


def _pos_str(x: float, y: float, z: float) -> str:
    return f"{_fmt(x)} {_fmt(y)} {_fmt(z)}"


def _indent_xml(elem: ET.Element, level: int = 0) -> str:
    """Pretty-print an ElementTree element to a string."""
    ET.indent(elem, space="    ", level=level)
    return ET.tostring(elem, encoding="unicode", xml_declaration=False)


def generate_baseplate_body_xml(
    baseplate: BaseplateType,
    name_prefix: str = "baseplate",
    color: tuple[float, float, float, float] = DEFAULT_BASEPLATE_COLOR,
    include_freejoint: bool = False,
) -> str:
    """Generate an MJCF ``<body>`` element for a LEGO baseplate.

    The body includes a solid plate surface and stud grid on top.
    No tubes (baseplates are solid). Origin at center of bottom face (Z=0).

    Args:
        baseplate: BaseplateType specification.
        name_prefix: Prefix for body and geom names.
        color: RGBA tuple for visual geoms.
        include_freejoint: Whether to add a freejoint (False for fixed baseplates).

    Returns:
        XML string for the ``<body>`` element.
    """
    body_name = f"{name_prefix}_{baseplate.name}"
    mass = compute_baseplate_mass(baseplate)
    connectors = get_baseplate_connectors(baseplate)

    body = ET.Element("body", name=body_name)

    if include_freejoint:
        ET.SubElement(body, "freejoint", name=f"{body_name}_joint")

    # Explicit inertial
    ET.SubElement(
        body,
        "inertial",
        pos=_pos_str(0, 0, baseplate.half_z),
        mass=_fmt(mass),
        diaginertia=_compute_plate_inertia_str(mass, baseplate),
    )

    hx, hy, hz = baseplate.half_x, baseplate.half_y, baseplate.half_z

    # --- Surface collision geom (solid plate, lego/baseplate class) ---
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_surface",
        type="box",
        attrib={"class": "lego/baseplate"},
        size=_pos_str(hx, hy, hz),
        pos=_pos_str(0, 0, hz),
        group=str(COLLISION_GROUP),
    )

    # --- Surface visual geom ---
    ET.SubElement(
        body,
        "geom",
        name=f"{body_name}_surface_vis",
        type="box",
        size=_pos_str(hx, hy, hz),
        pos=_pos_str(0, 0, hz),
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

    return _indent_xml(body)


def generate_baseplate_mjcf(
    baseplate: BaseplateType,
    name_prefix: str = "baseplate",
    color: tuple[float, float, float, float] = DEFAULT_BASEPLATE_COLOR,
) -> str:
    """Generate a complete standalone MJCF document for a LEGO baseplate.

    Includes compiler settings, default classes, and the baseplate body.
    The baseplate is fixed (no freejoint). Suitable for independent loading.

    Args:
        baseplate: BaseplateType specification.
        name_prefix: Prefix for body and geom names.
        color: RGBA tuple for visual geoms.

    Returns:
        Complete MJCF XML string.
    """
    root = ET.Element("mujoco", model=f"{name_prefix}_{baseplate.name}")

    ET.SubElement(root, "compiler", inertiafromgeom="auto", angle="radian")

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

    add_lego_defaults(root)

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

    # Baseplate body (fixed, no freejoint) at floor level
    body_xml = generate_baseplate_body_xml(baseplate, name_prefix, color, include_freejoint=False)
    baseplate_body = ET.fromstring(body_xml)
    baseplate_body.set("pos", _pos_str(0, 0, 0))
    worldbody.append(baseplate_body)

    return _indent_xml(root)


def write_baseplate_assets(
    output_dir: Path | None = None,
    baseplate_types: dict[str, BaseplateType] | None = None,
) -> list[Path]:
    """Write MJCF files for all baseplate types to disk.

    Args:
        output_dir: Directory to write baseplate files. Defaults to
            ``sim/assets/lego/baseplates/``.
        baseplate_types: Dict of baseplate types to generate. Defaults to all
            standard types from ``BASEPLATE_TYPES``.

    Returns:
        List of paths to generated files.
    """
    if output_dir is None:
        output_dir = _BASEPLATES_DIR
    if baseplate_types is None:
        baseplate_types = BASEPLATE_TYPES

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for name, baseplate in baseplate_types.items():
        xml = generate_baseplate_mjcf(baseplate)
        path = output_dir / f"baseplate_{name}.xml"
        path.write_text(xml, encoding="utf-8")
        paths.append(path)

    return paths


def _compute_plate_inertia_str(mass: float, baseplate: BaseplateType) -> str:
    """Compute diagonal inertia of a solid flat plate."""
    sx = 2 * baseplate.half_x
    sy = 2 * baseplate.half_y
    sz = 2 * baseplate.half_z
    ixx = mass / 12.0 * (sy**2 + sz**2)
    iyy = mass / 12.0 * (sx**2 + sz**2)
    izz = mass / 12.0 * (sx**2 + sy**2)
    return f"{_fmt(ixx, 10)} {_fmt(iyy, 10)} {_fmt(izz, 10)}"
