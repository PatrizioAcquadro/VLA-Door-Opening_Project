"""Scene builder for contact physics tests (Phase 1.2.2+).

Programmatically generates MJCF XML for insertion test scenarios by composing
brick and baseplate bodies.

Usage:
    from sim.lego.contact_scene import load_insertion_scene, check_stud_engagement
    from sim.lego.constants import BRICK_TYPES, BASEPLATE_TYPES

    model, data = load_insertion_scene(BRICK_TYPES["2x2"], BRICK_TYPES["2x2"])
    model, data = load_baseplate_insertion_scene(BASEPLATE_TYPES["8x8"], BRICK_TYPES["2x2"])
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from sim.lego.baseplate_generator import generate_baseplate_body_xml
from sim.lego.brick_generator import add_lego_defaults, generate_brick_body_xml
from sim.lego.constants import (
    BRICK_HEIGHT,
    STUD_HALF_HEIGHT,
    STUD_HEIGHT,
    BaseplateType,
    BrickType,
)

# Default weld constraint parameters for hybrid retention mode
_WELD_SOLREF = "0.01 1.0"
_WELD_SOLIMP = "0.95 0.99 0.001 0.5 2"

# Default height of base brick above floor (meters)
DEFAULT_BASE_HEIGHT: float = 0.05


def _fmt(v: float, precision: int = 6) -> str:
    """Format a float, stripping trailing zeros."""
    return f"{v:.{precision}f}".rstrip("0").rstrip(".")


def _pos_str(x: float, y: float, z: float) -> str:
    return f"{_fmt(x)} {_fmt(y)} {_fmt(z)}"


def _axis_angle_to_quat(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    """Convert axis-angle to MuJoCo quaternion [w, x, y, z]."""
    ax = np.array(axis, dtype=np.float64)
    norm = np.linalg.norm(ax)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    ax = ax / norm
    half = angle_rad / 2.0
    w = math.cos(half)
    s = math.sin(half)
    return (w, ax[0] * s, ax[1] * s, ax[2] * s)


def _quat_str(q: tuple[float, float, float, float]) -> str:
    return " ".join(_fmt(v, 8) for v in q)


def _add_option(root: ET.Element) -> None:
    """Add solver option with energy flag to MJCF root."""
    option = ET.SubElement(
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
    ET.SubElement(option, "flag", energy="enable")


def _indent_xml(elem: ET.Element) -> str:
    """Pretty-print an ElementTree element to a string."""
    ET.indent(elem, space="    ", level=0)
    return ET.tostring(elem, encoding="unicode", xml_declaration=False)


def generate_insertion_scene(
    base_brick: BrickType,
    top_brick: BrickType,
    lateral_offset: tuple[float, float] = (0.0, 0.0),
    angular_tilt_deg: float = 0.0,
    tilt_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    approach_gap: float = 0.002,
    base_height: float = DEFAULT_BASE_HEIGHT,
    base_name: str = "base",
    top_name: str = "top",
    retention_mode: str = "physics",
) -> str:
    """Generate MJCF XML for a two-brick insertion test scene.

    The base brick is fixed (no freejoint) with studs pointing up.
    The top brick has a freejoint and is positioned above the base,
    ready for downward insertion (tubes facing down onto base studs).

    Args:
        base_brick: BrickType for the fixed base brick.
        top_brick: BrickType for the free top brick.
        lateral_offset: (dx, dy) offset of top brick from aligned position (meters).
        angular_tilt_deg: Tilt angle of top brick from vertical (degrees).
        tilt_axis: Axis for angular tilt (default: tilt around X axis).
        approach_gap: Vertical clearance between stud tops and top brick bottom (meters).
        base_height: Height of base brick origin above floor (meters).
        base_name: Name prefix for base brick geoms.
        top_name: Name prefix for top brick geoms.
        retention_mode: ``"physics"`` (default) or ``"spec_proxy"`` (adds inactive
            weld constraint for hybrid retention).

    Returns:
        Complete MJCF XML string.
    """
    root = ET.Element("mujoco", model="insertion_test")

    ET.SubElement(root, "compiler", inertiafromgeom="auto", angle="radian")
    _add_option(root)
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

    # Base brick: fixed (no freejoint), positioned at base_height
    base_body_xml = generate_brick_body_xml(
        base_brick, name_prefix=base_name, include_freejoint=False
    )
    base_body = ET.fromstring(base_body_xml)
    base_body.set("pos", _pos_str(0, 0, base_height))
    worldbody.append(base_body)

    # Top brick: free (with freejoint), positioned above base studs
    top_z = base_height + BRICK_HEIGHT + STUD_HEIGHT + approach_gap
    top_body_xml = generate_brick_body_xml(top_brick, name_prefix=top_name, include_freejoint=True)
    top_body = ET.fromstring(top_body_xml)
    top_body.set("pos", _pos_str(lateral_offset[0], lateral_offset[1], top_z))

    # Apply angular tilt if specified
    if abs(angular_tilt_deg) > 1e-6:
        angle_rad = math.radians(angular_tilt_deg)
        quat = _axis_angle_to_quat(tilt_axis, angle_rad)
        top_body.set("quat", _quat_str(quat))

    worldbody.append(top_body)

    # Add pre-declared (inactive) weld constraint for hybrid retention
    if retention_mode == "spec_proxy":
        base_body_name = f"{base_name}_{base_brick.name}"
        top_body_name = f"{top_name}_{top_brick.name}"
        expected_z = BRICK_HEIGHT + STUD_HEIGHT
        equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name=f"snap_{base_name}_{top_name}",
            body1=base_body_name,
            body2=top_body_name,
            active="false",
            relpose=f"0 0 {_fmt(expected_z)} 1 0 0 0",
            solref=_WELD_SOLREF,
            solimp=_WELD_SOLIMP,
        )

    return _indent_xml(root)


def generate_stack_scene(
    brick_types: list[BrickType],
    base_height: float = DEFAULT_BASE_HEIGHT,
    approach_gap: float = 0.0005,
    retention_mode: str = "physics",
) -> str:
    """Generate MJCF XML for multi-brick stacking scene.

    The first brick is fixed. Subsequent bricks have freejoints and are
    positioned in a vertical stack with small gaps (pre-inserted positions).

    Args:
        brick_types: List of BrickType for each brick in the stack (bottom to top).
        base_height: Height of bottom brick origin above floor (meters).
        approach_gap: Small gap between stacked bricks for initial settling.
        retention_mode: ``"physics"`` (default) or ``"spec_proxy"`` (adds inactive
            weld constraints for hybrid retention).

    Returns:
        Complete MJCF XML string.
    """
    if len(brick_types) < 2:
        raise ValueError("Stack scene requires at least 2 bricks")

    root = ET.Element("mujoco", model="stack_test")

    ET.SubElement(root, "compiler", inertiafromgeom="auto", angle="radian")
    _add_option(root)
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

    # Bottom brick: fixed
    current_z = base_height
    bottom_xml = generate_brick_body_xml(
        brick_types[0], name_prefix="stack_0", include_freejoint=False
    )
    bottom_body = ET.fromstring(bottom_xml)
    bottom_body.set("pos", _pos_str(0, 0, current_z))
    worldbody.append(bottom_body)

    # Subsequent bricks: free, positioned in stack
    for i, brick in enumerate(brick_types[1:], start=1):
        current_z += BRICK_HEIGHT + STUD_HEIGHT + approach_gap
        body_xml = generate_brick_body_xml(brick, name_prefix=f"stack_{i}", include_freejoint=True)
        body = ET.fromstring(body_xml)
        body.set("pos", _pos_str(0, 0, current_z))
        worldbody.append(body)

    # Add pre-declared (inactive) weld constraints for hybrid retention
    if retention_mode == "spec_proxy":
        equality = ET.SubElement(root, "equality")
        expected_z = BRICK_HEIGHT + STUD_HEIGHT
        for i in range(len(brick_types) - 1):
            body1_name = f"stack_{i}_{brick_types[i].name}"
            body2_name = f"stack_{i + 1}_{brick_types[i + 1].name}"
            ET.SubElement(
                equality,
                "weld",
                name=f"snap_stack_{i}_stack_{i + 1}",
                body1=body1_name,
                body2=body2_name,
                active="false",
                relpose=f"0 0 {_fmt(expected_z)} 1 0 0 0",
                solref=_WELD_SOLREF,
                solimp=_WELD_SOLIMP,
            )

    return _indent_xml(root)


def load_insertion_scene(
    base_brick: BrickType,
    top_brick: BrickType,
    **kwargs,
) -> tuple:
    """Generate and load a two-brick insertion scene.

    Args:
        base_brick: BrickType for the fixed base brick.
        top_brick: BrickType for the free top brick.
        **kwargs: Passed to ``generate_insertion_scene()``.

    Returns:
        Tuple of (mujoco.MjModel, mujoco.MjData).
    """
    import mujoco

    xml = generate_insertion_scene(base_brick, top_brick, **kwargs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def load_stack_scene(
    brick_types: list[BrickType],
    **kwargs,
) -> tuple:
    """Generate and load a multi-brick stack scene.

    Args:
        brick_types: List of BrickType for each brick in the stack.
        **kwargs: Passed to ``generate_stack_scene()``.

    Returns:
        Tuple of (mujoco.MjModel, mujoco.MjData).
    """
    import mujoco

    xml = generate_stack_scene(brick_types, **kwargs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@dataclass
class EngagementResult:
    """Result of checking stud engagement between two bricks."""

    engaged: bool
    top_body_z: float
    base_stud_top_z: float
    gap: float  # positive = not yet engaged, negative = engaged


def check_stud_engagement(
    model,
    data,
    base_brick: BrickType,
    base_name: str = "base",
    top_name: str = "top",
    base_height: float = DEFAULT_BASE_HEIGHT,
    base_surface_height: float | None = None,
    top_body_name_override: str | None = None,
) -> EngagementResult:
    """Check whether studs of base are engaged in tubes of top brick.

    Engagement is detected when the top brick's bottom face is at or below
    the stud tops of the base (studs have entered the tube region).

    Args:
        model: MuJoCo model.
        data: MuJoCo data (after stepping).
        base_brick: BrickType of the base brick (used for name resolution).
        base_name: Name prefix of base body.
        top_name: Name prefix of top brick body.
        base_height: World Z of base body origin.
        base_surface_height: Height of base surface. Defaults to ``BRICK_HEIGHT``;
            use ``baseplate.thickness`` for baseplate scenes.
        top_body_name_override: Override the top body name lookup (e.g., when
            the top brick type differs from the base).

    Returns:
        EngagementResult with engagement status and gap measurement.
    """
    import mujoco

    if base_surface_height is None:
        base_surface_height = BRICK_HEIGHT

    if top_body_name_override is not None:
        top_body_name = top_body_name_override
    else:
        top_body_name = f"{top_name}_{base_brick.name}"
    top_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, top_body_name)

    # Top brick bottom face Z in world frame
    top_body_z = data.xpos[top_body_id][2]

    # Base stud center Z (fixed, computed from base_height)
    base_stud_center_z = base_height + base_surface_height + STUD_HALF_HEIGHT

    # Gap: positive = top brick bottom above stud center (not engaged)
    # negative = top brick bottom below stud center (studs inside tubes)
    gap = top_body_z - base_stud_center_z

    # Engaged when top brick body Z is at or below the stud center
    engaged = gap <= 0

    return EngagementResult(
        engaged=engaged,
        top_body_z=top_body_z,
        base_stud_top_z=base_stud_center_z,
        gap=gap,
    )


def setup_connection_manager(
    model,
    data,
    brick_pairs: list[tuple[str, str]],
    **config_kwargs,
):
    """Create a ConnectionManager and register all pre-declared weld pairs.

    Finds weld equality constraints by matching body IDs against pre-declared
    weld constraints in the model (added by ``generate_*_scene(retention_mode="spec_proxy")``).

    Args:
        model: MuJoCo model with pre-declared weld constraints.
        data: MuJoCo data.
        brick_pairs: List of ``(body1_name, body2_name)`` tuples to register.
        **config_kwargs: Passed to ``ConnectionManager.__init__()``.

    Returns:
        Configured ``ConnectionManager`` with all pairs registered.

    Raises:
        ValueError: If a weld constraint for a pair is not found in the model.
    """
    import mujoco

    from sim.lego.connection_manager import ConnectionManager

    mgr = ConnectionManager(model, data, **config_kwargs)

    for body1_name, body2_name in brick_pairs:
        body1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

        # Find the weld constraint matching this body pair
        eq_id = -1
        for i in range(model.neq):
            # MuJoCo weld type = 1 (mjEQ_WELD)
            if model.eq_type[i] == 1:
                if model.eq_obj1id[i] == body1_id and model.eq_obj2id[i] == body2_id:
                    eq_id = i
                    break

        if eq_id < 0:
            raise ValueError(
                f"No weld constraint found for pair ({body1_name}, {body2_name}). "
                f"Did you use retention_mode='spec_proxy' when generating the scene?"
            )

        mgr.register_pair(body1_name, body2_name, eq_id)

    return mgr


# ---------------------------------------------------------------------------
# Baseplate scene generation (Phase 1.2.3)
# ---------------------------------------------------------------------------


def generate_baseplate_insertion_scene(
    baseplate: BaseplateType,
    top_brick: BrickType,
    placement_offset: tuple[float, float] = (0.0, 0.0),
    lateral_offset: tuple[float, float] = (0.0, 0.0),
    angular_tilt_deg: float = 0.0,
    tilt_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    approach_gap: float = 0.002,
    base_height: float = 0.0,
    base_name: str = "base",
    top_name: str = "top",
    retention_mode: str = "physics",
) -> str:
    """Generate MJCF XML for a brick-on-baseplate insertion test scene.

    The baseplate is fixed (no freejoint) with studs pointing up.
    The top brick has a freejoint and is positioned above the baseplate studs.

    Args:
        baseplate: BaseplateType for the fixed baseplate.
        top_brick: BrickType for the free top brick.
        placement_offset: (dx, dy) position of brick center relative to
            baseplate center (for targeting specific stud locations).
        lateral_offset: Additional (dx, dy) misalignment for insertion tests.
        angular_tilt_deg: Tilt angle of top brick from vertical.
        tilt_axis: Axis for angular tilt.
        approach_gap: Vertical clearance above stud tops (meters).
        base_height: Height of baseplate origin above floor (meters).
        base_name: Name prefix for baseplate body.
        top_name: Name prefix for top brick body.
        retention_mode: ``"physics"`` or ``"spec_proxy"``.

    Returns:
        Complete MJCF XML string.
    """
    root = ET.Element("mujoco", model="baseplate_insertion_test")

    ET.SubElement(root, "compiler", inertiafromgeom="auto", angle="radian")
    _add_option(root)
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

    # Baseplate: fixed (no freejoint)
    base_body_xml = generate_baseplate_body_xml(
        baseplate, name_prefix=base_name, include_freejoint=False
    )
    base_body = ET.fromstring(base_body_xml)
    base_body.set("pos", _pos_str(0, 0, base_height))
    worldbody.append(base_body)

    # Top brick: free (with freejoint), positioned above baseplate studs
    top_x = placement_offset[0] + lateral_offset[0]
    top_y = placement_offset[1] + lateral_offset[1]
    top_z = base_height + baseplate.thickness + STUD_HEIGHT + approach_gap
    top_body_xml = generate_brick_body_xml(top_brick, name_prefix=top_name, include_freejoint=True)
    top_body = ET.fromstring(top_body_xml)
    top_body.set("pos", _pos_str(top_x, top_y, top_z))

    if abs(angular_tilt_deg) > 1e-6:
        angle_rad = math.radians(angular_tilt_deg)
        quat = _axis_angle_to_quat(tilt_axis, angle_rad)
        top_body.set("quat", _quat_str(quat))

    worldbody.append(top_body)

    # Pre-declared weld for hybrid retention
    if retention_mode == "spec_proxy":
        base_body_name = f"{base_name}_{baseplate.name}"
        top_body_name = f"{top_name}_{top_brick.name}"
        expected_z = baseplate.thickness + STUD_HEIGHT
        equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name=f"snap_{base_name}_{top_name}",
            body1=base_body_name,
            body2=top_body_name,
            active="false",
            relpose=f"0 0 {_fmt(expected_z)} 1 0 0 0",
            solref=_WELD_SOLREF,
            solimp=_WELD_SOLIMP,
        )

    return _indent_xml(root)


def load_baseplate_insertion_scene(
    baseplate: BaseplateType,
    top_brick: BrickType,
    **kwargs,
) -> tuple:
    """Generate and load a brick-on-baseplate insertion scene.

    Args:
        baseplate: BaseplateType for the fixed baseplate.
        top_brick: BrickType for the free top brick.
        **kwargs: Passed to ``generate_baseplate_insertion_scene()``.

    Returns:
        Tuple of (mujoco.MjModel, mujoco.MjData).
    """
    import mujoco

    xml = generate_baseplate_insertion_scene(baseplate, top_brick, **kwargs)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def generate_workspace_scene(
    baseplate: BaseplateType,
    table_pos: tuple[float, float, float] = (0.45, 0.0, 0.75),
    table_size: tuple[float, float, float] = (0.25, 0.30, 0.02),
    baseplate_offset: tuple[float, float] = (0.0, 0.0),
) -> str:
    """Generate MJCF XML for the Alex LEGO workspace scene.

    Includes Alex robot (via include), table, baseplate on table, and cameras.

    Args:
        baseplate: BaseplateType for the baseplate on the table.
        table_pos: (x, y, z) center of table surface in world frame.
        table_size: (half_x, half_y, half_z) box extents of table.
        baseplate_offset: (dx, dy) offset of baseplate center on table.

    Returns:
        Complete MJCF XML string.
    """
    root = ET.Element("mujoco", model="alex_lego_workspace")

    ET.SubElement(root, "compiler", meshdir="../robots/alex/", balanceinertia="true")
    ET.SubElement(root, "include", file="../robots/alex/alex.xml")

    _add_option(root)
    add_lego_defaults(root)

    ET.SubElement(root, "statistic", center="0 0 1.0", extent="1.5")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.6 0.6 0.6",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(
        visual,
        "global",
        azimuth="120",
        elevation="-20",
        offwidth="1280",
        offheight="720",
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="5 5",
    )
    ET.SubElement(asset, "material", name="table_wood", rgba="0.6 0.5 0.4 1")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 3",
        dir="0 0 -1",
        directional="true",
    )

    # Cameras
    ET.SubElement(
        worldbody,
        "camera",
        name="overhead",
        pos="0 -2 2",
        xyaxes="1 0 0 0 0.7 0.7",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="third_person",
        pos="2.0 -1.3 1.7",
        xyaxes="0.55 0.83 0 -0.25 0.17 0.95",
        fovy="60",
    )
    bp_x = table_pos[0] + baseplate_offset[0]
    bp_y = table_pos[1] + baseplate_offset[1]
    ET.SubElement(
        worldbody,
        "camera",
        name="workspace_closeup",
        pos=_pos_str(bp_x + 0.15, bp_y - 0.25, table_pos[2] + 0.30),
        xyaxes="0.85 0.53 0 -0.22 0.35 0.91",
        fovy="50",
    )

    # Floor
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        material="groundplane",
        contype="1",
        conaffinity="1",
        priority="1",
        friction="0.8",
        condim="3",
    )

    # Table
    table_body = ET.SubElement(
        worldbody,
        "body",
        name="table",
        pos=_pos_str(*table_pos),
    )
    ET.SubElement(
        table_body,
        "geom",
        type="box",
        size=_pos_str(*table_size),
        material="table_wood",
        contype="1",
        conaffinity="1",
        friction="0.6",
        mass="10",
    )

    # Baseplate on table (fixed, no freejoint)
    bp_z = table_pos[2] + table_size[2]  # on top of table surface
    bp_body_xml = generate_baseplate_body_xml(
        baseplate, name_prefix="baseplate", include_freejoint=False
    )
    bp_body = ET.fromstring(bp_body_xml)
    bp_body.set("pos", _pos_str(bp_x, bp_y, bp_z))
    worldbody.append(bp_body)

    # Keyframes (23 joints: 15 arm + 4 left EZGripper + 4 right EZGripper)
    # 17 actuators (15 arm + 2 EZGripper)
    keyframe = ET.SubElement(root, "keyframe")
    qpos_zeros = " ".join(["0"] * 23)
    ctrl_zeros = " ".join(["0"] * 17)
    ET.SubElement(
        keyframe,
        "key",
        name="home",
        qpos=qpos_zeros,
        ctrl=ctrl_zeros,
    )

    return _indent_xml(root)


def generate_episode_scene(
    baseplate: BaseplateType,
    brick_types: list[BrickType],
    table_pos: tuple[float, float, float] = (0.45, 0.0, 0.75),
    table_size: tuple[float, float, float] = (0.25, 0.30, 0.02),
    baseplate_offset: tuple[float, float] = (0.0, 0.0),
    retention_mode: str = "physics",
) -> str:
    """Generate MJCF XML for an episode scene with pre-declared brick slots.

    Extends the workspace scene (Alex robot + table + baseplate + cameras) with
    free brick bodies placed at a park position (Z=-10), ready to be repositioned
    at episode reset time via qpos manipulation.

    Args:
        baseplate: BaseplateType for the fixed baseplate on the table.
        brick_types: List of BrickType for each brick slot. Each slot gets
            one body with a freejoint, initially parked at (0, 0, -10).
            Length determines the number of brick slots (max_bricks).
        table_pos: (x, y, z) center of table surface in world frame.
        table_size: (half_x, half_y, half_z) box extents of table.
        baseplate_offset: (dx, dy) offset of baseplate center on table.
        retention_mode: ``"physics"`` (default) or ``"spec_proxy"`` (adds
            inactive weld constraints for baseplate-brick pairs).

    Returns:
        Complete MJCF XML string. Parse with ``mujoco.MjModel.from_xml_string()``.
    """
    root = ET.Element("mujoco", model="alex_lego_episode")

    ET.SubElement(root, "compiler", meshdir="../robots/alex/", balanceinertia="true")
    ET.SubElement(root, "include", file="../robots/alex/alex.xml")

    _add_option(root)
    add_lego_defaults(root)

    ET.SubElement(root, "statistic", center="0 0 1.0", extent="1.5")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.6 0.6 0.6",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(
        visual,
        "global",
        azimuth="120",
        elevation="-20",
        offwidth="1280",
        offheight="720",
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="5 5",
    )
    ET.SubElement(asset, "material", name="table_wood", rgba="0.6 0.5 0.4 1")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1", directional="true")

    # Cameras (same as workspace scene)
    ET.SubElement(
        worldbody,
        "camera",
        name="overhead",
        pos="0 -2 2",
        xyaxes="1 0 0 0 0.7 0.7",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="third_person",
        pos="2.0 -1.3 1.7",
        xyaxes="0.55 0.83 0 -0.25 0.17 0.95",
        fovy="60",
    )
    bp_x = table_pos[0] + baseplate_offset[0]
    bp_y = table_pos[1] + baseplate_offset[1]
    ET.SubElement(
        worldbody,
        "camera",
        name="workspace_closeup",
        pos=_pos_str(bp_x + 0.15, bp_y - 0.25, table_pos[2] + 0.30),
        xyaxes="0.85 0.53 0 -0.22 0.35 0.91",
        fovy="50",
    )

    # Floor
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.05",
        material="groundplane",
        contype="1",
        conaffinity="1",
        priority="1",
        friction="0.8",
        condim="3",
    )

    # Table
    table_body = ET.SubElement(worldbody, "body", name="table", pos=_pos_str(*table_pos))
    ET.SubElement(
        table_body,
        "geom",
        type="box",
        size=_pos_str(*table_size),
        material="table_wood",
        contype="1",
        conaffinity="1",
        friction="0.6",
        mass="10",
    )

    # Baseplate on table (fixed, no freejoint)
    bp_z = table_pos[2] + table_size[2]
    bp_body_xml = generate_baseplate_body_xml(
        baseplate, name_prefix="baseplate", include_freejoint=False
    )
    bp_body = ET.fromstring(bp_body_xml)
    bp_body.set("pos", _pos_str(bp_x, bp_y, bp_z))
    worldbody.append(bp_body)

    # Brick slots: free bodies parked below floor (Z=-10)
    # Body name: brick_{i}_{bt.name}, joint: brick_{i}_{bt.name}_joint
    _PARK_Z = -10.0
    for i, bt in enumerate(brick_types):
        brick_xml = generate_brick_body_xml(bt, name_prefix=f"brick_{i}", include_freejoint=True)
        brick_body = ET.fromstring(brick_xml)
        brick_body.set("pos", _pos_str(0.0, 0.0, _PARK_Z))
        worldbody.append(brick_body)

    # Pre-declared weld constraints (spec_proxy only) — baseplate-brick pairs
    if retention_mode == "spec_proxy" and brick_types:
        baseplate_body_name = f"baseplate_{baseplate.name}"
        equality = ET.SubElement(root, "equality")
        for i, bt in enumerate(brick_types):
            brick_body_name = f"brick_{i}_{bt.name}"
            expected_z = baseplate.thickness + STUD_HEIGHT
            ET.SubElement(
                equality,
                "weld",
                name=f"snap_bp_brick_{i}",
                body1=baseplate_body_name,
                body2=brick_body_name,
                active="false",
                relpose=f"0 0 {_fmt(expected_z)} 1 0 0 0",
                solref=_WELD_SOLREF,
                solimp=_WELD_SOLIMP,
            )

    # Keyframe: robot at home (23 joints = 0), all brick freejoints parked
    keyframe = ET.SubElement(root, "keyframe")
    qpos_robot = " ".join(["0"] * 23)
    ctrl_zeros = " ".join(["0"] * 17)
    if brick_types:
        qpos_bricks = " ".join([f"0 0 {_fmt(_PARK_Z)} 1 0 0 0"] * len(brick_types))
        qpos_all = f"{qpos_robot} {qpos_bricks}"
    else:
        qpos_all = qpos_robot
    ET.SubElement(keyframe, "key", name="home", qpos=qpos_all, ctrl=ctrl_zeros)

    return _indent_xml(root)
