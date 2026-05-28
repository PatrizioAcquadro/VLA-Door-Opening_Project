"""Asset loading contract for MuJoCo MJCF scenes (Phase 0.2.5).

Provides a single entrypoint for loading scenes with validated asset paths.
All MJCF loading should go through this module.

Usage:
    from sim.asset_loader import load_scene, ASSETS_DIR

    model = load_scene("test_scene")       # loads sim/assets/scenes/test_scene.xml
    model = load_scene("test_scene.xml")   # also works
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco

ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"
SCENES_DIR: Path = ASSETS_DIR / "scenes"
ROBOTS_DIR: Path = ASSETS_DIR / "robots"
LEGO_DIR: Path = ASSETS_DIR / "lego"
LEGO_BRICKS_DIR: Path = LEGO_DIR / "bricks"
LEGO_BASEPLATES_DIR: Path = LEGO_DIR / "baseplates"


def resolve_scene_path(scene_name: str) -> Path:
    """Resolve a scene name to an absolute MJCF path.

    Looks in ``sim/assets/scenes/`` for a matching XML file.

    Args:
        scene_name: Scene name with or without ``.xml`` extension.

    Returns:
        Absolute path to the MJCF XML file.

    Raises:
        FileNotFoundError: If no matching scene is found.
    """
    if not scene_name.endswith(".xml"):
        scene_name = scene_name + ".xml"

    path = SCENES_DIR / scene_name
    if not path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_name} (looked in {SCENES_DIR})")
    return path.resolve()


def resolve_robot_path(robot_name: str) -> Path:
    """Resolve a robot name to its main MJCF path.

    Expects the layout ``sim/assets/robots/<robot_name>/<robot_name>.xml``.

    Args:
        robot_name: Robot directory name under ``sim/assets/robots/``.

    Returns:
        Absolute path to the robot MJCF XML file.

    Raises:
        FileNotFoundError: If robot directory or XML does not exist.
    """
    robot_dir = ROBOTS_DIR / robot_name
    if not robot_dir.is_dir():
        raise FileNotFoundError(f"Robot directory not found: {robot_name} (looked in {ROBOTS_DIR})")

    xml_path = robot_dir / f"{robot_name}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Robot MJCF not found: {xml_path} " f"(expected <robot_name>/<robot_name>.xml)"
        )
    return xml_path.resolve()


def resolve_lego_brick_path(brick_name: str) -> Path:
    """Resolve a brick name to its MJCF path.

    Args:
        brick_name: Brick name, e.g., ``"2x4"`` or ``"brick_2x4"``.

    Returns:
        Absolute path to the brick MJCF file.

    Raises:
        FileNotFoundError: If brick file does not exist.
    """
    if not brick_name.startswith("brick_"):
        brick_name = f"brick_{brick_name}"
    if not brick_name.endswith(".xml"):
        brick_name = f"{brick_name}.xml"

    path = LEGO_BRICKS_DIR / brick_name
    if not path.exists():
        raise FileNotFoundError(f"LEGO brick not found: {brick_name} (looked in {LEGO_BRICKS_DIR})")
    return path.resolve()


def resolve_lego_baseplate_path(baseplate_name: str) -> Path:
    """Resolve a baseplate name to its MJCF path.

    Args:
        baseplate_name: Baseplate name, e.g., ``"8x8"`` or ``"baseplate_8x8"``.

    Returns:
        Absolute path to the baseplate MJCF file.

    Raises:
        FileNotFoundError: If baseplate file does not exist.
    """
    if not baseplate_name.startswith("baseplate_"):
        baseplate_name = f"baseplate_{baseplate_name}"
    if not baseplate_name.endswith(".xml"):
        baseplate_name = f"{baseplate_name}.xml"

    path = LEGO_BASEPLATES_DIR / baseplate_name
    if not path.exists():
        raise FileNotFoundError(
            f"LEGO baseplate not found: {baseplate_name} (looked in {LEGO_BASEPLATES_DIR})"
        )
    return path.resolve()


def load_scene(scene_name: str) -> mujoco.MjModel:
    """Load an MJCF scene by name.

    This is the primary entrypoint for loading simulation scenes.
    Resolves the scene path, then delegates to ``mujoco_env.load_model()``.

    Args:
        scene_name: Scene name (e.g., ``"test_scene"``).

    Returns:
        Compiled MuJoCo model.

    Raises:
        FileNotFoundError: If the scene file is not found.
        mujoco.FatalError: If the MJCF fails to parse.
    """
    from sim.mujoco_env import load_model

    path = resolve_scene_path(scene_name)
    return load_model(path)
