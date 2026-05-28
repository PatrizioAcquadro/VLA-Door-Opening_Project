"""Asset loading contract for MuJoCo MJCF scenes.

Provides a single entrypoint for loading scenes with validated asset paths.
All MJCF loading should go through this module.

Usage:
    from sim.asset_loader import load_scene, ASSETS_DIR

    model = load_scene("alex_door_workspace")     # loads sim/assets/scenes/alex_door_workspace.xml
    model = load_scene("alex_door_workspace.xml") # also works
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco

ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"
SCENES_DIR: Path = ASSETS_DIR / "scenes"
ROBOTS_DIR: Path = ASSETS_DIR / "robots"


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


def load_scene(scene_name: str) -> mujoco.MjModel:
    """Load an MJCF scene by name.

    This is the primary entrypoint for loading simulation scenes.
    Resolves the scene path, then delegates to ``mujoco_env.load_model()``.

    Args:
        scene_name: Scene name (e.g., ``"alex_door_workspace"``).

    Returns:
        Compiled MuJoCo model.

    Raises:
        FileNotFoundError: If the scene file is not found.
        mujoco.FatalError: If the MJCF fails to parse.
    """
    from sim.mujoco_env import load_model

    path = resolve_scene_path(scene_name)
    return load_model(path)
