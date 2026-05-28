"""Minimal MuJoCo environment loader and stepper for validation."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def load_model(mjcf_path: str | Path) -> mujoco.MjModel:
    """Load an MJCF XML file and return a compiled MjModel.

    Args:
        mjcf_path: Path to the MJCF XML file.

    Returns:
        Compiled MuJoCo model.

    Raises:
        FileNotFoundError: If mjcf_path does not exist.
        mujoco.FatalError: If the MJCF fails to parse.
    """
    mjcf_path = Path(mjcf_path)
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    return mujoco.MjModel.from_xml_path(str(mjcf_path))


def step_n(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
) -> np.ndarray:
    """Step the simulation n_steps times and return final qpos.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data instance.
        n_steps: Number of simulation steps to advance.

    Returns:
        Copy of qpos after n_steps.
    """
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    return np.array(data.qpos.copy())


def check_deterministic(
    mjcf_path: str | Path,
    n_steps: int = 1000,
    n_trials: int = 3,
) -> bool:
    """Verify that stepping is deterministic across multiple trials.

    Loads the model fresh each trial, steps n_steps, and checks that
    final qpos matches across all trials.

    Args:
        mjcf_path: Path to MJCF XML.
        n_steps: Steps per trial.
        n_trials: Number of independent trials.

    Returns:
        True if all trials produce identical final state.
    """
    results: list[np.ndarray] = []
    for _ in range(n_trials):
        model = load_model(mjcf_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        final_qpos = step_n(model, data, n_steps)
        results.append(final_qpos)

    return all(np.array_equal(results[0], r) for r in results[1:])
