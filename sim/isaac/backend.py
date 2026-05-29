"""Minimal import-safe Isaac door backend adapter surface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sim.isaac.contracts import (
    IsaacContractError,
    IsaacRuntimeUnavailable,
    validate_action_17d,
    validate_door_metrics,
    validate_render_views,
    validate_robot_state_52d,
)


class IsaacDoorAdapter(Protocol):
    """Protocol implemented by a live Isaac runtime adapter."""

    def reset(self) -> Any: ...

    def step(self, action_17d: np.ndarray) -> Any: ...

    def get_robot_state_52d(self) -> Any: ...

    def get_door_metrics(self) -> Mapping[str, Any]: ...

    def render_views(self) -> Mapping[str, Any]: ...


class IsaacDoorBackend:
    """Thin contract-checking facade for the first Isaac vertical slice.

    The default instance is intentionally not a fake simulator. Live methods
    raise ``IsaacRuntimeUnavailable`` until a real Isaac adapter is injected by
    code running under Isaac Sim/Isaac Lab.
    """

    def __init__(self, adapter: IsaacDoorAdapter | None = None) -> None:
        self._adapter = adapter

    def reset(self) -> Any:
        """Reset the live Isaac scene."""
        return self._require_adapter().reset()

    def step(self, action_17d: Any) -> Any:
        """Apply one frozen 17-D Alex action to the live Isaac scene."""
        action = validate_action_17d(action_17d)
        return self._require_adapter().step(action)

    def get_robot_state_52d(self) -> np.ndarray:
        """Return the frozen 52-D Alex robot-state vector from Isaac."""
        state = self._require_adapter().get_robot_state_52d()
        return validate_robot_state_52d(state)

    def get_door_metrics(self) -> dict[str, float | bool]:
        """Return frozen door-opening metric names from Isaac."""
        metrics = self._require_adapter().get_door_metrics()
        return validate_door_metrics(metrics)

    def render_views(self) -> dict[str, np.ndarray]:
        """Return four 320x320 RGB camera views from Isaac."""
        views = self._require_adapter().render_views()
        return validate_render_views(views)

    def record_episode(
        self,
        *,
        output_path: str | Path,
        language_instruction: str,
        actions_17d: Any,
        robot_states_52d: Any,
        door_metrics: Mapping[str, Any],
        camera_view_names: tuple[str, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Record an episode manifest without inventing Isaac observations.

        Pixel arrays are intentionally not serialized here. The manifest
        preserves the language/data surface and validates action, state, camera
        name, and metric contracts for data produced by a live adapter.
        """
        if not language_instruction:
            raise IsaacContractError("language_instruction must be non-empty")

        actions = np.asarray(actions_17d, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 17:
            raise IsaacContractError(f"Expected actions shape (N, 17), got {actions.shape}")
        states = np.asarray(robot_states_52d, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 52:
            raise IsaacContractError(f"Expected robot states shape (N, 52), got {states.shape}")
        if actions.shape[0] != states.shape[0]:
            raise IsaacContractError(
                f"Actions/states length mismatch: {actions.shape[0]} != {states.shape[0]}"
            )

        metrics = validate_door_metrics(door_metrics)
        if camera_view_names is not None:
            from sim.isaac.contracts import CAMERA_NAMES

            if tuple(camera_view_names) != CAMERA_NAMES:
                raise IsaacContractError(
                    f"camera_view_names must be {CAMERA_NAMES}, got {tuple(camera_view_names)}"
                )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language_instruction": language_instruction,
            "num_steps": int(actions.shape[0]),
            "action_shape": list(actions.shape),
            "robot_state_shape": list(states.shape),
            "camera_view_names": list(camera_view_names) if camera_view_names else None,
            "door_metrics": metrics,
            "metadata": dict(metadata or {}),
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return output_path

    def _require_adapter(self) -> IsaacDoorAdapter:
        if self._adapter is None:
            raise IsaacRuntimeUnavailable(
                "Live Isaac backend is not connected. Run under Isaac Sim/Isaac Lab and "
                "inject a real adapter; no MuJoCo or fake Isaac fallback is used."
            )
        return self._adapter
