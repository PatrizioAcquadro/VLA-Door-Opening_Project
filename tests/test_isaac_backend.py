"""Contract tests for the import-safe Isaac backend slice."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from sim.isaac import (
    CAMERA_NAMES,
    DOOR_METRIC_NAMES,
    IsaacContractError,
    IsaacDoorBackend,
    IsaacRuntimeUnavailable,
    validate_action_17d,
    validate_door_metrics,
    validate_render_views,
    validate_robot_state_52d,
)


def test_import_safety_does_not_import_isaac_runtime_modules() -> None:
    assert "omni" not in sys.modules
    assert "pxr" not in sys.modules
    assert "isaaclab" not in sys.modules


def test_backend_exposes_required_methods() -> None:
    backend = IsaacDoorBackend()
    for name in (
        "reset",
        "step",
        "get_robot_state_52d",
        "get_door_metrics",
        "render_views",
        "record_episode",
    ):
        assert callable(getattr(backend, name))


def test_step_validates_action_shape_before_runtime() -> None:
    backend = IsaacDoorBackend()
    with pytest.raises(IsaacContractError, match="Expected action shape"):
        backend.step(np.zeros(16))

    with pytest.raises(IsaacRuntimeUnavailable, match="Live Isaac backend is not connected"):
        backend.step(np.zeros(17))


def test_contract_validators_accept_frozen_shapes_and_names() -> None:
    action = validate_action_17d(np.zeros(17))
    state = validate_robot_state_52d(np.zeros(52))
    metrics = validate_door_metrics(
        {
            "handle_contact_success": True,
            "latch_release_success": False,
            "max_door_angle_rad": 0.0,
            "final_angle_error_rad": 1.0,
            "contact_stability": False,
            "force_limit_violation": False,
            "recovery_success": False,
        }
    )

    assert action.shape == (17,)
    assert state.shape == (52,)
    assert tuple(metrics) == DOOR_METRIC_NAMES


def test_render_views_requires_exact_four_320_square_rgb_views() -> None:
    views = {name: np.zeros((320, 320, 3), dtype=np.uint8) for name in CAMERA_NAMES}
    validated = validate_render_views(views)
    assert tuple(validated) == CAMERA_NAMES

    bad_views = dict(views)
    bad_views["overhead"] = np.zeros((320, 320, 4), dtype=np.uint8)
    with pytest.raises(IsaacContractError, match="expected shape"):
        validate_render_views(bad_views)


def test_record_episode_preserves_language_surface_and_shapes(tmp_path) -> None:
    backend = IsaacDoorBackend()
    path = backend.record_episode(
        output_path=tmp_path / "episode_manifest.json",
        language_instruction="open the door",
        actions_17d=np.zeros((2, 17)),
        robot_states_52d=np.zeros((2, 52)),
        camera_view_names=CAMERA_NAMES,
        door_metrics={
            "handle_contact_success": False,
            "latch_release_success": False,
            "max_door_angle_rad": 0.0,
            "final_angle_error_rad": 1.0472,
            "contact_stability": False,
            "force_limit_violation": False,
            "recovery_success": False,
        },
    )

    text = path.read_text()
    assert "open the door" in text
    assert '"action_shape": [' in text
