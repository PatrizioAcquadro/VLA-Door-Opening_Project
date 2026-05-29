"""Contract tests for the import-safe Isaac backend slice."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from sim.isaac import (
    ALEX_DEPENDENT_CAMERA_NAMES,
    CAMERA_NAMES,
    DOOR_METRIC_NAMES,
    IsaacContractError,
    IsaacDoorBackend,
    IsaacNativeDoorTask,
    IsaacRuntimeUnavailable,
    LatchProxy,
    NativeDoorTaskConfig,
    build_native_door_metrics,
    compute_target_angle_error,
    validate_action_17d,
    validate_door_metrics,
    validate_live_camera_request,
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
            "target_angle_error": 1.0,
        }
    )

    assert action.shape == (17,)
    assert state.shape == (52,)
    assert tuple(metrics) == (*DOOR_METRIC_NAMES, "target_angle_error")


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


def test_native_door_task_config_and_import_safe_surface() -> None:
    config = NativeDoorTaskConfig()
    task = IsaacNativeDoorTask(config)

    assert task.paths.hallway_scene == "/"
    assert task.paths.hinge_joint.endswith("/revolute_hinge")
    assert config.to_json_dict()["hallway_scene_path"].endswith("Hallway.usdc")
    assert "omni" not in sys.modules
    assert "pxr" not in sys.modules


def test_latch_proxy_transitions_only_after_real_contact_threshold() -> None:
    latch = LatchProxy(
        contact_threshold_n=0.5,
        latch_release_angle_rad=0.35,
        target_open_angle_rad=1.0,
        angle_tolerance_rad=0.1,
    )

    latch.update(step_index=0, handle_force_n=0.1, door_angle_rad=0.0)
    assert latch.state == "locked"

    latch.update(step_index=1, handle_force_n=0.6, door_angle_rad=0.0)
    assert latch.state == "released"
    assert latch.transitions[-1]["reason"] == "handle_contact_force_exceeded_threshold"

    latch.update(step_index=2, handle_force_n=0.6, door_angle_rad=0.4)
    assert latch.state == "opening"


def test_native_door_metrics_keep_final_error_compatibility() -> None:
    latch = LatchProxy(
        contact_threshold_n=0.1,
        latch_release_angle_rad=0.35,
        target_open_angle_rad=1.0,
        angle_tolerance_rad=0.1,
    )
    latch.update(step_index=0, handle_force_n=0.2, door_angle_rad=0.5)
    metrics = build_native_door_metrics(
        door_angle_rad=0.75,
        max_door_angle_rad=0.75,
        target_angle_rad=1.0,
        angle_tolerance_rad=0.1,
        latch=latch,
        handle_force_n=0.2,
        contact_stability=True,
    )

    assert compute_target_angle_error(0.75, 1.0) == pytest.approx(0.25)
    assert metrics["target_angle_error"] == pytest.approx(0.25)
    assert metrics["final_angle_error_rad"] == pytest.approx(metrics["target_angle_error"])
    assert validate_door_metrics(metrics)["target_angle_error"] == pytest.approx(0.25)


def test_native_door_camera_request_blocks_unavailable_wrist_views() -> None:
    assert validate_live_camera_request(("overhead", "third_person")) == (
        "overhead",
        "third_person",
    )

    with pytest.raises(IsaacRuntimeUnavailable, match="wrist camera"):
        validate_live_camera_request(ALEX_DEPENDENT_CAMERA_NAMES)
