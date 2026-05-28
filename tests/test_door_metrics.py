"""Tests for active door-opening metric extraction."""

from sim.door_metrics import DoorMetricConfig, DoorMetricsTracker, flatten_door_metrics


def test_empty_rollout_metrics_are_false() -> None:
    tracker = DoorMetricsTracker()
    metrics = tracker.finalize()

    assert metrics["handle_contact_success"] is False
    assert metrics["latch_release_success"] is False
    assert metrics["target_reached"] is False


def test_successful_rollout_metrics() -> None:
    tracker = DoorMetricsTracker(
        DoorMetricConfig(
            target_open_angle_rad=1.0,
            angle_tolerance_rad=0.1,
            latch_release_angle_rad=0.35,
        )
    )

    tracker.update_values(door_angle=0.1, handle_touch=0.0)
    tracker.update_values(door_angle=0.4, handle_touch=0.2)
    tracker.update_values(door_angle=0.95, handle_touch=0.3)

    metrics = tracker.finalize()

    assert metrics["handle_contact_success"] is True
    assert metrics["latch_release_success"] is True
    assert metrics["contact_stability"] is True
    assert metrics["target_reached"] is True
    assert metrics["force_limit_violation"] is False


def test_force_limit_violation_is_sticky() -> None:
    tracker = DoorMetricsTracker()
    tracker.update_values(door_angle=0.0, force_limit_violation=True)
    tracker.update_values(door_angle=0.1, force_limit_violation=False)

    assert tracker.finalize()["force_limit_violation"] is True


def test_flatten_door_metrics_for_logging() -> None:
    flat = flatten_door_metrics(
        {
            "handle_contact_success": True,
            "max_door_angle_rad": 0.75,
        }
    )

    assert flat == {
        "door/handle_contact_success": 1.0,
        "door/max_door_angle_rad": 0.75,
    }
