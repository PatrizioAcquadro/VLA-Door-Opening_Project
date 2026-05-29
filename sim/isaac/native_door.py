"""Isaac-native HallwayScene door task surface.

This module is import-safe: Isaac Sim, Isaac Lab, ``omni``, and ``pxr`` are
imported only from live methods. The live task composes the real Desktop
HallwayScene and adds a documented PhysX task door at the DoorObject_v2 doorway
pose so contract tests can import this file without an Isaac runtime.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sim.isaac.contracts import (
    ACTION_DIM,
    CAMERA_NAMES,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    IsaacContractError,
    IsaacRuntimeUnavailable,
    validate_action_17d,
)

DEFAULT_HALLWAY_SCENE_PATH = Path("/home/pacquadr/Desktop/HallwayScene/Hallway.usdc")
DEFAULT_DOOR_ASSET_PATH = Path("/home/pacquadr/Desktop/HallwayScene/Objects/DoorObject_v2.usda")
LIVE_CAMERA_NAMES: tuple[str, ...] = ("overhead", "third_person")
ALEX_DEPENDENT_CAMERA_NAMES: tuple[str, ...] = ("left_wrist_cam", "right_wrist_cam")


@dataclass(frozen=True)
class NativeDoorTaskConfig:
    """Configuration for the live Isaac-native HallwayScene door task."""

    hallway_scene_path: Path = DEFAULT_HALLWAY_SCENE_PATH
    door_asset_path: Path = DEFAULT_DOOR_ASSET_PATH
    hallway_prim_path: str = "/"
    task_root_path: str = "/World/IsaacNativeDoorTask"
    target_open_angle_rad: float = 1.0472
    angle_tolerance_rad: float = 0.0873
    latch_release_angle_rad: float = 0.35
    contact_threshold_n: float = 1e-5
    control_hz: float = 20.0
    camera_width: int = DEFAULT_CAMERA_WIDTH
    camera_height: int = DEFAULT_CAMERA_HEIGHT
    doorway_hinge_position: tuple[float, float, float] = (
        -1.1407638788223267,
        -2.826054096221924,
        1.009699821472168,
    )
    doorway_pose_source: str = (
        "/home/pacquadr/Desktop/HallwayScene/Objects/DoorObject_v2.usda:"
        "/Root/Geometry/DoorObject/Door xform"
    )
    hinge_axis_token: str = "Z"
    hinge_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    hinge_lower_limit_deg: float = -90.0
    hinge_upper_limit_deg: float = 0.0
    panel_size_m: tuple[float, float, float] = (0.08, 0.90, 2.00)
    frame_size_m: tuple[float, float, float] = (0.12, 0.12, 2.20)
    handle_local_position_m: tuple[float, float, float] = (0.08, 0.35, 0.0)
    handle_size_m: tuple[float, float, float] = (0.12, 0.20, 0.08)
    probe_size_m: tuple[float, float, float] = (0.08, 0.08, 0.08)
    live_camera_names: tuple[str, ...] = LIVE_CAMERA_NAMES

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hallway_scene_path"] = str(self.hallway_scene_path)
        payload["door_asset_path"] = str(self.door_asset_path)
        return payload


@dataclass(frozen=True)
class NativeDoorPrimPaths:
    """Stable prim paths used by the live native door task."""

    root: str
    hallway_scene: str
    frame: str
    panel: str
    hinge_joint: str
    handle: str
    handle_fixed_joint: str
    handle_region: str
    contact_probe: str
    contact_sensor: str
    cameras_root: str
    cameras: dict[str, str]

    @classmethod
    def from_config(cls, config: NativeDoorTaskConfig) -> NativeDoorPrimPaths:
        root = config.task_root_path.rstrip("/")
        cameras_root = f"{root}/Cameras"
        return cls(
            root=root,
            hallway_scene=config.hallway_prim_path,
            frame=f"{root}/fixed_frame",
            panel=f"{root}/dynamic_panel",
            hinge_joint=f"{root}/revolute_hinge",
            handle=f"{root}/handle_collision",
            handle_fixed_joint=f"{root}/handle_fixed_joint",
            handle_region=f"{root}/handle_region",
            contact_probe=f"{root}/handle_region/contact_probe",
            contact_sensor=f"{root}/handle_collision/contact_sensor",
            cameras_root=cameras_root,
            cameras={name: f"{cameras_root}/{name}" for name in LIVE_CAMERA_NAMES},
        )


@dataclass
class LatchProxy:
    """Simple locked-until-handle-contact latch proxy."""

    contact_threshold_n: float
    latch_release_angle_rad: float
    target_open_angle_rad: float
    angle_tolerance_rad: float
    state: str = "locked"
    transitions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def released(self) -> bool:
        return self.state in {"released", "opening", "target_reached"}

    def reset(self) -> None:
        self.state = "locked"
        self.transitions.clear()

    def update(self, *, step_index: int, handle_force_n: float, door_angle_rad: float) -> None:
        if self.state == "locked" and handle_force_n > self.contact_threshold_n:
            self._transition(
                "released",
                step_index=step_index,
                reason="handle_contact_force_exceeded_threshold",
                handle_force_n=handle_force_n,
                door_angle_rad=door_angle_rad,
            )
        if self.state == "released" and door_angle_rad >= self.latch_release_angle_rad:
            self._transition(
                "opening",
                step_index=step_index,
                reason="door_angle_exceeded_latch_release_angle",
                handle_force_n=handle_force_n,
                door_angle_rad=door_angle_rad,
            )
        if (
            self.state == "opening"
            and abs(door_angle_rad - self.target_open_angle_rad) <= self.angle_tolerance_rad
        ):
            self._transition(
                "target_reached",
                step_index=step_index,
                reason="target_angle_reached",
                handle_force_n=handle_force_n,
                door_angle_rad=door_angle_rad,
            )

    def _transition(
        self,
        state: str,
        *,
        step_index: int,
        reason: str,
        handle_force_n: float,
        door_angle_rad: float,
    ) -> None:
        if self.state == state:
            return
        self.transitions.append(
            {
                "step_index": int(step_index),
                "from": self.state,
                "to": state,
                "reason": reason,
                "handle_force_n": float(handle_force_n),
                "door_angle_rad": float(door_angle_rad),
            }
        )
        self.state = state


def compute_target_angle_error(angle_rad: float, target_angle_rad: float) -> float:
    """Return the absolute target-angle error in radians."""
    return float(abs(float(angle_rad) - float(target_angle_rad)))


def build_native_door_metrics(
    *,
    door_angle_rad: float,
    max_door_angle_rad: float,
    target_angle_rad: float,
    angle_tolerance_rad: float,
    latch: LatchProxy,
    handle_force_n: float,
    contact_stability: bool,
    force_limit_violation: bool = False,
    recovery_success: bool = False,
) -> dict[str, float | bool]:
    """Build backward-compatible door metrics plus native task diagnostics."""
    target_angle_error = compute_target_angle_error(door_angle_rad, target_angle_rad)
    handle_contact = handle_force_n > latch.contact_threshold_n
    return {
        "handle_contact_success": bool(handle_contact),
        "latch_release_success": bool(latch.released),
        "max_door_angle_rad": float(max_door_angle_rad),
        "final_angle_error_rad": float(target_angle_error),
        "contact_stability": bool(contact_stability),
        "force_limit_violation": bool(force_limit_violation),
        "recovery_success": bool(recovery_success),
        "target_angle_error": float(target_angle_error),
        "target_reached": bool(target_angle_error <= angle_tolerance_rad),
        "door_angle_rad": float(door_angle_rad),
        "handle_contact_force_n": float(handle_force_n),
        "handle_contact_detected": bool(handle_contact),
        "latch_released": bool(latch.released),
    }


def summarize_rgb_frame(frame: Any) -> dict[str, Any]:
    """Summarize and check a camera frame without storing the full array."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)) if arr.size else None,
        "std": float(np.std(arr)) if arr.size else None,
        "min": int(np.min(arr)) if arr.size else None,
        "max": int(np.max(arr)) if arr.size else None,
        "nonfake_check": bool(
            arr.shape == (DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, 3)
            and arr.dtype == np.uint8
            and arr.size > 0
            and np.isfinite(arr).all()
            and float(np.std(arr)) > 0.0
        ),
    }


def validate_live_camera_request(view_names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return live camera names or raise for Alex-dependent views."""
    requested = tuple(view_names or LIVE_CAMERA_NAMES)
    unknown = [name for name in requested if name not in CAMERA_NAMES]
    if unknown:
        raise IsaacContractError(f"Unknown camera view(s): {unknown}")
    blocked = [name for name in requested if name in ALEX_DEPENDENT_CAMERA_NAMES]
    if blocked:
        raise IsaacRuntimeUnavailable(
            "Alex-dependent wrist camera view(s) are not live in this native "
            f"Hallway door phase: {blocked}. No placeholder or fake frames are returned."
        )
    return requested


class IsaacNativeDoorTask:
    """Live Isaac adapter/task for the HallwayScene native door proof."""

    def __init__(self, config: NativeDoorTaskConfig | None = None) -> None:
        self.config = config or NativeDoorTaskConfig()
        self.paths = NativeDoorPrimPaths.from_config(self.config)
        self.latch = LatchProxy(
            contact_threshold_n=self.config.contact_threshold_n,
            latch_release_angle_rad=self.config.latch_release_angle_rad,
            target_open_angle_rad=self.config.target_open_angle_rad,
            angle_tolerance_rad=self.config.angle_tolerance_rad,
        )
        self._world: Any | None = None
        self._door: Any | None = None
        self._handle: Any | None = None
        self._probe: Any | None = None
        self._contact_sensor: Any | None = None
        self._camera_annotators: dict[str, Any] = {}
        self._render_products: dict[str, Any] = {}
        self._closed_yaw_rad = 0.0
        self._step_index = 0
        self._angle_samples: list[float] = []
        self._target_error_samples: list[float] = []
        self._contact_samples: list[dict[str, Any]] = []
        self._metrics_history: list[dict[str, float | bool]] = []
        self._reset_summary: dict[str, Any] = {}

    @property
    def reset_summary(self) -> dict[str, Any]:
        return dict(self._reset_summary)

    @property
    def angle_samples(self) -> list[float]:
        return list(self._angle_samples)

    @property
    def target_error_samples(self) -> list[float]:
        return list(self._target_error_samples)

    @property
    def contact_samples(self) -> list[dict[str, Any]]:
        return [dict(sample) for sample in self._contact_samples]

    @property
    def metrics_history(self) -> list[dict[str, float | bool]]:
        return [dict(sample) for sample in self._metrics_history]

    def reset(self) -> dict[str, Any]:
        """Compose HallwayScene, add the native PhysX task door, and reset state."""
        runtime = _import_isaac_runtime()
        hallway_summary = _open_usd_summary(self.config.hallway_scene_path, runtime["Usd"])
        door_asset_summary = _inspect_door_asset(self.config.door_asset_path, runtime["Usd"])

        opened_current_stage = runtime["open_stage"](str(self.config.hallway_scene_path))
        if not opened_current_stage:
            raise IsaacRuntimeUnavailable(
                f"Isaac open_stage failed for {self.config.hallway_scene_path}"
            )
        runtime["update_stage"]()

        self._world = runtime["World"](stage_units_in_meters=1.0)
        self._build_native_door(runtime)
        self._build_cameras(runtime)

        self._world.reset()
        self._closed_yaw_rad = self._measure_panel_yaw_rad()
        self._step_index = 0
        self._angle_samples.clear()
        self._target_error_samples.clear()
        self._contact_samples.clear()
        self._metrics_history.clear()
        self.latch.reset()

        initial_contact = self._read_contact()
        initial_angle = self._measure_door_angle_rad()
        self._append_measurements(initial_angle, initial_contact)

        self._reset_summary = {
            "hallway_scene": hallway_summary,
            "door_asset_inspection": door_asset_summary,
            "selected_task_door": {
                "selection_reason": (
                    "Hallway.usdc is opened as the live stage room context. DoorObject_v2.usda "
                    "is inspected and its doorway transform is used for the native "
                    "PhysX task door pose; the task door is built from Isaac-native "
                    "rigid bodies to keep panel, hinge, handle collision, contacts, "
                    "and angle measurement programmatically testable."
                ),
                "doorway_pose_source": self.config.doorway_pose_source,
                "doorway_hinge_position": list(self.config.doorway_hinge_position),
                "hallway_stage_composition": "isaacsim.core.utils.stage.open_stage",
                "prim_paths": asdict(self.paths),
                "hinge_axis": list(self.config.hinge_axis),
                "hinge_axis_token": self.config.hinge_axis_token,
                "hinge_limits_deg": [
                    self.config.hinge_lower_limit_deg,
                    self.config.hinge_upper_limit_deg,
                ],
                "panel_size_m": list(self.config.panel_size_m),
                "handle_size_m": list(self.config.handle_size_m),
            },
            "unavailable_views": dict.fromkeys(
                ALEX_DEPENDENT_CAMERA_NAMES,
                "requires live Alex articulation; no fake frame is returned",
            ),
        }
        return self.reset_summary

    def step(self, action: Any) -> dict[str, Any]:
        """Advance the live simulation using the native task control path."""
        action_arr = validate_action_17d(action)
        self._require_live()

        contact = self._read_contact()
        door_angle = self._measure_door_angle_rad()
        self.latch.update(
            step_index=self._step_index,
            handle_force_n=contact["force_n"],
            door_angle_rad=door_angle,
        )

        self._apply_probe_control(action_arr)
        if self.latch.released:
            self._apply_door_open_control(action_arr)
        else:
            self._door.set_angular_velocity(np.zeros(3, dtype=np.float64))

        self._world.step(render=True)
        self._step_index += 1

        contact = self._read_contact()
        door_angle = self._measure_door_angle_rad()
        self.latch.update(
            step_index=self._step_index,
            handle_force_n=contact["force_n"],
            door_angle_rad=door_angle,
        )
        self._append_measurements(door_angle, contact)
        return {
            "step_index": self._step_index,
            "door_angle_rad": door_angle,
            "contact": contact,
            "latch_state": self.latch.state,
            "metrics": self.get_door_metrics(),
        }

    def get_robot_state_52d(self) -> np.ndarray:
        """The native door phase has no live Alex articulation yet."""
        raise IsaacRuntimeUnavailable(
            "52-D Alex robot state is not live in the native HallwayScene door phase. "
            "This task exposes real door/camera/contact state only; no synthetic Alex "
            "state is returned."
        )

    def get_door_metrics(self) -> dict[str, float | bool]:
        """Return real measured door/contact/latch metrics from the current task state."""
        self._require_live()
        if self._metrics_history:
            return dict(self._metrics_history[-1])
        contact = self._read_contact()
        angle = self._measure_door_angle_rad()
        return self._build_metrics(angle, contact["force_n"])

    def render_views(self, view_names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """Render live Isaac camera frames for available native task views."""
        requested = validate_live_camera_request(view_names)
        self._require_live()
        self._world.step(render=True)

        frames: dict[str, np.ndarray] = {}
        for name in requested:
            data = self._camera_annotators[name].get_data()
            frame = _coerce_rgb_frame(data)
            summary = summarize_rgb_frame(frame)
            if not summary["nonfake_check"]:
                raise IsaacContractError(
                    f"Camera {name!r} produced a non-live or invalid frame summary: {summary}"
                )
            frames[name] = frame
        return frames

    def record_episode(
        self,
        *,
        output_dir: str | Path,
        actions_17d: Any | None = None,
        language_instruction: str = "open the hallway door",
    ) -> Path:
        """Record a real native-door proof episode manifest and camera frame arrays."""
        if not language_instruction:
            raise IsaacContractError("language_instruction must be non-empty")

        self._require_live()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        actions = _default_episode_actions() if actions_17d is None else np.asarray(actions_17d)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise IsaacContractError(
                f"Expected actions shape (N, {ACTION_DIM}), got {actions.shape}"
            )

        observations: list[dict[str, Any]] = []
        for action in actions:
            observations.append(self.step(action))

        frames = self.render_views()
        frames_path = output_dir / "episode_camera_frames.npz"
        np.savez_compressed(frames_path, **frames)

        manifest_path = output_dir / "episode_manifest.json"
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language_instruction": language_instruction,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "task_config": self.config.to_json_dict(),
            "reset_summary": self.reset_summary,
            "num_steps": int(actions.shape[0]),
            "action_shape": list(actions.shape),
            "actions": actions.astype(float).tolist(),
            "observations": _json_safe(observations),
            "metrics_history": _json_safe(self.metrics_history),
            "latch_state_transitions": self.latch.transitions,
            "camera_frame_archive": str(frames_path),
            "camera_frame_summaries": {
                name: summarize_rgb_frame(frame) for name, frame in frames.items()
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path

    def _build_native_door(self, runtime: dict[str, Any]) -> None:
        Gf = runtime["Gf"]
        Sdf = runtime["Sdf"]
        UsdGeom = runtime["UsdGeom"]
        UsdLux = runtime["UsdLux"]
        UsdPhysics = runtime["UsdPhysics"]
        DynamicCuboid = runtime["DynamicCuboid"]
        FixedCuboid = runtime["FixedCuboid"]
        ContactSensor = runtime["ContactSensor"]

        stage = self._world.stage
        UsdGeom.Xform.Define(stage, Sdf.Path(self.paths.root))
        UsdGeom.Xform.Define(stage, Sdf.Path(self.paths.handle_region))
        UsdGeom.Xform.Define(stage, Sdf.Path(self.paths.cameras_root))
        lights_root = f"{self.paths.root}/Lights"
        UsdGeom.Xform.Define(stage, Sdf.Path(lights_root))
        key_light = UsdLux.DistantLight.Define(stage, Sdf.Path(f"{lights_root}/key_light"))
        key_light.CreateIntensityAttr(5000.0)
        fill_light = UsdLux.SphereLight.Define(stage, Sdf.Path(f"{lights_root}/fill_light"))
        fill_light.CreateIntensityAttr(25000.0)
        fill_light.CreateRadiusAttr(2.5)
        fill_xform = UsdGeom.Xformable(fill_light.GetPrim())
        fill_xform.AddTranslateOp().Set(
            Gf.Vec3d(
                float(self.config.doorway_hinge_position[0] + 1.5),
                float(self.config.doorway_hinge_position[1] - 1.5),
                float(self.config.doorway_hinge_position[2] + 2.5),
            )
        )

        hinge = np.asarray(self.config.doorway_hinge_position, dtype=np.float64)
        panel_size = np.asarray(self.config.panel_size_m, dtype=np.float64)
        frame_size = np.asarray(self.config.frame_size_m, dtype=np.float64)
        handle_local = np.asarray(self.config.handle_local_position_m, dtype=np.float64)
        handle_size = np.asarray(self.config.handle_size_m, dtype=np.float64)
        probe_size = np.asarray(self.config.probe_size_m, dtype=np.float64)

        panel_center = hinge + np.array([panel_size[0] / 2.0, panel_size[1] / 2.0, 0.0])
        handle_world = panel_center + handle_local
        probe_start = handle_world + np.array([0.28, 0.0, 0.0])

        self._world.scene.add(
            FixedCuboid(
                prim_path=self.paths.frame,
                name="isaac_native_door_fixed_frame",
                position=hinge,
                scale=frame_size,
                size=1.0,
                color=np.array([0.15, 0.15, 0.15]),
            )
        )
        self._door = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.paths.panel,
                name="isaac_native_door_dynamic_panel",
                mass=18.0,
                position=panel_center,
                scale=panel_size,
                size=1.0,
                color=np.array([0.10, 0.22, 0.75]),
            )
        )

        self._handle = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.paths.handle,
                name="isaac_native_door_handle_collision",
                mass=0.75,
                position=handle_world,
                scale=handle_size,
                size=1.0,
                color=np.array([0.08, 0.08, 0.08]),
            )
        )

        joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(self.paths.hinge_joint))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths.frame)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths.panel)])
        joint.CreateAxisAttr().Set(getattr(UsdPhysics.Tokens, self.config.hinge_axis_token.lower()))
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(float(-panel_size[0] / 2.0), float(-panel_size[1] / 2.0), 0.0)
        )
        joint.CreateLowerLimitAttr().Set(float(self.config.hinge_lower_limit_deg))
        joint.CreateUpperLimitAttr().Set(float(self.config.hinge_upper_limit_deg))

        handle_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(self.paths.handle_fixed_joint))
        handle_joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths.panel)])
        handle_joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths.handle)])
        handle_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in handle_local]))
        handle_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        self._probe = self._world.scene.add(
            DynamicCuboid(
                prim_path=self.paths.contact_probe,
                name="isaac_native_door_handle_contact_probe",
                mass=0.25,
                position=probe_start,
                scale=probe_size,
                size=1.0,
                color=np.array([0.85, 0.12, 0.10]),
            )
        )
        self._contact_sensor = self._world.scene.add(
            ContactSensor(
                prim_path=self.paths.contact_sensor,
                name="isaac_native_door_handle_contact_sensor",
                min_threshold=0.0,
                max_threshold=100000.0,
                radius=-1.0,
            )
        )

    def _build_cameras(self, runtime: dict[str, Any]) -> None:
        rep = runtime["rep"]
        stage = self._world.stage
        hinge = np.asarray(self.config.doorway_hinge_position, dtype=np.float64)
        target = hinge + np.array([0.05, 0.45, 0.0])
        camera_specs = {
            "overhead": {
                "position": target + np.array([0.0, 0.0, 3.2]),
                "target": target,
            },
            "third_person": {
                "position": target + np.array([2.2, -2.4, 1.35]),
                "target": target + np.array([0.0, 0.1, 0.1]),
            },
        }

        for name, spec in camera_specs.items():
            position = np.asarray(spec["position"], dtype=np.float64)
            target_pos = np.asarray(spec["target"], dtype=np.float64)
            camera = rep.create.camera(
                position=tuple(float(v) for v in position),
                look_at=tuple(float(v) for v in target_pos),
                parent=self.paths.cameras_root,
                name=name,
                focal_length=24.0,
            )
            render_product = rep.create.render_product(
                camera,
                (self.config.camera_width, self.config.camera_height),
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([render_product])
            self._render_products[name] = render_product
            self._camera_annotators[name] = annotator

        runtime["update_stage"]()
        camera_prims = [
            prim
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Camera"
            and str(prim.GetPath()).startswith(self.paths.cameras_root)
        ]
        for prim in camera_prims:
            text = str(prim.GetPath())
            for name in self.paths.cameras:
                if name in text:
                    self.paths.cameras[name] = text

    def _apply_probe_control(self, action: np.ndarray) -> None:
        speed = float(np.clip(abs(action[0]), 0.25, 1.75))
        if self.latch.released:
            self._probe.set_linear_velocity(np.array([speed, 0.0, 0.0], dtype=np.float64))
            return
        self._probe.set_linear_velocity(np.array([-speed, 0.0, 0.0], dtype=np.float64))

    def _apply_door_open_control(self, action: np.ndarray) -> None:
        speed = float(np.clip(abs(action[1]) if abs(action[1]) > 1e-6 else 0.75, 0.25, 1.4))
        angular_velocity = np.array([0.0, 0.0, -speed], dtype=np.float64)
        self._door.set_angular_velocity(angular_velocity)
        if self._handle is not None:
            self._handle.set_angular_velocity(angular_velocity)

    def _append_measurements(self, angle: float, contact: dict[str, Any]) -> None:
        self._angle_samples.append(float(angle))
        self._target_error_samples.append(
            compute_target_angle_error(angle, self.config.target_open_angle_rad)
        )
        self._contact_samples.append(dict(contact))
        self._metrics_history.append(self._build_metrics(angle, contact["force_n"]))

    def _build_metrics(self, angle: float, handle_force_n: float) -> dict[str, float | bool]:
        contact_forces = np.asarray(
            [sample["force_n"] for sample in self._contact_samples], dtype=np.float64
        )
        max_handle_force_n = (
            float(np.max(contact_forces)) if contact_forces.size else handle_force_n
        )
        contact_stability = bool(
            contact_forces.size
            and np.mean(contact_forces > self.config.contact_threshold_n) >= 0.25
        )
        return build_native_door_metrics(
            door_angle_rad=angle,
            max_door_angle_rad=max(self._angle_samples or [angle]),
            target_angle_rad=self.config.target_open_angle_rad,
            angle_tolerance_rad=self.config.angle_tolerance_rad,
            latch=self.latch,
            handle_force_n=max_handle_force_n,
            contact_stability=contact_stability,
        )

    def _read_contact(self) -> dict[str, Any]:
        frame = self._contact_sensor.get_current_frame()
        force_n = _force_scalar(frame.get("force", 0.0))
        return {
            "step_index": int(self._step_index),
            "sensor_prim_path": self.paths.contact_sensor,
            "contact_probe_prim_path": self.paths.contact_probe,
            "handle_region_prim_path": self.paths.handle_region,
            "handle_collision_prim_path": self.paths.handle,
            "in_contact": bool(frame.get("in_contact", False)),
            "force_n": float(force_n),
            "number_of_contacts": int(frame.get("number_of_contacts", 0)),
            "physics_step": float(frame.get("physics_step", 0.0)),
        }

    def _measure_panel_yaw_rad(self) -> float:
        _, quat = self._door.get_world_pose()
        return _quat_yaw_rad(quat)

    def _measure_door_angle_rad(self) -> float:
        return abs(_normalize_angle_rad(self._measure_panel_yaw_rad() - self._closed_yaw_rad))

    def _require_live(self) -> None:
        if self._world is None or self._door is None or self._contact_sensor is None:
            raise IsaacRuntimeUnavailable(
                "IsaacNativeDoorTask is not reset. Call reset() under Isaac Sim first."
            )


def _import_isaac_runtime() -> dict[str, Any]:
    try:
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.utils.stage import (
            add_reference_to_stage,
            create_new_stage,
            open_stage,
            update_stage,
        )
        from isaacsim.sensors.physics import ContactSensor
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
    except Exception as exc:  # pragma: no cover - requires Isaac runtime
        raise IsaacRuntimeUnavailable(
            "Isaac Sim runtime imports failed. Run this task through the configured "
            "Isaac/Isaac Lab wrapper; no MuJoCo fallback is used."
        ) from exc
    return {
        "rep": rep,
        "World": World,
        "DynamicCuboid": DynamicCuboid,
        "FixedCuboid": FixedCuboid,
        "ContactSensor": ContactSensor,
        "add_reference_to_stage": add_reference_to_stage,
        "create_new_stage": create_new_stage,
        "open_stage": open_stage,
        "update_stage": update_stage,
        "Gf": Gf,
        "PhysxSchema": PhysxSchema,
        "Sdf": Sdf,
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "UsdLux": UsdLux,
        "UsdPhysics": UsdPhysics,
    }


def _open_usd_summary(path: Path, Usd: Any) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise IsaacRuntimeUnavailable(f"Usd.Stage.Open returned None for {path}")
    prims = [str(prim.GetPath()) for prim in stage.Traverse()]
    return {
        "path": str(path),
        "opened": True,
        "prim_count": len(prims),
        "sample_prims": prims[:40],
    }


def _inspect_door_asset(path: Path, Usd: Any) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise IsaacRuntimeUnavailable(f"Usd.Stage.Open returned None for {path}")

    prims = []
    joints = []
    rigid_bodies = []
    collision_prims = []
    for prim in stage.Traverse():
        applied = [str(schema) for schema in prim.GetAppliedSchemas()]
        prim_info = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "applied_schemas": applied,
        }
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            joints.append(
                {
                    **prim_info,
                    "axis": _attr_value(prim, "physics:axis"),
                    "lower_limit_deg": _attr_value(prim, "physics:lowerLimit"),
                    "upper_limit_deg": _attr_value(prim, "physics:upperLimit"),
                    "body0": _rel_targets(prim, "physics:body0"),
                    "body1": _rel_targets(prim, "physics:body1"),
                }
            )
        if any("RigidBodyAPI" in schema for schema in applied):
            rigid_bodies.append(prim_info)
        if any("CollisionAPI" in schema for schema in applied):
            collision_prims.append(prim_info)
        if "Door" in str(prim.GetPath()) or prim.GetTypeName() == "PhysicsRevoluteJoint":
            prims.append(prim_info)

    return {
        "path": str(path),
        "opened": True,
        "door_related_prim_count": len(prims),
        "sample_door_related_prims": prims[:40],
        "revolute_joints": joints,
        "rigid_bodies": rigid_bodies,
        "collision_prim_count": len(collision_prims),
        "has_dynamic_door_candidate": bool(joints and rigid_bodies),
    }


def _attr_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.HasValue() else None


def _rel_targets(prim: Any, name: str) -> list[str]:
    rel = prim.GetRelationship(name)
    return [str(target) for target in rel.GetTargets()] if rel else []


def _look_at_quat(Gf: Any, position: np.ndarray, target: np.ndarray) -> Any:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rotation = np.column_stack((right, true_up, -forward))
    w, x, y, z = _quat_from_matrix(rotation)
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _quat_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        )
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return (
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        )
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return (
        (matrix[1, 0] - matrix[0, 1]) / scale,
        (matrix[0, 2] + matrix[2, 0]) / scale,
        (matrix[1, 2] + matrix[2, 1]) / scale,
        0.25 * scale,
    )


def _coerce_rgb_frame(data: Any) -> np.ndarray:
    if isinstance(data, dict):
        for key in ("data", "rgb", "LdrColor"):
            if key in data:
                data = data[key]
                break
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _force_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.shape == ():
        return float(arr)
    return float(np.linalg.norm(arr))


def _quat_yaw_rad(quat: Any) -> float:
    w, x, y, z = [float(v) for v in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _normalize_angle_rad(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _default_episode_actions(num_steps: int = 48) -> np.ndarray:
    actions = np.zeros((num_steps, ACTION_DIM), dtype=np.float64)
    actions[:, 0] = 1.0
    actions[:, 1] = 0.85
    return actions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
