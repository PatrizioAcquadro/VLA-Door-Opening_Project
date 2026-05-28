"""Control pipeline for Alex robot: safety clamps, rate limiting, velocity bounds.

Provides a thin wrapper around MuJoCo position actuators with:
- Joint position clamps (redundant with MJCF limits, defense-in-depth)
- Velocity-based rate limiting (max delta per control step)
- Effort awareness (informational; actual clamping done by actuator forcerange)

Usage::

    from sim.control import AlexController

    controller = AlexController(model)
    safe_ctrl = controller.apply(desired_ctrl, data)
    data.ctrl[:] = safe_ctrl
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# URDF-sourced physical limits
# ---------------------------------------------------------------------------

# Velocity limits (rad/s) from URDF `velocity` attribute.
# Right arm uses left arm values (URDF right side has placeholder 100 rad/s).
JOINT_VELOCITY_LIMITS: dict[str, float] = {
    "spine_z": 9.0,
    "left_shoulder_y": 9.0,
    "left_shoulder_x": 9.0,
    "left_shoulder_z": 11.5,
    "left_elbow_y": 11.5,
    "left_wrist_z": 25.0,
    "left_wrist_x": 25.0,
    "left_gripper_z": 25.0,
    "right_shoulder_y": 9.0,
    "right_shoulder_x": 9.0,
    "right_shoulder_z": 11.5,
    "right_elbow_y": 11.5,
    "right_wrist_z": 25.0,
    "right_wrist_x": 25.0,
    "right_gripper_z": 25.0,
    # EZGripper joints — Dynamixel MX-64AR (~6.6 rad/s no-load)
    "left_knuckle_palm_l1_1": 6.6,
    "left_knuckle_palm_l1_2": 6.6,
    "left_knuckle_l1_l2_1": 6.6,
    "left_knuckle_l1_l2_2": 6.6,
    "right_knuckle_palm_l1_1": 6.6,
    "right_knuckle_palm_l1_2": 6.6,
    "right_knuckle_l1_l2_1": 6.6,
    "right_knuckle_l1_l2_2": 6.6,
}

# Effort limits (N·m) matching MJCF actuator forcerange.
# Based on URDF effort attribute, with shoulder_x reduced to 100 N·m
# (conservative for roll axis). Right arm mirrors left arm values.
JOINT_EFFORT_LIMITS: dict[str, float] = {
    "spine_z": 150.0,
    "left_shoulder_y": 150.0,
    "left_shoulder_x": 100.0,
    "left_shoulder_z": 80.0,
    "left_elbow_y": 80.0,
    "left_wrist_z": 20.0,
    "left_wrist_x": 20.0,
    "left_gripper_z": 20.0,
    "right_shoulder_y": 150.0,
    "right_shoulder_x": 100.0,
    "right_shoulder_z": 80.0,
    "right_elbow_y": 80.0,
    "right_wrist_z": 20.0,
    "right_wrist_x": 20.0,
    "right_gripper_z": 20.0,
    # EZGripper — Dynamixel MX-64AR (~6 N·m stall), MJCF forcerange=8 N·m
    "left_knuckle_palm_l1_1": 8.0,
    "right_knuckle_palm_l1_1": 8.0,
}


class AlexController:
    """Safety-clamped position controller for the Alex robot.

    Wraps the MuJoCo position actuator interface with software-level
    safety clamps and rate limiting.

    Args:
        model: Compiled MuJoCo model.
        rate_limit_factor: Fraction of max velocity used for rate limiting.
            Default 0.8 (80% of hardware velocity limit per timestep).
    """

    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        rate_limit_factor: float = 0.8,
    ) -> None:
        self._model = model
        self._nu = model.nu
        self._dt = model.opt.timestep
        self._rate_limit_enabled = rate_limit_factor > 0.0

        # Build ordered arrays aligned with actuator indices
        self._jnt_ids = np.array([model.actuator_trnid[i, 0] for i in range(self._nu)], dtype=int)
        self._pos_lo = np.array([model.jnt_range[j, 0] for j in self._jnt_ids], dtype=np.float64)
        self._pos_hi = np.array([model.jnt_range[j, 1] for j in self._jnt_ids], dtype=np.float64)

        # Max position delta per physics timestep = vel_limit * dt * factor
        self._max_delta = np.zeros(self._nu, dtype=np.float64)
        for i in range(self._nu):
            jnt_name = mujoco.mj_id2name(  # type: ignore[name-defined]
                model, mujoco.mjtObj.mjOBJ_JOINT, int(self._jnt_ids[i])
            )
            vel_limit = JOINT_VELOCITY_LIMITS.get(jnt_name, 25.0)
            self._max_delta[i] = vel_limit * self._dt * rate_limit_factor

    def apply(
        self,
        desired_ctrl: np.ndarray,
        data: mujoco.MjData,  # type: ignore[name-defined]
    ) -> np.ndarray:
        """Apply safety clamps and rate limiting to desired control.

        1. Clamp desired_ctrl to joint position limits.
        2. Compute delta from current qpos.
        3. Clamp delta by max allowed change per timestep.
        4. Return current_qpos + clamped_delta.

        Args:
            desired_ctrl: Raw desired joint positions, shape ``(nu,)``.
            data: Current MjData (reads qpos for rate limiting).

        Returns:
            Safe control vector, shape ``(nu,)``.
        """
        ctrl = np.asarray(desired_ctrl, dtype=np.float64).copy()

        # 1. Position clamp
        np.clip(ctrl, self._pos_lo, self._pos_hi, out=ctrl)

        # 2. Rate limiting (skipped when rate_limit_factor == 0)
        if self._rate_limit_enabled:
            current_q = np.array([data.qpos[j] for j in self._jnt_ids])
            delta = ctrl - current_q
            np.clip(delta, -self._max_delta, self._max_delta, out=delta)
            ctrl = current_q + delta

            # 3. Re-clamp after rate limiting (belt-and-suspenders)
            np.clip(ctrl, self._pos_lo, self._pos_hi, out=ctrl)

        result: np.ndarray = ctrl

        return result

    # ------------------------------------------------------------------
    # Read-only accessors (used by AlexActionSpace)
    # ------------------------------------------------------------------

    def get_current_positions(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
    ) -> np.ndarray:
        """Read current joint positions for all actuated joints.

        Returns:
            Array of shape ``(nu,)`` with current qpos for each actuator's joint.
        """
        return np.array([data.qpos[j] for j in self._jnt_ids])

    @property
    def jnt_ids(self) -> np.ndarray:
        """Joint indices (into qpos) for each actuator, shape ``(nu,)``."""
        return self._jnt_ids.copy()

    @property
    def max_delta(self) -> np.ndarray:
        """Max position delta per physics timestep for each actuator, shape ``(nu,)``."""
        return self._max_delta.copy()

    @property
    def pos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint position limits ``(lower, upper)``, each shape ``(nu,)``."""
        return self._pos_lo.copy(), self._pos_hi.copy()
