"""Frozen 17-D action space for the Alex robot (Phase 1.1.5).

Defines the canonical action vector layout, normalization, and the
action-to-control pipeline that converts normalized policy outputs
into safe MuJoCo actuator commands.

Action vector (17-D):
    [0]     Δq spine_z
    [1:8]   Δq left arm  (shoulder_y, shoulder_x, shoulder_z, elbow_y,
                           wrist_z, wrist_x, gripper_z)
    [8:15]  Δq right arm (same joint order)
    [15]    gripper_left  (absolute, 0 = closed, 1 = open)
    [16]    gripper_right (absolute, 0 = closed, 1 = open)

Arm actions are normalized to ``[-1, 1]`` and mapped to joint-space
deltas via per-joint ``delta_q_max``.  Gripper actions are absolute
commands in ``[0, 1]``.

Usage::

    from sim.action_space import AlexActionSpace

    action_space = AlexActionSpace(model)
    action = np.zeros(17)
    action_space.apply_action(action, data)       # zero-delta, grippers closed
    state = action_space.get_current_state(data)   # 17-D observation
"""

from __future__ import annotations

import numpy as np

from sim.control import JOINT_VELOCITY_LIMITS, AlexController

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Frozen contract constants
# ---------------------------------------------------------------------------

ACTION_DIM: int = 17
ARM_DIM: int = 15  # spine(1) + left_arm(7) + right_arm(7)
GRIPPER_DIM: int = 2

#: Canonical arm actuator ordering — matches MJCF actuator order for first 15.
#: Note: elbow actuators are named ``left_elbow`` / ``right_elbow`` in MJCF
#: (not ``left_elbow_y`` / ``right_elbow_y``).
ARM_ACTUATOR_NAMES: list[str] = [
    "spine_z",
    # Left arm (7 DoF)
    "left_shoulder_y",
    "left_shoulder_x",
    "left_shoulder_z",
    "left_elbow",
    "left_wrist_z",
    "left_wrist_x",
    "left_gripper_z",
    # Right arm (7 DoF)
    "right_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_z",
    "right_elbow",
    "right_wrist_z",
    "right_wrist_x",
    "right_gripper_z",
]

#: Mapping from actuator name to joint name (for velocity limit lookup).
#: Only differs for elbows where actuator is ``left_elbow`` but joint is ``left_elbow_y``.
_ACTUATOR_TO_JOINT: dict[str, str] = {
    "left_elbow": "left_elbow_y",
    "right_elbow": "right_elbow_y",
}

#: EZGripper actuator names (left, right).
GRIPPER_ACTUATOR_NAMES: list[str] = ["left_ezgripper", "right_ezgripper"]

#: EZGripper joint range [0, 1.94] rad — from EZGripperInterface.
EZGRIPPER_JOINT_RANGE_HI: float = 1.94

#: Default policy control rate (Hz).
DEFAULT_CONTROL_HZ: float = 20.0

#: Default fraction of hardware velocity limit used for delta_q_max.
DEFAULT_RATE_LIMIT_FACTOR: float = 0.8


class AlexActionSpace:
    """Frozen 17-D action space for the Alex robot.

    Converts normalized actions ``[-1, 1]^15 × [0, 1]^2`` into safe
    MuJoCo actuator commands using per-joint delta limits and the
    ``AlexController`` position clamp.

    Args:
        model: Compiled MuJoCo model.
        control_hz: Policy control frequency in Hz (default 20).
        rate_limit_factor: Fraction of hardware velocity limit used to
            compute ``delta_q_max`` (default 0.8).
    """

    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        control_hz: float = DEFAULT_CONTROL_HZ,
        rate_limit_factor: float = DEFAULT_RATE_LIMIT_FACTOR,
    ) -> None:
        self._model = model
        self._control_hz = control_hz
        self._control_dt = 1.0 / control_hz

        # --- Resolve arm actuator indices (first 15) -----------------------
        self._arm_actuator_ids = np.zeros(ARM_DIM, dtype=int)
        for i, name in enumerate(ARM_ACTUATOR_NAMES):
            act_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if act_id < 0:
                raise ValueError(f"Arm actuator '{name}' not found in model")
            self._arm_actuator_ids[i] = act_id

        # Map from actuator -> joint (for reading qpos)
        self._arm_jnt_ids = np.array(
            [model.actuator_trnid[a, 0] for a in self._arm_actuator_ids], dtype=int
        )

        # --- Resolve gripper actuator indices (last 2) ---------------------
        self._gripper_actuator_ids = np.zeros(GRIPPER_DIM, dtype=int)
        for i, name in enumerate(GRIPPER_ACTUATOR_NAMES):
            act_id = mujoco.mj_name2id(  # type: ignore[name-defined]
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if act_id < 0:
                raise ValueError(f"Gripper actuator '{name}' not found in model")
            self._gripper_actuator_ids[i] = act_id

        # Gripper primary joints (for reading grasp state)
        self._gripper_jnt_ids = np.array(
            [model.actuator_trnid[a, 0] for a in self._gripper_actuator_ids], dtype=int
        )

        # --- Compute delta_q_max per arm joint -----------------------------
        self._delta_q_max = np.zeros(ARM_DIM, dtype=np.float64)
        for i, act_name in enumerate(ARM_ACTUATOR_NAMES):
            # Look up velocity limit using joint name (differs for elbows)
            jnt_name = _ACTUATOR_TO_JOINT.get(act_name, act_name)
            vel_limit = JOINT_VELOCITY_LIMITS.get(jnt_name, 25.0)
            self._delta_q_max[i] = vel_limit * self._control_dt * rate_limit_factor

        # --- Arm joint position limits (for clamping targets) --------------
        self._pos_lo = np.array(
            [model.jnt_range[j, 0] for j in self._arm_jnt_ids], dtype=np.float64
        )
        self._pos_hi = np.array(
            [model.jnt_range[j, 1] for j in self._arm_jnt_ids], dtype=np.float64
        )

        # --- Internal AlexController for position clamping only ------------
        # Rate limiting is disabled (factor=0) because the action normalization
        # already bounds deltas via delta_q_max.
        self._controller = AlexController(model, rate_limit_factor=0.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Total action dimensionality (17)."""
        return ACTION_DIM

    @property
    def arm_dim(self) -> int:
        """Arm action dimensionality (15)."""
        return ARM_DIM

    @property
    def gripper_dim(self) -> int:
        """Gripper action dimensionality (2)."""
        return GRIPPER_DIM

    @property
    def delta_q_max(self) -> np.ndarray:
        """Max joint delta per control step for each arm joint, shape ``(15,)``."""
        return self._delta_q_max.copy()

    @property
    def control_hz(self) -> float:
        """Policy control frequency in Hz."""
        return self._control_hz

    @property
    def control_dt(self) -> float:
        """Control period in seconds (1 / control_hz)."""
        return self._control_dt

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def denormalize(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert a normalized 17-D action to (arm_delta_q, gripper_cmd).

        Args:
            action: Normalized action, shape ``(17,)``.
                Arm dims in ``[-1, 1]``, gripper dims in ``[0, 1]``.

        Returns:
            Tuple of:
                - ``arm_delta_q``: joint deltas in radians, shape ``(15,)``.
                - ``gripper_cmd``: absolute gripper commands in ``[0, 1]``, shape ``(2,)``.
        """
        action = np.asarray(action, dtype=np.float64)
        arm_action = np.clip(action[:ARM_DIM], -1.0, 1.0)
        gripper_action = np.clip(action[ARM_DIM:], 0.0, 1.0)

        arm_delta_q = arm_action * self._delta_q_max
        return arm_delta_q, gripper_action

    def normalize(
        self,
        arm_delta_q: np.ndarray,
        gripper_cmd: np.ndarray,
    ) -> np.ndarray:
        """Convert raw arm deltas and gripper commands to a normalized 17-D action.

        Args:
            arm_delta_q: Joint deltas in radians, shape ``(15,)``.
            gripper_cmd: Gripper commands in ``[0, 1]``, shape ``(2,)``.

        Returns:
            Normalized action, shape ``(17,)``.
        """
        arm_delta_q = np.asarray(arm_delta_q, dtype=np.float64)
        gripper_cmd = np.asarray(gripper_cmd, dtype=np.float64)

        arm_norm = arm_delta_q / self._delta_q_max
        arm_norm = np.clip(arm_norm, -1.0, 1.0)
        gripper_norm = np.clip(gripper_cmd, 0.0, 1.0)
        return np.concatenate([arm_norm, gripper_norm])

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action(
        self,
        action: np.ndarray,
        data: mujoco.MjData,  # type: ignore[name-defined]
    ) -> None:
        """Apply a normalized 17-D action to the simulation.

        Pipeline:
            1. Clip and denormalize action.
            2. Compute arm targets: ``current_q + delta_q``.
            3. Compute gripper targets: ``cmd * EZGRIPPER_JOINT_RANGE_HI``.
            4. Assemble full ctrl vector and pass through AlexController
               for position-limit clamping.
            5. Write to ``data.ctrl``.

        Args:
            action: Normalized action, shape ``(17,)``.
            data: MjData instance to update.
        """
        arm_delta_q, gripper_cmd = self.denormalize(action)

        # Current arm joint positions
        current_arm_q = np.array([data.qpos[j] for j in self._arm_jnt_ids], dtype=np.float64)

        # Arm targets: current + delta, clamped to joint limits
        arm_target = current_arm_q + arm_delta_q
        np.clip(arm_target, self._pos_lo, self._pos_hi, out=arm_target)

        # Gripper targets: map [0, 1] -> [0, 1.94] rad
        gripper_target = gripper_cmd * EZGRIPPER_JOINT_RANGE_HI

        # Assemble full ctrl (17-D) in actuator order and apply safety clamp
        desired_ctrl = np.zeros(self._model.nu, dtype=np.float64)
        desired_ctrl[self._arm_actuator_ids] = arm_target
        desired_ctrl[self._gripper_actuator_ids] = gripper_target

        safe_ctrl = self._controller.apply(desired_ctrl, data)
        data.ctrl[:] = safe_ctrl

    # ------------------------------------------------------------------
    # State observation
    # ------------------------------------------------------------------

    def get_current_state(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
    ) -> np.ndarray:
        """Return current 17-D state (arm qpos + gripper state).

        Returns:
            Array of shape ``(17,)`` — first 15 are arm joint positions
            (radians), last 2 are gripper states in ``[0, 1]``.
        """
        arm_q = np.array([data.qpos[j] for j in self._arm_jnt_ids], dtype=np.float64)
        gripper_q = np.array([data.qpos[j] for j in self._gripper_jnt_ids], dtype=np.float64)
        gripper_state = np.clip(gripper_q / EZGRIPPER_JOINT_RANGE_HI, 0.0, 1.0)
        return np.concatenate([arm_q, gripper_state])
