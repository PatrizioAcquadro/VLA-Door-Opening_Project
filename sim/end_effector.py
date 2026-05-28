"""End-effector abstraction layer (Phase 1.1.3).

Provides a uniform command interface for different end-effectors.
EZGripper (1-DoF grasp) and future Ability Hand (6-DoF) implement
the same ABC.

Usage::

    from sim.end_effector import EZGripperInterface

    ee = EZGripperInterface(model, data, side="left")
    ee.set_grasp(0.5)  # half open
    ctrl = ee.get_ctrl()

Abstraction notes:
    - ``gripper_cmd ∈ [0, 1]``: 0 = closed, 1 = open (all end-effectors)
    - EZGripper: 1 actuated DoF, maps cmd to knuckle_palm_l1 joint [0, 1.94] rad
    - Future Ability Hand: 6+ DoF, scalar cmd maps to power grasp;
      per-finger control via extended API (not defined here)
"""

from __future__ import annotations

import abc
from typing import Literal

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


class EndEffectorInterface(abc.ABC):
    """Abstract base class for end-effector command interfaces.

    All end-effectors support a normalized grasp command in [0, 1]
    (0 = closed, 1 = open) and report tool frame pose.
    """

    @abc.abstractmethod
    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        data: mujoco.MjData,  # type: ignore[name-defined]
        side: Literal["left", "right"],
    ) -> None: ...

    @property
    @abc.abstractmethod
    def n_actuated_dof(self) -> int:
        """Number of independently actuated degrees of freedom."""
        ...

    @property
    @abc.abstractmethod
    def actuator_indices(self) -> np.ndarray:
        """Indices into data.ctrl for this end-effector's actuators."""
        ...

    @abc.abstractmethod
    def set_grasp(self, cmd: float) -> None:
        """Set grasp command. 0 = fully closed, 1 = fully open."""
        ...

    @abc.abstractmethod
    def get_ctrl(self) -> np.ndarray:
        """Return current MuJoCo ctrl values for this EE's actuators."""
        ...

    @abc.abstractmethod
    def get_grasp_state(self) -> float:
        """Return current grasp state in [0, 1] from joint positions."""
        ...

    @abc.abstractmethod
    def get_tool_frame_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position[3], quaternion[4]) of the tool frame site."""
        ...


class EZGripperInterface(EndEffectorInterface):
    """SAKE EZGripper Gen2 command interface.

    Maps a scalar grasp command [0, 1] to the single actuated
    knuckle_palm_l1_1 joint. All other finger joints are driven
    by MuJoCo equality constraints (1:1 coupling).

    Args:
        model: Compiled MuJoCo model containing EZGripper.
        data: MjData instance.
        side: ``"left"`` or ``"right"``.
    """

    # Joint range for knuckle_palm_l1: [0, 1.94] rad
    JOINT_RANGE_LO: float = 0.0
    JOINT_RANGE_HI: float = 1.94

    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        data: mujoco.MjData,  # type: ignore[name-defined]
        side: Literal["left", "right"],
    ) -> None:
        self._model = model
        self._data = data
        self._side = side

        # Resolve actuator index
        actuator_name = f"{side}_ezgripper"
        self._actuator_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if self._actuator_id < 0:
            raise ValueError(f"Actuator '{actuator_name}' not found in model")

        # Resolve primary joint index (for reading grasp state)
        joint_name = f"{side}_knuckle_palm_l1_1"
        self._joint_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if self._joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in model")

        # Resolve tool frame site
        site_name = f"{side}_tool_frame"
        self._tool_site_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
            model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self._tool_site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")

        self._cmd: float = 0.0

    @property
    def n_actuated_dof(self) -> int:
        return 1

    @property
    def actuator_indices(self) -> np.ndarray:
        return np.array([self._actuator_id], dtype=int)

    def set_grasp(self, cmd: float) -> None:
        """Set grasp command. 0 = fully closed, 1 = fully open."""
        cmd = float(np.clip(cmd, 0.0, 1.0))
        self._cmd = cmd
        joint_target = cmd * self.JOINT_RANGE_HI
        self._data.ctrl[self._actuator_id] = joint_target

    def get_ctrl(self) -> np.ndarray:
        joint_target = self._cmd * self.JOINT_RANGE_HI
        return np.array([joint_target], dtype=np.float64)

    def get_grasp_state(self) -> float:
        """Return current grasp state in [0, 1] from primary joint position."""
        qpos_idx = self._model.jnt_qposadr[self._joint_id]
        current = self._data.qpos[qpos_idx]
        return float(np.clip(current / self.JOINT_RANGE_HI, 0.0, 1.0))

    def get_tool_frame_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position[3], quaternion[4]) of the tool frame site."""
        pos = self._data.site_xpos[self._tool_site_id].copy()
        mat = self._data.site_xmat[self._tool_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())  # type: ignore[name-defined]
        return pos, quat
