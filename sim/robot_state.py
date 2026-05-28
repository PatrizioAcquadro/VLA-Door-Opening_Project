"""Frozen 52-D robot state contract for the Alex robot (Phase 1.1.6).

Defines the canonical state vector layout, normalization ranges, and
extraction from MuJoCo simulation data.

State vector (52-D):
    [0:15]   q           -- arm joint positions (radians)
    [15:30]  q_dot       -- arm joint velocities (rad/s)
    [30:32]  gripper     -- gripper state [0=closed, 1=open]
    [32:35]  left_ee_pos -- left EE position (m, world frame)
    [35:39]  left_ee_quat-- left EE quaternion [w,x,y,z]
    [39:42]  right_ee_pos-- right EE position (m, world frame)
    [42:46]  right_ee_quat-- right EE quaternion [w,x,y,z]
    [46:49]  left_ee_vel -- left EE linear velocity (m/s, world frame)
    [49:52]  right_ee_vel-- right EE linear velocity (m/s, world frame)

Reference frame: World (Z-up, X-forward, Y-left).
Robot base fixed at [0, 0, 1.0], same orientation as world.
Quaternion convention: MuJoCo [w, x, y, z].

Usage::

    from sim.robot_state import AlexRobotState

    robot_state = AlexRobotState(model)
    state = robot_state.get_state(data)          # RobotState dataclass
    flat = robot_state.get_flat_state(data)       # 52-D numpy array
    normed = robot_state.normalize(flat)          # normalized to ~[-1, 1]
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sim.action_space import (
    _ACTUATOR_TO_JOINT,
    ARM_ACTUATOR_NAMES,
    ARM_DIM,
    EZGRIPPER_JOINT_RANGE_HI,
    GRIPPER_ACTUATOR_NAMES,
    GRIPPER_DIM,
)
from sim.control import JOINT_VELOCITY_LIMITS

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Frozen contract constants
# ---------------------------------------------------------------------------

STATE_DIM: int = 52
Q_DIM: int = 15  # spine(1) + left_arm(7) + right_arm(7)
Q_DOT_DIM: int = 15
GRIPPER_STATE_DIM: int = 2
EE_POSE_DIM: int = 14  # (pos3 + quat4) x 2 arms
EE_VEL_DIM: int = 6  # lin_vel3 x 2 arms

# Slice definitions for flat array access
Q_SLICE: slice = slice(0, 15)
Q_DOT_SLICE: slice = slice(15, 30)
GRIPPER_SLICE: slice = slice(30, 32)
LEFT_EE_POS_SLICE: slice = slice(32, 35)
LEFT_EE_QUAT_SLICE: slice = slice(35, 39)
RIGHT_EE_POS_SLICE: slice = slice(39, 42)
RIGHT_EE_QUAT_SLICE: slice = slice(42, 46)
LEFT_EE_VEL_SLICE: slice = slice(46, 49)
RIGHT_EE_VEL_SLICE: slice = slice(49, 52)

#: EE site names for tool frame pose extraction
LEFT_EE_SITE: str = "left_tool_frame"
RIGHT_EE_SITE: str = "right_tool_frame"

#: Arm joint names in canonical order (derived from actuator names)
ARM_JOINT_NAMES: list[str] = [_ACTUATOR_TO_JOINT.get(name, name) for name in ARM_ACTUATOR_NAMES]


# ---------------------------------------------------------------------------
# Normalization ranges
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StateNormRanges:
    """Per-component normalization ranges for the state vector.

    All ranges are stored as numpy arrays aligned with the
    flat state vector. Used for min-max normalization to [-1, 1].
    """

    q_lo: np.ndarray  # (15,) joint position lower bounds (rad)
    q_hi: np.ndarray  # (15,) joint position upper bounds (rad)
    q_dot_lo: np.ndarray  # (15,) negative velocity limits (rad/s)
    q_dot_hi: np.ndarray  # (15,) positive velocity limits (rad/s)
    # gripper: already [0, 1] -- no further normalization needed
    # EE pos: normalized by workspace bounds
    ee_pos_lo: np.ndarray  # (3,) workspace lower bound (m)
    ee_pos_hi: np.ndarray  # (3,) workspace upper bound (m)
    # EE quat: unit quaternion, not min-max normalized (use as-is)
    # EE vel: normalized by max expected velocity
    ee_vel_max: float  # max expected EE velocity (m/s)


# ---------------------------------------------------------------------------
# Robot state dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RobotState:
    """Structured robot state with named fields.

    Provides clear field access for debugging and logging, plus
    ``to_flat_array()`` for the flat vector consumed by the policy.
    """

    q: np.ndarray  # (15,) arm joint positions, radians
    q_dot: np.ndarray  # (15,) arm joint velocities, rad/s
    gripper_state: np.ndarray  # (2,) [0=closed, 1=open]
    left_ee_pos: np.ndarray  # (3,) meters, world frame
    left_ee_quat: np.ndarray  # (4,) [w, x, y, z]
    right_ee_pos: np.ndarray  # (3,) meters, world frame
    right_ee_quat: np.ndarray  # (4,) [w, x, y, z]
    left_ee_vel: np.ndarray  # (3,) m/s, world frame
    right_ee_vel: np.ndarray  # (3,) m/s, world frame
    timestamp: float = 0.0  # simulation time (s)

    def to_flat_array(self) -> np.ndarray:
        """Return the 52-D flat state vector for the policy network."""
        return np.concatenate(
            [
                self.q,
                self.q_dot,
                self.gripper_state,
                self.left_ee_pos,
                self.left_ee_quat,
                self.right_ee_pos,
                self.right_ee_quat,
                self.left_ee_vel,
                self.right_ee_vel,
            ]
        )

    @staticmethod
    def from_flat_array(flat: np.ndarray, timestamp: float = 0.0) -> RobotState:
        """Reconstruct a RobotState from a 52-D flat array."""
        flat = np.asarray(flat, dtype=np.float64)
        if flat.shape != (STATE_DIM,):
            raise ValueError(f"Expected shape ({STATE_DIM},), got {flat.shape}")
        return RobotState(
            q=flat[Q_SLICE].copy(),
            q_dot=flat[Q_DOT_SLICE].copy(),
            gripper_state=flat[GRIPPER_SLICE].copy(),
            left_ee_pos=flat[LEFT_EE_POS_SLICE].copy(),
            left_ee_quat=flat[LEFT_EE_QUAT_SLICE].copy(),
            right_ee_pos=flat[RIGHT_EE_POS_SLICE].copy(),
            right_ee_quat=flat[RIGHT_EE_QUAT_SLICE].copy(),
            left_ee_vel=flat[LEFT_EE_VEL_SLICE].copy(),
            right_ee_vel=flat[RIGHT_EE_VEL_SLICE].copy(),
            timestamp=timestamp,
        )

    def validate(self) -> list[str]:
        """Return list of validation warnings (empty = all good)."""
        warnings: list[str] = []
        if self.q.shape != (Q_DIM,):
            warnings.append(f"q shape {self.q.shape} != ({Q_DIM},)")
        if self.q_dot.shape != (Q_DOT_DIM,):
            warnings.append(f"q_dot shape {self.q_dot.shape} != ({Q_DOT_DIM},)")
        if self.gripper_state.shape != (GRIPPER_STATE_DIM,):
            warnings.append(f"gripper shape {self.gripper_state.shape} != ({GRIPPER_STATE_DIM},)")
        if not np.all(np.isfinite(self.to_flat_array())):
            warnings.append("NaN or Inf detected in state")
        for name, quat in [("left", self.left_ee_quat), ("right", self.right_ee_quat)]:
            norm = float(np.linalg.norm(quat))
            if abs(norm - 1.0) > 0.01:
                warnings.append(f"{name}_ee_quat norm = {norm:.4f}, expected ~1.0")
        return warnings


# ---------------------------------------------------------------------------
# State extractor
# ---------------------------------------------------------------------------


class AlexRobotState:
    """Frozen 52-D state extractor for the Alex robot.

    Extracts joint positions, velocities, gripper states, and
    end-effector poses/velocities from MuJoCo simulation data.

    Args:
        model: Compiled MuJoCo model.
    """

    def __init__(self, model: mujoco.MjModel) -> None:  # type: ignore[name-defined]
        self._model = model

        # --- Resolve arm joint IDs (reuse action_space logic) ---
        self._arm_jnt_ids = np.zeros(ARM_DIM, dtype=int)
        self._arm_dof_ids = np.zeros(ARM_DIM, dtype=int)
        for i, act_name in enumerate(ARM_ACTUATOR_NAMES):
            act_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )
            if act_id < 0:
                raise ValueError(f"Arm actuator '{act_name}' not found in model")
            jnt_id = model.actuator_trnid[act_id, 0]
            self._arm_jnt_ids[i] = jnt_id
            self._arm_dof_ids[i] = model.jnt_dofadr[jnt_id]

        # --- Resolve gripper joint IDs ---
        self._gripper_jnt_ids = np.zeros(GRIPPER_DIM, dtype=int)
        for i, act_name in enumerate(GRIPPER_ACTUATOR_NAMES):
            act_id = mujoco.mj_name2id(  # type: ignore[name-defined]
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )
            if act_id < 0:
                raise ValueError(f"Gripper actuator '{act_name}' not found")
            self._gripper_jnt_ids[i] = model.actuator_trnid[act_id, 0]

        # --- Resolve EE site IDs ---
        self._left_ee_site_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
            model, mujoco.mjtObj.mjOBJ_SITE, LEFT_EE_SITE
        )
        self._right_ee_site_id: int = mujoco.mj_name2id(  # type: ignore[name-defined]
            model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_EE_SITE
        )
        if self._left_ee_site_id < 0:
            raise ValueError(f"Site '{LEFT_EE_SITE}' not found in model")
        if self._right_ee_site_id < 0:
            raise ValueError(f"Site '{RIGHT_EE_SITE}' not found in model")

        # --- Precompute normalization ranges from model ---
        self._q_lo = np.array([model.jnt_range[j, 0] for j in self._arm_jnt_ids], dtype=np.float64)
        self._q_hi = np.array([model.jnt_range[j, 1] for j in self._arm_jnt_ids], dtype=np.float64)
        self._vel_limits = np.zeros(ARM_DIM, dtype=np.float64)
        for i, act_name in enumerate(ARM_ACTUATOR_NAMES):
            jnt_name = _ACTUATOR_TO_JOINT.get(act_name, act_name)
            self._vel_limits[i] = JOINT_VELOCITY_LIMITS.get(jnt_name, 25.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Total state dimensionality (52)."""
        return STATE_DIM

    @property
    def norm_ranges(self) -> StateNormRanges:
        """Normalization ranges derived from model and URDF limits."""
        return StateNormRanges(
            q_lo=self._q_lo.copy(),
            q_hi=self._q_hi.copy(),
            q_dot_lo=-self._vel_limits.copy(),
            q_dot_hi=self._vel_limits.copy(),
            ee_pos_lo=np.array([-0.5, -0.8, 0.5]),  # conservative workspace
            ee_pos_hi=np.array([1.0, 0.8, 1.8]),
            ee_vel_max=2.0,  # m/s, conservative upper bound
        )

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def get_state(self, data: mujoco.MjData) -> RobotState:  # type: ignore[name-defined]
        """Extract the full robot state from MuJoCo data.

        Assumes mj_forward (or mj_step) has been called so that
        site positions/orientations and qvel are current.

        Args:
            data: MjData instance with current simulation state.

        Returns:
            RobotState dataclass with all fields populated.
        """
        # Joint positions
        q = np.array([data.qpos[j] for j in self._arm_jnt_ids], dtype=np.float64)

        # Joint velocities (via DOF address, not joint ID)
        q_dot = np.array([data.qvel[d] for d in self._arm_dof_ids], dtype=np.float64)

        # Gripper state [0, 1]
        gripper_q = np.array([data.qpos[j] for j in self._gripper_jnt_ids], dtype=np.float64)
        gripper_state = np.clip(gripper_q / EZGRIPPER_JOINT_RANGE_HI, 0.0, 1.0)

        # EE poses (from site data)
        left_ee_pos, left_ee_quat = self._get_site_pose(data, self._left_ee_site_id)
        right_ee_pos, right_ee_quat = self._get_site_pose(data, self._right_ee_site_id)

        # EE linear velocities (from MuJoCo velocity computation)
        left_ee_vel = self._get_site_linear_velocity(data, self._left_ee_site_id)
        right_ee_vel = self._get_site_linear_velocity(data, self._right_ee_site_id)

        return RobotState(
            q=q,
            q_dot=q_dot,
            gripper_state=gripper_state,
            left_ee_pos=left_ee_pos,
            left_ee_quat=left_ee_quat,
            right_ee_pos=right_ee_pos,
            right_ee_quat=right_ee_quat,
            left_ee_vel=left_ee_vel,
            right_ee_vel=right_ee_vel,
            timestamp=float(data.time),
        )

    def get_flat_state(self, data: mujoco.MjData) -> np.ndarray:  # type: ignore[name-defined]
        """Return the 52-D flat state vector directly.

        Convenience method; equivalent to ``get_state(data).to_flat_array()``
        but avoids dataclass construction overhead for hot-path usage.
        """
        return self.get_state(data).to_flat_array()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize a 52-D state vector to approximately [-1, 1].

        Normalization scheme:
        - q: min-max scaled to [-1, 1] using joint limits
        - q_dot: divided by velocity limits, clipped to [-1, 1]
        - gripper: already [0, 1], mapped to [-1, 1] via (2*x - 1)
        - EE pos: min-max scaled using workspace bounds
        - EE quat: passed through unchanged (unit quaternion)
        - EE vel: divided by ee_vel_max, clipped to [-1, 1]

        Args:
            state: Raw state vector, shape (52,).

        Returns:
            Normalized state vector, shape (52,).
        """
        state = np.asarray(state, dtype=np.float64).copy()
        nr = self.norm_ranges

        # q: min-max to [-1, 1]
        q_range = nr.q_hi - nr.q_lo
        q_range = np.where(q_range < 1e-8, 1.0, q_range)  # avoid div by zero
        state[Q_SLICE] = 2.0 * (state[Q_SLICE] - nr.q_lo) / q_range - 1.0

        # q_dot: scale by velocity limits
        state[Q_DOT_SLICE] = np.clip(state[Q_DOT_SLICE] / nr.q_dot_hi, -1.0, 1.0)

        # gripper: [0,1] -> [-1,1]
        state[GRIPPER_SLICE] = 2.0 * state[GRIPPER_SLICE] - 1.0

        # EE pos: min-max per component
        pos_range = nr.ee_pos_hi - nr.ee_pos_lo
        for sl in [LEFT_EE_POS_SLICE, RIGHT_EE_POS_SLICE]:
            state[sl] = 2.0 * (state[sl] - nr.ee_pos_lo) / pos_range - 1.0

        # EE quat: leave as-is (unit quaternion)

        # EE vel: scale by max
        for sl in [LEFT_EE_VEL_SLICE, RIGHT_EE_VEL_SLICE]:
            state[sl] = np.clip(state[sl] / nr.ee_vel_max, -1.0, 1.0)

        return state

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        """Inverse of normalize(). Recovers physical units from [-1, 1]."""
        state = np.asarray(state, dtype=np.float64).copy()
        nr = self.norm_ranges

        q_range = nr.q_hi - nr.q_lo
        q_range = np.where(q_range < 1e-8, 1.0, q_range)
        state[Q_SLICE] = (state[Q_SLICE] + 1.0) / 2.0 * q_range + nr.q_lo

        state[Q_DOT_SLICE] = state[Q_DOT_SLICE] * nr.q_dot_hi

        state[GRIPPER_SLICE] = (state[GRIPPER_SLICE] + 1.0) / 2.0

        pos_range = nr.ee_pos_hi - nr.ee_pos_lo
        for sl in [LEFT_EE_POS_SLICE, RIGHT_EE_POS_SLICE]:
            state[sl] = (state[sl] + 1.0) / 2.0 * pos_range + nr.ee_pos_lo

        for sl in [LEFT_EE_VEL_SLICE, RIGHT_EE_VEL_SLICE]:
            state[sl] = state[sl] * nr.ee_vel_max

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_site_pose(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
        site_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (pos[3], quat[4]) from a site."""
        pos = data.site_xpos[site_id].copy()
        mat = data.site_xmat[site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())  # type: ignore[name-defined]
        return pos, quat

    def _get_site_linear_velocity(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
        site_id: int,
    ) -> np.ndarray:
        """Compute linear velocity of a site using mj_objectVelocity.

        Uses MuJoCo's native velocity computation which is exact and
        based on the mass-matrix Jacobian. The result array has format
        ``[rot(3), lin(3)]``; we extract only the linear component.

        Args:
            data: MjData with current state (after mj_step or mj_forward).
            site_id: Site index in the model.

        Returns:
            Linear velocity (3,) in world frame, m/s.
        """
        res = np.zeros(6)
        mujoco.mj_objectVelocity(  # type: ignore[name-defined]
            self._model,
            data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            res,
            0,  # flg_local=0 -> world frame
        )
        return res[3:].copy()  # linear velocity is in res[3:6]
