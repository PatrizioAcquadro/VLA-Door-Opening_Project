"""Runtime connection manager for hybrid LEGO retention (Phase 1.2.2b).

Monitors brick pair engagement and activates/deactivates pre-declared
weld equality constraints at runtime. Only used in ``spec_proxy`` mode.

The engagement gate requires:
  1. Z-threshold: top brick body Z within margin of engaged position
  2. XY alignment: lateral offset within tolerance
  3. Sustained contact: both conditions met for ``engage_min_steps`` consecutive steps

Release uses hysteresis:
  - Displacement from welded pose exceeds threshold for ``release_dwell_steps``

Usage::

    from sim.lego.connection_manager import ConnectionManager

    mgr = ConnectionManager(model, data)
    mgr.register_pair("base_2x2", "top_2x2", eq_id=6)

    # In stepping loop:
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        mgr.update()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from sim.lego.constants import BRICK_HEIGHT, STUD_HALF_HEIGHT

logger = logging.getLogger(__name__)

# Default expected Z offset between stacked brick body origins.
# Uses STUD_HALF_HEIGHT (stud center) to match run_insertion() engagement threshold.
_DEFAULT_Z_OFFSET: float = BRICK_HEIGHT + STUD_HALF_HEIGHT


@dataclass
class BrickPairState:
    """Tracks engagement state for one brick pair."""

    body1_name: str
    body2_name: str
    body1_id: int
    body2_id: int
    eq_id: int

    # Engagement tracking
    engaged_steps: int = 0
    weld_active: bool = False
    welded_relpose: np.ndarray | None = field(default=None, repr=False)

    # Release tracking
    release_steps: int = 0

    # Expected geometry
    expected_z_offset: float = _DEFAULT_Z_OFFSET


class ConnectionManager:
    """Manages weld constraint activation/deactivation for brick pairs.

    Args:
        model: MuJoCo model with pre-declared (inactive) weld constraints.
        data: MuJoCo data.
        engage_xy_tol_m: XY alignment tolerance for engagement gate (meters).
        engage_min_steps: Consecutive engaged steps before weld activation.
        engage_z_margin_m: Z margin below expected engaged position (meters).
        release_displacement_m: Displacement from welded pose to trigger release.
        release_dwell_steps: Consecutive over-threshold steps before release.
    """

    def __init__(
        self,
        model,
        data,
        engage_xy_tol_m: float = 0.0005,
        engage_min_steps: int = 50,
        engage_z_margin_m: float = 0.0002,
        release_displacement_m: float = 0.002,
        release_dwell_steps: int = 25,
    ) -> None:
        self._model = model
        self._data = data
        self._pairs: list[BrickPairState] = []

        self._engage_xy_tol = engage_xy_tol_m
        self._engage_min_steps = engage_min_steps
        self._engage_z_margin = engage_z_margin_m
        self._release_displacement = release_displacement_m
        self._release_dwell_steps = release_dwell_steps

    def register_pair(
        self,
        body1_name: str,
        body2_name: str,
        eq_id: int,
        expected_z_offset: float | None = None,
    ) -> None:
        """Register a brick pair for engagement monitoring.

        Args:
            body1_name: Fixed/lower body name.
            body2_name: Free/upper body name.
            eq_id: Index of pre-declared weld constraint in model.
            expected_z_offset: Expected Z offset of body2 relative to body1
                when engaged. Defaults to ``BRICK_HEIGHT + STUD_HEIGHT``.
        """
        import mujoco

        body1_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

        if body1_id < 0:
            raise ValueError(f"Body '{body1_name}' not found in model")
        if body2_id < 0:
            raise ValueError(f"Body '{body2_name}' not found in model")
        if eq_id < 0 or eq_id >= self._model.neq:
            raise ValueError(f"Equality ID {eq_id} out of range [0, {self._model.neq})")

        z_off = expected_z_offset if expected_z_offset is not None else _DEFAULT_Z_OFFSET

        pair = BrickPairState(
            body1_name=body1_name,
            body2_name=body2_name,
            body1_id=body1_id,
            body2_id=body2_id,
            eq_id=eq_id,
            expected_z_offset=z_off,
        )

        # Ensure constraint starts disabled
        self._data.eq_active[eq_id] = 0

        self._pairs.append(pair)
        logger.debug("Registered pair %s <-> %s (eq_id=%d)", body1_name, body2_name, eq_id)

    def update(self) -> list[str]:
        """Check all pairs and activate/deactivate welds as needed.

        Returns:
            List of transition log messages (empty if no transitions).
        """
        messages: list[str] = []
        for pair in self._pairs:
            if not pair.weld_active:
                msg = self._update_engagement(pair)
            else:
                msg = self._update_release(pair)
            if msg:
                messages.append(msg)
        return messages

    def _update_engagement(self, pair: BrickPairState) -> str | None:
        """Check engagement and potentially activate weld."""
        if self._check_engagement(pair):
            pair.engaged_steps += 1
            if pair.engaged_steps >= self._engage_min_steps:
                self._activate_weld(pair)
                msg = (
                    f"ACTIVATE weld eq_id={pair.eq_id} "
                    f"for {pair.body1_name} <-> {pair.body2_name} "
                    f"(after {pair.engaged_steps} steps)"
                )
                logger.info(msg)
                return msg
        else:
            pair.engaged_steps = 0
        return None

    def _update_release(self, pair: BrickPairState) -> str | None:
        """Check release condition and potentially deactivate weld."""
        if self._check_release(pair):
            pair.release_steps += 1
            if pair.release_steps >= self._release_dwell_steps:
                self._deactivate_weld(pair)
                msg = (
                    f"DEACTIVATE weld eq_id={pair.eq_id} "
                    f"for {pair.body1_name} <-> {pair.body2_name} "
                    f"(after {pair.release_steps} release steps)"
                )
                logger.info(msg)
                return msg
        else:
            pair.release_steps = 0
        return None

    def _check_engagement(self, pair: BrickPairState) -> bool:
        """Check Z-threshold + XY alignment for a pair."""
        pos1 = self._data.xpos[pair.body1_id]
        pos2 = self._data.xpos[pair.body2_id]

        dz = pos2[2] - pos1[2]
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        # Z check: body2 is at or below expected engaged height (with margin)
        z_engaged = dz <= pair.expected_z_offset + self._engage_z_margin

        # XY check: lateral alignment within tolerance
        xy_dist = math.sqrt(dx * dx + dy * dy)
        xy_aligned = xy_dist <= self._engage_xy_tol

        return z_engaged and xy_aligned

    def _compute_current_relpose(self, pair: BrickPairState) -> np.ndarray:
        """Compute relative pose of body2 w.r.t. body1 in body1's frame.

        Returns:
            7-element array: [rel_pos(3), rel_quat(4)] where quat is [w,x,y,z].
        """
        import mujoco

        pos1 = self._data.xpos[pair.body1_id].copy()
        pos2 = self._data.xpos[pair.body2_id].copy()
        quat1 = self._data.xquat[pair.body1_id].copy()
        quat2 = self._data.xquat[pair.body2_id].copy()

        # Relative position in world frame
        rel_pos_world = pos2 - pos1

        # Rotate into body1's frame: rel_pos_local = R1^T * rel_pos_world
        neg_q1 = np.zeros(4)
        mujoco.mju_negQuat(neg_q1, quat1)

        rel_pos_local = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos_local, rel_pos_world, neg_q1)

        # Relative quaternion: q_rel = q1^(-1) * q2
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, neg_q1, quat2)

        return np.concatenate([rel_pos_local, rel_quat])

    def _activate_weld(self, pair: BrickPairState) -> None:
        """Set weld relpose to current relative pose and enable constraint."""
        relpose = self._compute_current_relpose(pair)

        # Write relpose into eq_data: anchor(3) + relpose_pos(3) + relpose_quat(4) + torquescale(1)
        # For weld: eq_data layout is [anchor_x, anchor_y, anchor_z,
        #                              relpose_x, relpose_y, relpose_z,
        #                              relpose_qw, relpose_qx, relpose_qy, relpose_qz,
        #                              torquescale]
        self._model.eq_data[pair.eq_id, 3:6] = relpose[:3]  # position
        self._model.eq_data[pair.eq_id, 6:10] = relpose[3:7]  # quaternion

        self._data.eq_active[pair.eq_id] = 1
        pair.weld_active = True
        pair.welded_relpose = relpose.copy()
        pair.release_steps = 0

    def _check_release(self, pair: BrickPairState) -> bool:
        """Check if displacement from welded pose exceeds threshold."""
        if pair.welded_relpose is None:
            return False

        current = self._compute_current_relpose(pair)
        displacement = float(np.linalg.norm(current[:3] - pair.welded_relpose[:3]))
        return displacement > self._release_displacement

    def _deactivate_weld(self, pair: BrickPairState) -> None:
        """Disable the weld constraint."""
        self._data.eq_active[pair.eq_id] = 0
        pair.weld_active = False
        pair.engaged_steps = 0
        pair.release_steps = 0
        pair.welded_relpose = None

    @property
    def active_connections(self) -> int:
        """Number of currently active weld constraints."""
        return sum(1 for p in self._pairs if p.weld_active)

    @property
    def pairs(self) -> list[BrickPairState]:
        """All registered pairs."""
        return list(self._pairs)

    def reset(self) -> None:
        """Reset all pair states and deactivate all active welds.

        Call this between episodes to clear engagement counters and
        disable any weld constraints that were activated during the prior episode.
        """
        for pair in self._pairs:
            if pair.weld_active:
                self._data.eq_active[pair.eq_id] = 0
            pair.engaged_steps = 0
            pair.weld_active = False
            pair.welded_relpose = None
            pair.release_steps = 0

    def get_pair_state(self, body1_name: str, body2_name: str) -> BrickPairState | None:
        """Look up state for a specific pair."""
        for pair in self._pairs:
            if pair.body1_name == body1_name and pair.body2_name == body2_name:
                return pair
        return None
