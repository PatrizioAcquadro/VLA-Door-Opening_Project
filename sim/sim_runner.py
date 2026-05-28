"""Fixed-rate simulation runner with physics substepping (Phase 1.1.5).

Enforces a 20 Hz control loop: each call to ``step()`` applies one
normalized action and advances the simulation by exactly
``substeps`` physics steps (default 25 × 0.002 s = 50 ms).

Usage::

    from sim.asset_loader import load_scene
    from sim.sim_runner import SimRunner

    model = load_scene("alex_upper_body")
    data = mujoco.MjData(model)
    runner = SimRunner(model, data)

    action = np.zeros(17)
    state = runner.step(action)              # one control tick (50 ms)
    states = runner.step_sequence(actions_16x17)  # action chunk (h=16)
"""

from __future__ import annotations

import numpy as np

from sim.action_space import AlexActionSpace
from sim.robot_state import AlexRobotState, RobotState

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


class SimRunner:
    """Run MuJoCo simulation at a fixed control rate with substepping.

    Args:
        model: Compiled MuJoCo model.
        data: MjData instance.
        control_hz: Policy control frequency (default 20.0).
        action_space: Pre-built ``AlexActionSpace``. Created internally
            if *None* (uses ``control_hz`` and default rate limit factor).
    """

    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        data: mujoco.MjData,  # type: ignore[name-defined]
        control_hz: float = 20.0,
        action_space: AlexActionSpace | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._control_hz = control_hz
        self._control_dt = 1.0 / control_hz
        self._physics_dt = model.opt.timestep

        # Compute substeps — must evenly divide
        substeps_f = self._control_dt / self._physics_dt
        self._substeps: int = round(substeps_f)
        if abs(self._substeps - substeps_f) > 1e-6:
            raise ValueError(
                f"Control period ({self._control_dt:.6f} s) is not an integer "
                f"multiple of physics timestep ({self._physics_dt:.6f} s). "
                f"Got {substeps_f:.6f} substeps."
            )
        if self._substeps < 1:
            raise ValueError(
                f"substeps must be >= 1, got {self._substeps} "
                f"(control_hz={control_hz}, physics_dt={self._physics_dt})"
            )

        self._action_space = action_space or AlexActionSpace(model, control_hz=control_hz)
        self._robot_state = AlexRobotState(model)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def substeps(self) -> int:
        """Number of physics steps per control tick."""
        return self._substeps

    @property
    def control_hz(self) -> float:
        """Policy control frequency in Hz."""
        return self._control_hz

    @property
    def control_dt(self) -> float:
        """Control period in seconds."""
        return self._control_dt

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds."""
        return float(self._data.time)

    @property
    def action_space(self) -> AlexActionSpace:
        """The action space used by this runner."""
        return self._action_space

    @property
    def robot_state(self) -> AlexRobotState:
        """The robot state extractor used by this runner."""
        return self._robot_state

    # ------------------------------------------------------------------
    # State observation
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        """Extract the current robot state (call after step or mj_forward)."""
        return self._robot_state.get_state(self._data)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> RobotState:
        """Apply one normalized action and advance physics by one control period.

        Sets ``data.ctrl`` via the action space, then calls ``mj_step()``
        for ``substeps`` iterations (default 25 × 0.002 s = 50 ms).

        Args:
            action: Normalized action, shape ``(17,)``.

        Returns:
            RobotState after stepping.
        """
        self._action_space.apply_action(action, self._data)
        for _ in range(self._substeps):
            mujoco.mj_step(self._model, self._data)  # type: ignore[name-defined]
        return self._robot_state.get_state(self._data)

    def step_sequence(self, actions: np.ndarray) -> list[RobotState]:
        """Apply a sequence of actions (e.g. an action chunk).

        Args:
            actions: Array of shape ``(N, 17)`` — one action per control tick.

        Returns:
            List of RobotState, one per action.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            return [self.step(actions)]
        return [self.step(action) for action in actions]
