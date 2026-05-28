"""Action head contracts and utilities for Phase 3.2.

Defines the frozen action chunk format, token types for loss routing, and
utility functions for chunking continuous action sequences. These contracts
govern how all Phase 3.2 components interact and how Phase 3.3's dataloader
will produce training batches.

Frozen constants (must not change without updating all downstream consumers):
    ACTION_CHUNK_SIZE = 16      # 0.8 s at 20 Hz
    ACTION_DIM = 17             # Phase 1.1.5 frozen action space
    STATE_DIM = 52              # Phase 1.1.6 frozen robot state
    TOKENS_PER_ACTION_STEP = 1  # 1 token per action step in sequence
    TOKENS_PER_STATE = 1        # 1 token for full 52-D state

Usage::

    from models.action_head import (
        ACTION_CHUNK_SIZE, ACTION_DIM, STATE_DIM,
        ActionChunkConfig, TokenType,
        chunk_actions, compute_action_context_tokens,
    )
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
from torch import Tensor

# --- Frozen constants ---------------------------------------------------------
# These mirror sim/action_space.py ACTION_DIM and sim/robot_state.py STATE_DIM
# but are defined independently to avoid importing sim code (requires mujoco).

ACTION_CHUNK_SIZE: int = 16
"""Number of action steps per chunk (0.8 s at 20 Hz control rate)."""

ACTION_DIM: int = 17
"""Frozen 17-D action space: [Δq_spine(1), Δq_left_arm(7), Δq_right_arm(7),
gripper_left(1), gripper_right(1)]."""

STATE_DIM: int = 52
"""Frozen 52-D robot state: [q(15), q_dot(15), gripper(2), left_ee_pos(3),
left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), left_ee_vel(3),
right_ee_vel(3)]."""

TOKENS_PER_ACTION_STEP: int = 1
"""Each action step occupies 1 token in the VLM sequence."""

TOKENS_PER_STATE: int = 1
"""Full 52-D state projected to 1 hidden-dim token per segment."""


# --- TokenType enum -----------------------------------------------------------


class TokenType(IntEnum):
    """Per-position token type for loss routing in the VLA sequence.

    Used to build a token_type_ids tensor that determines which loss
    function applies at each sequence position:
        TEXT   → autoregressive cross-entropy next-token loss
        IMAGE  → no loss (backbone-internal vision tokens)
        STATE  → no loss (conditioning input only)
        ACTION → flow matching velocity MSE loss
    """

    TEXT = 0
    IMAGE = 1
    STATE = 2
    ACTION = 3


# --- ActionChunkConfig dataclass ---------------------------------------------


@dataclass(frozen=True)
class ActionChunkConfig:
    """Frozen configuration for action chunks.

    All fields have defaults matching the frozen contract. Constructed
    from Hydra config at model init time via ``from_cfg()``.
    """

    chunk_size: int = ACTION_CHUNK_SIZE
    action_dim: int = ACTION_DIM
    state_dim: int = STATE_DIM
    tokens_per_action_step: int = TOKENS_PER_ACTION_STEP
    tokens_per_state: int = TOKENS_PER_STATE

    @property
    def tokens_per_chunk(self) -> int:
        """Total tokens per action chunk in the VLM sequence."""
        return self.chunk_size * self.tokens_per_action_step

    @property
    def chunk_shape(self) -> tuple[int, int]:
        """Shape of a single action chunk tensor: (chunk_size, action_dim)."""
        return (self.chunk_size, self.action_dim)

    @classmethod
    def from_cfg(cls, cfg: dict | None = None) -> ActionChunkConfig:
        """Construct from a Hydra config dict (action_head section).

        Args:
            cfg: Dict with optional keys matching field names.
                 If None, returns defaults.

        Returns:
            Frozen ActionChunkConfig.
        """
        if cfg is None:
            return cls()
        return cls(
            chunk_size=cfg.get("chunk_size", ACTION_CHUNK_SIZE),
            action_dim=cfg.get("action_dim", ACTION_DIM),
            state_dim=cfg.get("state_dim", STATE_DIM),
            tokens_per_action_step=cfg.get("tokens_per_action_step", TOKENS_PER_ACTION_STEP),
            tokens_per_state=cfg.get("tokens_per_state", TOKENS_PER_STATE),
        )


# --- FlowMatchingConfig dataclass --------------------------------------------


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Frozen configuration for the flow matching module.

    All fields have defaults matching ``configs/model/action_head.yaml``.
    Constructed from Hydra config at model init time via ``from_cfg()``.
    """

    n_denoising_steps: int = 10
    """Number of ODE integration steps K for inference denoising."""

    solver: str = "euler"
    """ODE solver: ``"euler"`` | ``"midpoint"`` | ``"rk4"``."""

    sigma_min: float = 0.001
    """Lower bound on sampled timestep (avoids exact t=0 during training)."""

    time_sampling: str = "beta"
    """Timestep sampling distribution: ``"beta"`` | ``"uniform"``."""

    time_alpha: float = 1.5
    """Beta distribution alpha parameter (EO-1 default)."""

    time_beta: float = 1.0
    """Beta distribution beta parameter (EO-1 default)."""

    time_min: float = 0.001
    """Minimum sampled timestep after scaling."""

    time_max: float = 0.999
    """Maximum sampled timestep after scaling."""

    @classmethod
    def from_cfg(cls, cfg: dict | None = None) -> FlowMatchingConfig:
        """Construct from a Hydra config dict (flow_matching section).

        Args:
            cfg: Dict with optional keys matching field names.
                 If None, returns defaults.

        Returns:
            Frozen FlowMatchingConfig.
        """
        if cfg is None:
            return cls()
        return cls(
            n_denoising_steps=cfg.get("n_denoising_steps", 10),
            solver=cfg.get("solver", "euler"),
            sigma_min=cfg.get("sigma_min", 0.001),
            time_sampling=cfg.get("time_sampling", "beta"),
            time_alpha=cfg.get("time_alpha", 1.5),
            time_beta=cfg.get("time_beta", 1.0),
            time_min=cfg.get("time_min", 0.001),
            time_max=cfg.get("time_max", 0.999),
        )


# --- FlowMatchingModule -------------------------------------------------------


class FlowMatchingModule(nn.Module):
    """Conditional flow matching math for action generation.

    Implements the standard OT-CFM formulation (Lipman et al., 2023) adapted
    for the EO-1-style VLA action head:

    - **Time convention**: t=0 → noise, t=1 → data  (NOT EO-1's reversed convention)
    - **OT path**: ``x_t = (1-t) * noise + t * x_data``
    - **Target velocity**: ``u_t = x_data - noise``  (constant along OT path, independent of t)
    - **Time sampling**: Beta(1.5, 1.0) following EO-1 (biases toward noisier timesteps)
    - **Solvers**: Euler (default), midpoint, RK4 — all exact for constant velocity fields

    Has no learnable parameters. All computation is deterministic given config.

    Args:
        config: FlowMatchingConfig with solver and time sampling parameters.
                Defaults to ``FlowMatchingConfig()`` if None.

    Example::

        fm = FlowMatchingModule()
        # Training:
        t = fm.sample_timestep(B, device)               # (B, 1, 1)
        x_t = fm.interpolate(x_data, noise, t)          # (B, 16, 17)
        u_t = fm.target_velocity(x_data, noise)         # (B, 16, 17)
        loss = fm.loss(pred_velocity, u_t, mask)        # scalar
        # Inference:
        x_pred = fm.denoise(predict_fn, (B, 16, 17))   # (B, 16, 17)
    """

    def __init__(self, config: FlowMatchingConfig | None = None) -> None:
        super().__init__()
        self.config = config or FlowMatchingConfig()

    def sample_timestep(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Sample flow matching timesteps for training.

        Samples t from Beta(alpha, beta) scaled to [time_min, time_max], or
        from Uniform[time_min, time_max] if ``config.time_sampling == "uniform"``.

        Args:
            batch_size: Number of samples (B).
            device: Target device for the returned tensor.

        Returns:
            Timestep tensor of shape ``(B, 1, 1)`` — broadcasts over
            ``(B, chunk_size, action_dim)`` without explicit expansion.
        """
        cfg = self.config
        if cfg.time_sampling == "beta":
            dist = torch.distributions.Beta(
                torch.tensor(cfg.time_alpha, dtype=torch.float32),
                torch.tensor(cfg.time_beta, dtype=torch.float32),
            )
            t = dist.sample((batch_size,))  # (B,) in [0, 1]
        else:
            t = torch.rand(batch_size)  # uniform [0, 1]

        # Scale to [time_min, time_max]
        t = cfg.time_min + t * (cfg.time_max - cfg.time_min)
        t = t.reshape(batch_size, 1, 1)  # (B, 1, 1)

        if device is not None:
            t = t.to(device)
        return t

    def interpolate(self, x_data: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Interpolate between noise and data along the OT straight-line path.

        Standard OT-CFM: ``x_t = (1-t) * noise + t * x_data``

        - At t=0: pure noise
        - At t=1: clean data

        Args:
            x_data: Clean action chunk, shape ``(B, C, D)``.
            noise: Gaussian noise, shape ``(B, C, D)``.
            t: Timestep in [0, 1], broadcastable to ``(B, C, D)`` — typically
               ``(B, 1, 1)`` from ``sample_timestep()``.

        Returns:
            Noisy actions ``x_t`` of shape ``(B, C, D)``.
        """
        return (1.0 - t) * noise + t * x_data

    def target_velocity(self, x_data: Tensor, noise: Tensor) -> Tensor:
        """Compute the target velocity field for CFM training.

        Along the straight-line OT path the target velocity is constant
        (independent of t): ``u_t = x_data - noise``.

        Args:
            x_data: Clean action chunk, shape ``(B, C, D)``.
            noise: Gaussian noise, shape ``(B, C, D)``.

        Returns:
            Target velocity ``u_t`` of shape ``(B, C, D)``.
        """
        return x_data - noise

    def loss(
        self,
        pred_velocity: Tensor,
        target_velocity: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute masked MSE loss between predicted and target velocity.

        Per-step loss = mean over action_dim of squared error.
        Final scalar = masked mean over all valid (batch, step) positions.

        Args:
            pred_velocity: Predicted velocity, shape ``(B, C, D)``.
            target_velocity: Target velocity, shape ``(B, C, D)``.
            mask: Binary mask, shape ``(B, C)`` — 1 for valid steps, 0 for
                  padding. If None, all positions are valid.

        Returns:
            Scalar loss tensor.
        """
        diff_sq = (pred_velocity - target_velocity) ** 2  # (B, C, D)
        per_step = diff_sq.mean(dim=-1)  # (B, C) — mean over action_dim

        if mask is None:
            return per_step.mean()

        masked_sum = (per_step * mask).sum()
        n_valid = mask.sum().clamp(min=1.0)
        return masked_sum / n_valid

    def denoise(
        self,
        predict_fn: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        K: int | None = None,
        device: torch.device | str | None = None,
        x_init: Tensor | None = None,
    ) -> Tensor:
        """Denoise from noise to action by integrating the velocity ODE.

        Integrates ``dx = v_θ(x_t, t) dt`` from t=0 to t=1 using K discrete
        Euler/midpoint/RK4 steps. Starting point is sampled from N(0, I) unless
        ``x_init`` is provided.

        Args:
            predict_fn: Callable ``(x_t, t) → velocity`` where x_t has shape
                        matching ``shape`` and t has shape ``(B, 1, 1)``.
            shape: Shape of the action tensor to generate, e.g. ``(B, 16, 17)``.
            K: Number of integration steps. Defaults to ``config.n_denoising_steps``.
            device: Device for the initial noise tensor.
            x_init: Optional initial noise tensor of shape ``shape``. When
                    provided, ``shape`` and ``device`` are ignored for init.
                    Useful for deterministic testing.

        Returns:
            Denoised action chunk of shape ``shape``.
        """
        if K is None:
            K = self.config.n_denoising_steps
        solver = self.config.solver

        if x_init is not None:
            x = x_init.clone()
        else:
            x = torch.randn(shape, device=device)

        dt = 1.0 / K
        B = shape[0]

        for k in range(K):
            t_scalar = k * dt
            t = torch.full((B, 1, 1), t_scalar, dtype=x.dtype, device=x.device)

            if solver == "euler":
                v = predict_fn(x, t)
                x = x + dt * v

            elif solver == "midpoint":
                v1 = predict_fn(x, t)
                t_mid = t + dt / 2.0
                x_mid = x + (dt / 2.0) * v1
                v2 = predict_fn(x_mid, t_mid)
                x = x + dt * v2

            elif solver == "rk4":
                k1 = predict_fn(x, t)
                k2 = predict_fn(x + (dt / 2.0) * k1, t + dt / 2.0)
                k3 = predict_fn(x + (dt / 2.0) * k2, t + dt / 2.0)
                k4 = predict_fn(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(f"Unknown solver: {solver!r}. Choose euler | midpoint | rk4")

        return x


# --- RobotStateProjector ------------------------------------------------------


class RobotStateProjector(nn.Module):
    """Projects normalized 52-D robot state into a single hidden-dim token.

    Architecture: LayerNorm(state_dim) → Linear(state_dim, H) → SiLU → Linear(H, H)
    Output: (B, 1, H) — single token for VLM sequence injection.

    EO-1 uses a simple Linear(state_dim, hidden_size). We add LayerNorm and a
    2-layer MLP because the 52-D state has heterogeneous components (joint
    positions, velocities, quaternions, EE poses) that benefit from normalization
    before projection.

    Parameter count for H=2560:
        LayerNorm(52):       104  (weight + bias)
        Linear(52, 2560):    135,680
        Linear(2560, 2560):  6,556,160
        Total:               ~6.7M (negligible compared to 4B backbone)

    Args:
        state_dim: Input state dimension (default: STATE_DIM = 52).
        hidden_dim: Output dimension matching backbone hidden size (default: 2560).
        activation: Nonlinearity between MLP layers: ``"silu"`` | ``"gelu"`` | ``"relu"``.

    Example::

        projector = RobotStateProjector(hidden_dim=2560)
        state = torch.randn(4, 52)       # (B, 52) normalized
        token = projector(state)          # (B, 1, 2560)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 2560,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(state_dim)

        _act_map: dict[str, type[nn.Module]] = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
        }
        if activation not in _act_map:
            raise ValueError(f"Unknown activation {activation!r}. Choose silu | gelu | relu")
        act_cls = _act_map[activation]

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, robot_state: Tensor) -> Tensor:
        """Project robot state to a hidden-dim token.

        Args:
            robot_state: Normalized state vector, shape ``(B, state_dim)``.

        Returns:
            State token of shape ``(B, 1, hidden_dim)`` for VLM sequence injection.
        """
        x = self.norm(robot_state)  # (B, state_dim)
        projected: Tensor = self.mlp(x)  # (B, hidden_dim)
        return projected.unsqueeze(1)  # (B, 1, hidden_dim)

    @classmethod
    def from_cfg(cls, cfg: dict | None, hidden_dim: int) -> RobotStateProjector:
        """Construct from a Hydra config dict (action_head section).

        Args:
            cfg: Dict with action_head config keys. If None, uses defaults.
            hidden_dim: ``backbone.hidden_size`` resolved at runtime (the config
                        stores ``projector.hidden_dim: null`` to signal this).

        Returns:
            Constructed ``RobotStateProjector``.
        """
        if cfg is None:
            return cls(hidden_dim=hidden_dim)
        proj: dict = cfg.get("projector", {}) or {}
        return cls(
            state_dim=cfg.get("state_dim", STATE_DIM),
            hidden_dim=proj.get("hidden_dim") or hidden_dim,
            activation=proj.get("activation", "silu"),
        )


# --- Sinusoidal timestep embedding -------------------------------------------


def sinusoidal_timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Compute sinusoidal embedding for a scalar flow matching timestep.

    Standard positional encoding formula applied to the flow matching timestep
    t ∈ [0, 1]:

        embed[2i]   = sin(t / 10000^(2i / dim))
        embed[2i+1] = cos(t / 10000^(2i / dim))

    Parameter-free — no learnable weights. Produces a ``dim``-dimensional vector
    that encodes denoising progress.  Following EO-1's ``embed_suffix()`` timestep
    embedding convention.

    Args:
        t: Timestep tensor of shape ``(B, 1)`` with values in ``[0, 1]``.
        dim: Embedding dimension (must be even and positive).

    Returns:
        Embedding tensor of shape ``(B, dim)``.

    Raises:
        ValueError: If ``dim`` is not positive or is odd.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be a positive even integer, got {dim}")

    half = dim // 2
    # Frequency bands: 10000^(2i/dim) for i in [0, half)
    freqs = torch.pow(
        10000.0,
        torch.arange(half, dtype=torch.float32, device=t.device) / half,
    )  # (half,)

    # t: (B, 1), freqs: (half,) → (B, half)
    args = t.float() / freqs  # broadcast: (B, half)

    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# --- NoisyActionProjector -----------------------------------------------------


class NoisyActionProjector(nn.Module):
    """Projects noisy action chunk + flow matching timestep into hidden-dim tokens.

    Implements EO-1's ``embed_suffix()`` pattern adapted for our 17-D action space
    and chunk size 16:

    1. Sinusoidal embedding of the scalar timestep t → ``(B, d_t)``
    2. Expand across chunk steps → ``(B, chunk_size, d_t)``
    3. Concatenate with noisy actions → ``(B, chunk_size, action_dim + d_t)``
    4. 2-layer MLP → ``(B, chunk_size, H)``

    Each of the 16 action steps becomes one hidden-dimension token. The entire
    chunk shares a single denoising timestep (broadcast across steps).

    Parameter count (H=2560, d_t=256):
        Linear(17+256, 2560) = 701,440 + bias 2,560
        Linear(2560, 2560) = 6,553,600 + bias 2,560
        Total: ~7,257,600

    Args:
        action_dim: Action space dimension (default: ``ACTION_DIM = 17``).
        hidden_dim: Output dimension matching backbone hidden size (default: 2560).
        timestep_embed_dim: Sinusoidal embedding dimension (default: 256).
        activation: Nonlinearity between MLP layers: ``"silu"`` | ``"gelu"`` | ``"relu"``.

    Example::

        projector = NoisyActionProjector(hidden_dim=2560)
        noisy_actions = torch.randn(4, 16, 17)  # (B, chunk_size, action_dim)
        t = torch.rand(4, 1)                    # (B, 1) timestep
        tokens = projector(noisy_actions, t)    # (B, 16, 2560)
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 2560,
        timestep_embed_dim: int = 256,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.timestep_embed_dim = timestep_embed_dim

        _act_map: dict[str, type[nn.Module]] = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
        }
        if activation not in _act_map:
            raise ValueError(f"Unknown activation {activation!r}. Choose silu | gelu | relu")
        act_cls = _act_map[activation]

        in_dim = action_dim + timestep_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, noisy_actions: Tensor, t: Tensor) -> Tensor:
        """Project noisy actions and timestep to hidden-dim tokens.

        Args:
            noisy_actions: Noisy action chunk, shape ``(B, chunk_size, action_dim)``.
            t: Flow matching timestep, shape ``(B, 1)``.

        Returns:
            Action tokens of shape ``(B, chunk_size, hidden_dim)`` for VLM
            sequence injection.
        """
        B, chunk_size, _ = noisy_actions.shape

        # Sinusoidal timestep embedding: (B, 1) → (B, d_t)
        t_embed = sinusoidal_timestep_embedding(t, self.timestep_embed_dim)  # (B, d_t)

        # Broadcast across chunk steps: (B, d_t) → (B, chunk_size, d_t)
        t_embed = t_embed.unsqueeze(1).expand(B, chunk_size, self.timestep_embed_dim)

        # Concatenate along feature dim: (B, chunk_size, action_dim + d_t)
        x = torch.cat([noisy_actions.float(), t_embed], dim=-1)

        out: Tensor = self.mlp(x)
        return out  # (B, chunk_size, hidden_dim)

    @classmethod
    def from_cfg(cls, cfg: dict | None, hidden_dim: int) -> NoisyActionProjector:
        """Construct from a Hydra config dict (action_head section).

        Args:
            cfg: Dict with action_head config keys. If None, uses defaults.
            hidden_dim: ``backbone.hidden_size`` resolved at runtime.

        Returns:
            Constructed ``NoisyActionProjector``.
        """
        if cfg is None:
            return cls(hidden_dim=hidden_dim)
        proj: dict = cfg.get("projector", {}) or {}
        return cls(
            action_dim=cfg.get("action_dim", ACTION_DIM),
            hidden_dim=proj.get("hidden_dim") or hidden_dim,
            timestep_embed_dim=cfg.get("timestep_embed_dim", 256),
            activation=proj.get("activation", "silu"),
        )


# --- ActionOutputHead ---------------------------------------------------------


class ActionOutputHead(nn.Module):
    """Decodes backbone hidden states at action positions to velocity predictions.

    Implements EO-1's 2-layer action head MLP adapted for our 17-D action space:

        Linear(H, H) + SiLU + Linear(H, action_dim)

    Intentionally lightweight — the backbone transformer has already performed
    the heavy reasoning. This head simply projects from hidden space back to
    the action space.

    Parameter count (H=2560, action_dim=17):
        Linear(2560, 2560) = 6,553,600 + bias 2,560
        Linear(2560, 17)   = 43,520    + bias 17
        Total: ~6,599,697

    Args:
        action_dim: Action space dimension (default: ``ACTION_DIM = 17``).
        hidden_dim: Input dimension matching backbone hidden size (default: 2560).
        activation: Nonlinearity between MLP layers: ``"silu"`` | ``"gelu"`` | ``"relu"``.

    Example::

        head = ActionOutputHead(hidden_dim=2560)
        hidden = torch.randn(4, 16, 2560)  # (B, chunk_size, H)
        velocity = head(hidden)            # (B, 16, 17)
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 2560,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        _act_map: dict[str, type[nn.Module]] = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
        }
        if activation not in _act_map:
            raise ValueError(f"Unknown activation {activation!r}. Choose silu | gelu | relu")
        act_cls = _act_map[activation]

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Decode backbone hidden states to velocity predictions.

        Args:
            hidden_states: Backbone output at action positions,
                           shape ``(B, chunk_size, hidden_dim)``.

        Returns:
            Velocity predictions of shape ``(B, chunk_size, action_dim)``.
        """
        out: Tensor = self.mlp(hidden_states)
        return out

    @classmethod
    def from_cfg(cls, cfg: dict | None, hidden_dim: int) -> ActionOutputHead:
        """Construct from a Hydra config dict (action_head section).

        Args:
            cfg: Dict with action_head config keys. If None, uses defaults.
            hidden_dim: ``backbone.hidden_size`` resolved at runtime.

        Returns:
            Constructed ``ActionOutputHead``.
        """
        if cfg is None:
            return cls(hidden_dim=hidden_dim)
        proj: dict = cfg.get("projector", {}) or {}
        return cls(
            action_dim=cfg.get("action_dim", ACTION_DIM),
            hidden_dim=proj.get("hidden_dim") or hidden_dim,
            activation=proj.get("activation", "silu"),
        )


# --- Chunking utilities -------------------------------------------------------


def chunk_actions(
    actions: Tensor,
    chunk_size: int = ACTION_CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Split a variable-length action sequence into fixed-size chunks.

    The last chunk is zero-padded if it has fewer than ``chunk_size`` steps.
    A binary mask indicates which steps are valid (1) vs padding (0).

    Args:
        actions: Action tensor of shape ``(n_steps, action_dim)``.
        chunk_size: Number of steps per chunk.

    Returns:
        Tuple of:
            chunks: ``(n_chunks, chunk_size, action_dim)`` — zero-padded.
            masks:  ``(n_chunks, chunk_size)`` — 1 for valid, 0 for padding.

    Raises:
        ValueError: If actions has wrong number of dimensions.
    """
    if actions.ndim != 2:
        raise ValueError(f"Expected 2-D tensor (n_steps, action_dim), got {actions.ndim}-D")

    n_steps, action_dim = actions.shape

    if n_steps == 0:
        return (
            actions.new_zeros(0, chunk_size, action_dim),
            actions.new_zeros(0, chunk_size),
        )

    n_chunks = math.ceil(n_steps / chunk_size)
    padded_len = n_chunks * chunk_size

    # Pad actions to multiple of chunk_size
    if padded_len > n_steps:
        padding = actions.new_zeros(padded_len - n_steps, action_dim)
        padded = torch.cat([actions, padding], dim=0)
    else:
        padded = actions

    # Reshape into chunks
    chunks = padded.view(n_chunks, chunk_size, action_dim)

    # Build mask: 1 for valid steps, 0 for padding
    mask = actions.new_ones(padded_len)
    if padded_len > n_steps:
        mask[n_steps:] = 0.0
    mask = mask.view(n_chunks, chunk_size)

    return chunks, mask


def chunk_actions_batched(
    actions: Tensor,
    masks: Tensor,
    chunk_size: int = ACTION_CHUNK_SIZE,
) -> tuple[Tensor, Tensor]:
    """Batch-level chunking for variable-length action sequences.

    Each sample in the batch may have a different number of valid steps
    (indicated by ``masks``). All samples are padded to the same number
    of chunks (determined by the longest sequence in the batch).

    Args:
        actions: ``(B, max_steps, action_dim)`` — zero-padded batch.
        masks: ``(B, max_steps)`` — 1 for valid steps, 0 for padding.
        chunk_size: Number of steps per chunk.

    Returns:
        Tuple of:
            chunks: ``(B, max_chunks, chunk_size, action_dim)``
            chunk_masks: ``(B, max_chunks, chunk_size)``
    """
    B, max_steps, action_dim = actions.shape
    n_chunks = math.ceil(max_steps / chunk_size)
    padded_len = n_chunks * chunk_size

    # Pad to multiple of chunk_size along step dimension
    if padded_len > max_steps:
        act_pad = actions.new_zeros(B, padded_len - max_steps, action_dim)
        actions_padded = torch.cat([actions, act_pad], dim=1)
        mask_pad = masks.new_zeros(B, padded_len - max_steps)
        masks_padded = torch.cat([masks, mask_pad], dim=1)
    else:
        actions_padded = actions
        masks_padded = masks

    chunks = actions_padded.view(B, n_chunks, chunk_size, action_dim)
    chunk_masks = masks_padded.view(B, n_chunks, chunk_size)

    return chunks, chunk_masks


def compute_action_context_tokens(
    n_steps: int,
    chunk_size: int = ACTION_CHUNK_SIZE,
) -> int:
    """Compute total action tokens in the VLM sequence for a given step count.

    Each step occupies 1 token (``TOKENS_PER_ACTION_STEP = 1``), and steps
    are grouped into chunks of ``chunk_size``. The last chunk is zero-padded,
    so the token count is always a multiple of ``chunk_size``.

    Args:
        n_steps: Number of action steps in the episode.
        chunk_size: Steps per chunk.

    Returns:
        Total number of action tokens in the sequence.
    """
    if n_steps <= 0:
        return 0
    n_chunks = math.ceil(n_steps / chunk_size)
    return n_chunks * chunk_size * TOKENS_PER_ACTION_STEP
