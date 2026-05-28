# Phase 3.2 — Action Head Implementation (5 days)

**Goal:** Implement and validate the flow matching action head — comprising the flow matching module, robot state projector, noisy action projector, action output head, and chunked action generation — that attaches to the Phase 3.1 VLM backbone and produces continuous 17-D robot action trajectories, reusing and adapting the EO-1 action head code wherever that code is available in the project copy, so that Phase 3.3 can wire the complete VLA model into the training loop with a known, tested action generation pipeline.

**Phase 2.3 data dependency note (added during 3.2.0 implementation):**
Phase 2.3 (interleaved VLA data sequences) has not been conducted yet. This does NOT block Phase 3.2: all subphases (3.2.0–3.2.5) are "model-only scope" and test exclusively with synthetic tensors. No images, text descriptions, or HDF5 action data from Phase 2.3 or the EO-1 dataset are needed. References to Phase 2.3 data format below (JSONL sequences, `action_ref`, HDF5) define the *design contract* that the Phase 3.3 dataloader will eventually produce — they are not runtime dependencies for Phase 3.2. When Phase 2.3 is conducted, the dataloader (Phase 3.3) will produce batches matching the contracts defined here.

**EO-1 code reuse note (added during 3.2.0 implementation):**
The EO-1 codebase (`eo/model/modeling_eo1.py`, `eo/constants.py`, `eo/model/configuration_eo1.py`) was inspected for reusable patterns. Key findings: EO-1 uses reversed time convention (t=0→data, t=1→noise); we follow standard CFM (t=0→noise, t=1→data). EO-1 keeps action head components in float32 for numerical stability. EO-1 uses Beta(1.5, 1.0) time sampling. EO-1's `embed_prefix`/`embed_suffix` split pattern is adopted for our prefix/suffix injection approach.

**Fixed upstream decisions (from 1.1–1.2 + 2.1–2.3 + 3.1):**
- **Action space:** Frozen 17-D `[Δq_spine(1), Δq_left_arm(7), Δq_right_arm(7), gripper_left(1), gripper_right(1)]`, arm actions normalized `[-1, 1]`, gripper actions absolute `[0, 1]`
- **Robot state:** Frozen 52-D normalized vector `[q(15), q_dot(15), gripper(2), left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), left_ee_vel(3), right_ee_vel(3)]`
- **Control rate:** 20 Hz (50 ms per action step, 25 physics substeps at 0.002 s)
- **VLM backbone:** Qwen3.5-4B loaded and profiled in Phase 3.1 — provides `get_hidden_states()` → `(B, seq_len, hidden_size)`, `hidden_size` property, `processor`, freeze/unfreeze control
- **Context window:** 8192 tokens; vision token count per 320×320 image measured in Phase 3.1.2
- **Memory budget:** Action head VRAM headroom measured in Phase 3.1.4 (`logs/vlm_memory/action_head_budget.json`)
- **Training data:** Phase 2.3 reference-based JSONL sequences with `action_ref: {episode_seed, step_range}` → continuous 17-D float32 actions in HDF5; ~200 steps per single-placement episode
- **Target architecture (EO-1):** Transfusion-style unified decoder — autoregressive next-token loss on text positions + conditional flow matching loss on continuous action positions, within a single forward pass through the shared backbone

**Key Phase 3.2 stance:**
- **EO-1 code reuse first.** If EO-1 provides an action head, flow matching module, state/action projectors, or chunked decoding logic, reuse and adapt that code rather than building from scratch. The first step of every subphase is to inspect the EO-1 codebase for existing implementations. Only introduce new modules where the inspected EO-1 code does not provide a realistic integration path.
- **Backbone as denoiser (Transfusion-style).** Noisy action tokens are injected into the VLM sequence alongside vision and language tokens. The backbone transformer itself serves as the denoising network. This maximizes conditioning from visual and language context and matches the EO-1 architecture.
- **Chunk size 16 as default.** 16 action steps × 50 ms = 0.8 s prediction horizon. Each step becomes one token in the sequence (16 tokens per chunk). This fits within the 8K context window with large headroom and provides per-step attention resolution within the chunk.
- **Model-only scope.** This phase delivers PyTorch modules that can be called with prepared tensors. It does **not** build the training dataloader, the distributed training loop, or the data pipeline integration. Those are Phase 3.3 (training integration) concerns. Tests in this phase use synthetic tensors.
- **Backward compatible.** The Phase 3.1 VLM backbone wrapper and all existing model paths (`model=base`, `model=large`, `model=vlm`) continue to work unchanged. The action head is a new additive component.

**Critical gap this phase closes:**
Phase 3.1 delivers a VLM backbone that processes vision and language and produces hidden states — but it has no mechanism for generating robot actions. The existing `TransformerModel` uses MSE state prediction, which is architecturally incompatible with the EO-1 VLA approach. Without Phase 3.2, there is no way to convert the backbone's contextual understanding into the 17-D continuous action trajectories that the robot controller consumes. The action head is the bridge between perception-language reasoning and physical manipulation.

---

## 3.2.0) Action Chunk Contract & Sequence Integration

### What we will do
Define the frozen action chunk format, specify how action chunks tokenize into the VLM sequence alongside vision, language, and state tokens, and establish the tensor interface contracts between all Phase 3.2 components. This contract governs how Phase 2.3 action references are consumed at training time and how action predictions are produced at inference time.

### Why this matters
Every downstream component — the projectors, the flow matching module, the output head, the training dataloader (Phase 3.3) — depends on a shared understanding of tensor shapes, sequence layout, and the training-time vs inference-time data flow. Defining this contract upfront prevents cascading shape mismatches and rework. The chunk size choice (16) propagates into context budget calculations, memory estimates, and inference latency — all of which must be verified against measured Phase 3.1 numbers before committing.

### Design

**Action chunk definition:**

| Parameter | Value | Derivation |
|---|---|---|
| `chunk_size` | 16 | Target design (0.8 s at 20 Hz) |
| `action_dim` | 17 | Frozen action space (Phase 1.1.5) |
| `chunk_shape` | `(16, 17)` | Per-sample chunk tensor |
| `tokens_per_chunk` | 16 | 1 token per action step (per-step attention) |
| `state_dim` | 52 | Frozen robot state (Phase 1.1.6) |
| `state_tokens_per_segment` | 1 | Full 52-D state projected to 1 hidden-dim token |

**Chunking from Phase 2.3 action references:**

Phase 2.3 interleaved VLA sequences contain segments with `action_ref: {episode_seed, step_range: [start, end)}`. The dataloader (Phase 3.3) resolves these to continuous 17-D action arrays. Phase 3.2 defines how these arrays become chunks:

1. Load `action[start:end]` from HDF5 → shape `(n_steps, 17)` where `n_steps = end - start`
2. Split into non-overlapping chunks of `chunk_size`: `[(0:16), (16:32), ...]`
3. If the last chunk has fewer than `chunk_size` steps, zero-pad and create a binary `chunk_mask` indicating valid steps
4. Each chunk `(16, 17)` becomes 16 action tokens in the sequence

**Context budget verification (example for default single-placement sequence layout, subject to final Phase 2.3 image/segment density):**

The following is a reference estimate based on the Phase 2.3 default single-placement episode structure (8 segments, 2 images per segment). The actual token counts depend on Phase 2.3's finalized number of images per segment, whether multi-view images are included per keyframe, and whether some subsets use denser keyframe sampling. This budget must be re-verified once Phase 2.3 freezes the image reference density.

| Component | Tokens | Notes |
|---|---|---|
| Task description text | ~40 | From Phase 2.3 |
| 16 images × V vision tokens | ~16V | V measured in Phase 3.1.2 (expected 130–260); actual image count depends on Phase 2.3 |
| 8 narration texts × ~25 tokens | ~200 | From Phase 2.3 |
| 8 robot state tokens | 8 | 1 per segment |
| ~13 action chunks × 16 tokens | ~208 | ~200 steps ÷ 16 |
| Outcome text | ~5 | From Phase 2.3 |
| **Total** | **~461 + 16V** | At V=200: ~3661 tokens → fits in 8K with headroom |

**Risk note:** If Phase 2.3 increases the image count per segment (e.g., multi-view per keyframe, or denser keyframe sampling), the vision token total could grow significantly. At 32 images × V=200 ≈ 6861 tokens — still within 8K but with less margin. Multi-placement Level 3 episodes (longer sequences) could approach or exceed the 8K budget. The 3.2.0 verification step must confirm the budget against actual Phase 2.3 reference counts, not just this example.

**Sequence layout during training:**

The full interleaved VLA sequence follows this token order for each segment:
```
[observation_images] [narration_text] [state_token] [action_chunk_tokens]
```

Wrapped in an episode-level structure:
```
[task_desc_text] [seg₀_images] [narr₀_text] [state₀] [actions₀_chunk₀] [actions₀_chunk₁] [seg₁_images] [narr₁_text] [state₁] [actions₁_chunk₀] ... [outcome_text]
```

**Token type mask** — a per-position label used to route loss computation:
- `TEXT` positions → autoregressive cross-entropy next-token loss
- `IMAGE` positions → no loss (backbone-internal vision tokens)
- `STATE` positions → no loss (conditioning input only)
- `ACTION` positions → flow matching velocity MSE loss

**Training-time data flow (single sample):**

```
                    ┌─── text tokens ──→ embed_tokens() ──→ text_embeds (seq_t, H)
                    │
action_ref (HDF5) ──┼─── chunk + noise ──→ action_projector ──→ action_embeds (16*C, H)
                    │
robot_state (52-D) ─┼─── state_projector ──→ state_embeds (S, H)
                    │
pixel_values ───────┼─── vision_encoder ──→ vision_embeds (seq_v, H)
                    │
                    └──→ assemble sequence ──→ (B, seq_total, H) ──→ backbone transformer
                                                                         │
                                              ┌──────────────────────────┘
                                              │
                              text positions ──→ LM head ──→ AR cross-entropy loss
                              action positions ─→ action_head ──→ velocity ──→ FM MSE loss
```

Where `H` = `backbone.hidden_size`, `C` = number of chunks per episode, `S` = number of segments.

**Inference-time data flow:**

```
observation + state ──→ context tokens ──→ backbone
                                              │
                          action_positions ────┘
                               │
                    ┌──────────┘
                    │  K denoising steps:
                    │    a_t = noisy action ──→ action_projector ──→ action_embeds
                    │    backbone(context + action_embeds) ──→ hidden at action positions
                    │    action_head(hidden) ──→ velocity v
                    │    a_{t+dt} = a_t + dt * v (Euler step)
                    └──→ denoised action chunk (16, 17)
```

**Tensor interface contracts:**

| Component | Input shape | Output shape |
|---|---|---|
| `RobotStateProjector.forward(state)` | `(B, 52)` | `(B, 1, H)` |
| `NoisyActionProjector.forward(noisy_actions, t)` | `(B, 16, 17)`, `(B, 1)` | `(B, 16, H)` |
| `ActionOutputHead.forward(hidden)` | `(B, 16, H)` | `(B, 16, 17)` |
| `FlowMatchingModule.interpolate(x_data, noise, t)` | `(B, 16, 17)` × 2, `(B, 1)` | `(B, 16, 17)` |
| `FlowMatchingModule.target_velocity(x_data, noise)` | `(B, 16, 17)` × 2 | `(B, 16, 17)` |
| `FlowMatchingModule.loss(pred_v, target_v, mask)` | `(B, 16, 17)` × 2, `(B, 16)` | scalar |
| `VLAModel.forward(batch)` | training batch dict | `{text_loss, action_loss, total_loss}` |
| `VLAModel.predict_actions(obs, state, K)` | observation + `(B, 52)` | `(B, 16, 17)` |

**Config** — `configs/model/action_head.yaml`:

```yaml
action_head:
  chunk_size: 16
  action_dim: 17
  state_dim: 52
  tokens_per_action_step: 1

  projector:
    hidden_dim: null          # defaults to backbone.hidden_size
    n_layers: 2               # MLP depth for projectors
    activation: "gelu"
    dropout: 0.0

  timestep_embed_dim: 256     # sinusoidal timestep embedding dimension

  flow_matching:
    n_denoising_steps: 10     # K for inference ODE integration
    solver: "euler"           # euler | midpoint | rk4
    sigma_min: 0.001          # minimum noise scale

  loss:
    lambda_text: 1.0          # AR text loss weight
    lambda_action: 1.0        # FM action loss weight

  inference:
    execute_steps: 8          # execute first N of 16 predicted steps before re-predicting
```

### Execution checklist
- Inspect EO-1 codebase for action chunk definitions, sequence layout conventions, and token type routing
- Define frozen dataclasses: `ActionChunk`, `ChunkMask`, `TokenTypeMask`, `SequenceLayout`
- Write `configs/model/action_head.yaml` with all parameters
- Verify context budget against measured Phase 3.1 vision token count — confirm single-placement and Level 3 multi-placement episodes fit in 8K
- Document tensor interface contracts for all components (table above)
- Write 1 synthetic training batch by hand and verify all shapes propagate correctly (on paper or in a shape-checking unit test)

### Milestone (minimum success criteria)
- Action chunk contract is defined and frozen: `chunk_size=16`, `action_dim=17`, 1 token per step, 1 state token per segment. Context budget is verified against Phase 3.1 measurements. Config file exists. Tensor interfaces are documented. A synthetic shape-propagation test passes for all component signatures.

---

## 3.2.1) Flow Matching Action Head

### What we will do
Inspect the EO-1 codebase for its flow matching module. If EO-1 provides conditional flow matching (CFM) primitives — noise interpolation, velocity field computation, training loss, ODE-based inference — reuse and adapt them for our 17-D action space and chunk size 16. Only implement a new `FlowMatchingModule` where the inspected EO-1 code does not provide a realistic integration path.

### Why this matters
Flow matching is the mechanism that converts the backbone's contextual hidden states into continuous action trajectories. Unlike discretized action bins (which lose precision) or simple MSE regression (which suffers from mode averaging), flow matching generates high-quality continuous actions by learning a velocity field that transforms noise into the target action distribution. This is the core technical differentiator of the EO-1 architecture over prior VLA approaches.

### Design

**Implementation strategy — EO-1 reuse first:**

Before writing any new code, inspect the EO-1 codebase for:
- A flow matching or CFM module (noise schedule, interpolation, velocity parameterization)
- Training loss computation (MSE on predicted velocity vs target velocity)
- ODE solver for inference (Euler, midpoint, or adaptive)
- Sigma schedule or noise conditioning

If EO-1 provides these, import or copy into `models/` and adapt for our action dimensions. If EO-1's flow matching module has a different API than described below, prefer the EO-1 interface.

**If EO-1 does not provide a reusable flow matching module**, implement `FlowMatchingModule` with these components:

**Conditional Flow Matching (Lipman et al., 2023):**

The optimal transport conditional flow matching formulation:
- **Forward interpolation** (noise → data path): `x_t = (1 - t) · noise + t · x_data`, where `t ∈ [0, 1]`, `noise ~ N(0, I)`
  - At `t = 0`: pure noise
  - At `t = 1`: clean data (ground truth actions)
- **Target velocity field**: `u_t = x_data - noise` (constant along the straight-line path, independent of `t`)
- **Training objective**: `L_FM = ||v_θ(x_t, t) - u_t||²` (MSE between predicted and target velocity)
- **Inference**: Start at `x_0 ~ N(0, I)`, integrate `dx = v_θ(x_t, t) dt` from `t = 0` to `t = 1` using K discrete steps

The straight-line optimal transport path is simpler and more stable than variance-preserving diffusion schedules. The target velocity is constant (does not depend on `t`), which makes training straightforward.

**Training step:**
```
Input: x_data (B, 16, 17) — ground truth action chunk
1. Sample t ~ U[0, 1] → (B, 1, 1)  (one t per sample)
2. Sample noise ~ N(0, I) → (B, 16, 17)
3. Interpolate: x_t = (1 - t) * noise + t * x_data → (B, 16, 17)
4. Target velocity: u_t = x_data - noise → (B, 16, 17)
5. [Model predicts v_θ(x_t, t) via backbone + action_head — see 3.2.4]
6. Loss: MSE(v_θ, u_t) with optional chunk_mask for padded chunks
```

**Inference step (Euler solver):**
```
Input: context (observation, text, state)
1. x_0 = noise ~ N(0, I) → (16, 17)
2. dt = 1.0 / K
3. For k = 0, 1, ..., K-1:
     t_k = k * dt
     v = model.predict_velocity(context, x_{t_k}, t_k) → (16, 17)
     x_{t_{k+1}} = x_{t_k} + dt * v
4. Return x_1 ≈ x_data → (16, 17) denoised action chunk
```

**Higher-order solvers** (optional, configurable):
- **Midpoint**: `k_1 = v(x_t, t)`, `k_2 = v(x_t + dt/2 * k_1, t + dt/2)`, `x_{t+dt} = x_t + dt * k_2`
- **RK4**: Standard 4th-order Runge-Kutta (4 function evaluations per step)
- Default: Euler (simplest, good enough with K ≥ 10 for smooth velocity fields)

**Sigma min clamp:** Apply `sigma_min` (default 0.001) as a lower bound on noise magnitude to avoid numerical instability near `t = 1`:
```
noise_clamped = noise * max(1.0, sigma_min / ||noise||)
```

**`FlowMatchingModule` API (EO-1-derived or new):**

| Method | Signature | Purpose |
|---|---|---|
| `sample_timestep(batch_size)` | `→ (B, 1, 1)` | Sample `t ~ U[0, 1]` |
| `interpolate(x_data, noise, t)` | `→ (B, 16, 17)` | Compute `x_t` |
| `target_velocity(x_data, noise)` | `→ (B, 16, 17)` | Compute `u_t = x_data - noise` |
| `loss(pred_velocity, target_velocity, mask)` | `→ scalar` | Masked MSE loss |
| `denoise(predict_fn, shape, K, context)` | `→ (B, 16, 17)` | Full ODE integration from noise to data |

`predict_fn` is a callable that takes `(x_t, t)` and returns predicted velocity. This decouples the flow matching math from the neural network architecture — the backbone + action head provide `predict_fn`, while `FlowMatchingModule` handles the ODE integration.

### Execution checklist
- **Inspect EO-1 codebase** for flow matching / CFM module, noise schedule, velocity parameterization, loss computation, and ODE solver. Document what exists and what can be reused
- Reuse/adapt EO-1's flow matching code if available. Only implement `FlowMatchingModule` from scratch if EO-1 does not provide equivalent functionality
- Implement or verify: `sample_timestep()`, `interpolate()`, `target_velocity()`, `loss()`, `denoise()`
- Unit test: `interpolate(x_data, noise, 0.0)` == `noise`, `interpolate(x_data, noise, 1.0)` == `x_data`
- Unit test: `target_velocity()` == `x_data - noise` for all inputs
- Unit test: `loss()` returns zero when predicted velocity equals target velocity
- Unit test: `denoise()` with identity `predict_fn` (returns true velocity) recovers `x_data` exactly
- Unit test: `denoise()` with Euler, midpoint, and RK4 solvers all converge
- Unit test: `loss()` correctly masks padded chunk positions
- Verify numerical stability: no NaN/Inf for edge cases (`t=0`, `t=1`, zero noise, large noise)

### Milestone (minimum success criteria)
- Flow matching module (EO-1-derived or new) passes all unit tests. `interpolate()` produces correct noisy actions for boundary values of `t`. `loss()` computes masked MSE correctly. `denoise()` with K=10 Euler steps recovers ground truth actions from noise when given a perfect velocity predictor (identity test). No NaN/Inf in any test case.

---

## 3.2.2) Robot State Projector

### What we will do
Inspect the EO-1 codebase for a robot state projector or proprioceptive conditioning module. If EO-1 provides one, reuse and adapt it for our 52-D state vector. Otherwise, implement `RobotStateProjector` — an MLP that projects the normalized 52-D robot state into a single hidden-dimension token for injection into the VLM sequence as proprioceptive conditioning.

### Why this matters
The VLM backbone sees images and text but has no direct access to joint positions, velocities, gripper states, or end-effector poses. The robot state projector bridges this gap: it injects proprioceptive information into the same representation space as vision and language tokens, allowing the backbone's attention mechanism to condition action predictions on the robot's current physical configuration. Without state conditioning, the model must infer joint states from images alone — feasible for coarse motion but insufficient for the millimeter-level precision required by LEGO press-fit assembly.

### Design

**Implementation strategy — EO-1 reuse first:**

Inspect the EO-1 codebase for:
- A state projector, proprioceptive encoder, or state embedding module
- How EO-1 injects robot state into the transformer sequence (single token, multiple tokens, or added to existing tokens)
- What state dimensions EO-1 uses and whether the projection architecture is dimension-agnostic

If EO-1 provides a state projector, adapt it for 52-D input and `backbone.hidden_size` output.

**If EO-1 does not provide a reusable state projector**, implement `RobotStateProjector(nn.Module)`:

```
Input:  robot_state (B, 52) — normalized, all components in ~[-1, 1]
   → LayerNorm(52)
   → Linear(52, H) + GELU + Linear(H, H)
Output: state_token (B, 1, H)
```

Where `H = backbone.hidden_size` (from Phase 3.1).

**Design choices:**
- **1 token for full 52-D state**: The proprioceptive state is low-dimensional compared to the backbone's hidden size. A 2-layer MLP with `H` hidden units has capacity to encode all 52 dimensions into a single `H`-dimensional token. Splitting into multiple tokens (e.g., separate for joints vs EE) adds complexity without clear benefit at this scale.
- **LayerNorm on input**: The 52-D state vector components have different scales after normalization (joint positions ≈ [-1, 1], quaternions ≈ [-1, 1], velocities ≈ [-1, 1]). LayerNorm harmonizes these before projection.
- **No dropout**: The state projector is small (~`52*H + H*H` ≈ 0.3M parameters for H=2560). Regularization through the overall training setup (Phase 3.3) is sufficient.

**Sequence position:** The state token is inserted immediately before the action tokens for each segment:
```
... [narration_text] [STATE_TOKEN] [action_tokens_chunk₀] [action_tokens_chunk₁] ...
```

This places the proprioceptive state in the causal context of subsequent action tokens — the backbone attends to the state token when processing action positions.

**Parameter count estimate:** For H = 2560 (Qwen3.5-4B hidden size):
- Layer 1: 52 × 2560 + 2560 = 135,680
- Layer 2: 2560 × 2560 + 2560 = 6,556,160
- LayerNorm: 52 × 2 = 104
- **Total: ~6.7M parameters** (negligible compared to 4B backbone)

### Execution checklist
- **Inspect EO-1 codebase** for state projector or proprioceptive conditioning module. Document what exists
- Reuse/adapt EO-1's state projector if available. Only implement `RobotStateProjector` from scratch if EO-1 has no equivalent
- Verify output shape: `(B, 1, H)` where `H = backbone.hidden_size`
- Unit test: forward pass with random 52-D input produces correct output shape, no NaN
- Unit test: gradient flows through the projector (non-zero gradients on all parameters after backward pass)
- Unit test: output magnitude is in a reasonable range (not exploding or vanishing) — check `||output|| / sqrt(H)` is O(1)
- Unit test: projector correctly handles batched input
- Verify parameter count matches estimate

### Milestone (minimum success criteria)
- Robot state projector (EO-1-derived or new) produces `(B, 1, H)` output from `(B, 52)` input. All unit tests pass. Gradients flow correctly. Parameter count is ~6.7M (or matches EO-1's projector size). Output magnitudes are numerically stable.

---

## 3.2.3) Noisy Action Projector & Action Output Head

### What we will do
Inspect the EO-1 codebase for its action embedding and de-embedding modules. If EO-1 provides a noisy action projector (input side: noisy actions + timestep → hidden tokens) and an action output head (output side: hidden states → velocity predictions), reuse and adapt them for our 17-D action space and chunk size 16. Otherwise, implement `NoisyActionProjector` and `ActionOutputHead` as complementary MLP modules.

### Why this matters
These two modules are the entry and exit points for continuous actions in the backbone's hidden space. The noisy action projector encodes the current denoising state (noisy action + flow matching timestep) into the same representation space as vision and language tokens, so the backbone can reason about actions in context. The action output head decodes the backbone's hidden representations back into the 17-D action space as velocity predictions. Together, they form the action-space interface of the Transfusion architecture.

### Design

**Implementation strategy — EO-1 reuse first:**

Inspect the EO-1 codebase for:
- A noisy action projector or action embedding module that takes (noisy actions + timestep) and produces hidden-dim tokens
- An action output head or action de-embedding that maps hidden states to action-space predictions
- Timestep embedding implementation (sinusoidal, learned, or FiLM-style)
- Whether EO-1 uses per-step tokens (one token per action timestep) or per-chunk tokens (one token for the whole chunk)

If EO-1 provides these modules, adapt them for `action_dim=17` and `chunk_size=16`.

**If EO-1 does not provide reusable modules:**

**Noisy Action Projector — `NoisyActionProjector(nn.Module)`:**

```
Input:  noisy_actions (B, 16, 17), t (B, 1)
   → timestep_embed = sinusoidal_embedding(t) → (B, 1, d_t)     [d_t = 256]
   → timestep_embed_expanded = expand to (B, 16, d_t)            [broadcast across chunk steps]
   → concat = [noisy_actions, timestep_embed_expanded] → (B, 16, 17 + d_t)
   → Linear(17 + d_t, H) + GELU + Linear(H, H)
Output: action_tokens (B, 16, H)
```

Each of the 16 action steps becomes one hidden-dimension token. The timestep `t` is broadcast to all steps within the same chunk — the entire chunk shares the same denoising timestep.

**Sinusoidal timestep embedding:**
```
t_embed[2i]   = sin(t / 10000^(2i/d_t))
t_embed[2i+1] = cos(t / 10000^(2i/d_t))
```
Standard sinusoidal positional encoding applied to the scalar flow matching timestep. Produces a `d_t`-dimensional vector (default 256) that encodes the denoising progress.

**Action Output Head — `ActionOutputHead(nn.Module)`:**

```
Input:  hidden_states (B, 16, H) — backbone output at action token positions
   → Linear(H, H) + GELU + Linear(H, 17)
Output: velocity_pred (B, 16, 17)
```

The output head is intentionally lightweight — the backbone transformer has already performed the heavy reasoning. The head simply projects from hidden space to action space.

**Parameter count estimates** (for H = 2560, d_t = 256):
- Noisy action projector: (17 + 256) × 2560 + 2560 × 2560 ≈ 7.2M
- Action output head: 2560 × 2560 + 2560 × 17 ≈ 6.6M
- Timestep embedding: 0 (parameter-free sinusoidal)
- **Total: ~13.8M parameters**

**Combined with state projector (3.2.2): ~20.5M total** — approximately 0.5% of the 4B backbone. VRAM overhead is negligible.

### Execution checklist
- **Inspect EO-1 codebase** for action embedding/de-embedding modules and timestep embedding. Document what exists
- Reuse/adapt EO-1's modules if available. Only implement from scratch where necessary
- Implement or verify `NoisyActionProjector`: input `(B, 16, 17)` + `(B, 1)` → output `(B, 16, H)`
- Implement or verify `ActionOutputHead`: input `(B, 16, H)` → output `(B, 16, 17)`
- Implement or verify sinusoidal timestep embedding: input scalar `t ∈ [0, 1]` → output `(d_t,)` vector
- Unit test: `NoisyActionProjector` output shape is `(B, 16, H)` for all valid inputs
- Unit test: `ActionOutputHead` output shape is `(B, 16, 17)` for all valid inputs
- Unit test: timestep embedding produces distinct vectors for `t=0.0`, `t=0.5`, `t=1.0`
- Unit test: round-trip shape check — projector output is compatible with output head input dimension
- Unit test: gradients flow through both modules (non-zero after backward)
- Unit test: output head with random hidden states produces values in reasonable range (not exploding)
- Verify total parameter count for projector + head matches estimates

### Milestone (minimum success criteria)
- Noisy action projector (EO-1-derived or new) correctly encodes `(B, 16, 17)` noisy actions with timestep `t` into `(B, 16, H)` tokens. Action output head correctly decodes `(B, 16, H)` hidden states into `(B, 16, 17)` velocity predictions. Timestep embedding produces distinct, non-degenerate vectors across `[0, 1]`. All unit tests pass. Combined parameter count is ~20.5M (or matches EO-1's action head parameter count).

---

## 3.2.4) VLA Model Assembly

### What we will do
Inspect the EO-1 codebase for its top-level VLA model class — the module that wires the backbone, state projector, action projector, output head, and flow matching module into a single trainable model with Transfusion-style dual loss. If EO-1 provides this assembly, reuse and adapt it. Otherwise, implement `VLAModel(nn.Module)` that composes the Phase 3.1 backbone with all Phase 3.2 components, supporting both training (dual loss forward pass) and inference (multi-step denoising action prediction).

### Why this matters
This is the integration subphase — where individual components become a working system. The VLA model must correctly assemble heterogeneous token types (vision, text, state, action) into a single sequence, route them through the backbone, and compute the appropriate loss at each position type. An error in sequence assembly, attention masking, or loss routing silently degrades training rather than producing an obvious crash. Getting this right is prerequisite for Phase 3.3 (training loop) and determines whether the model can learn to produce useful actions.

### Design

**Implementation strategy — EO-1 reuse first:**

Inspect the EO-1 codebase for:
- A top-level model class that composes backbone + action head + projectors
- How EO-1 assembles the interleaved sequence (token ordering, embedding concatenation)
- How EO-1 implements the Transfusion dual loss (AR on text, FM on actions)
- How EO-1 handles the `inputs_embeds` injection (bypassing the backbone's token embedding layer to insert projected state/action tokens)
- Inference implementation (multi-step denoising with cached context)

If EO-1 provides a model assembly class, adapt it for our component interfaces.

**If EO-1 does not provide a reusable model class**, implement `VLAModel(nn.Module)`:

**Core components (composed, not re-implemented):**

| Component | Source | Purpose |
|---|---|---|
| `self.backbone` | Phase 3.1 `VLMBackbone` | Frozen (or LoRA) VLM transformer |
| `self.state_projector` | Phase 3.2.2 `RobotStateProjector` | 52-D → H state token |
| `self.action_projector` | Phase 3.2.3 `NoisyActionProjector` | (17-D + t) → H action tokens |
| `self.action_head` | Phase 3.2.3 `ActionOutputHead` | H → 17-D velocity predictions |
| `self.flow_matching` | Phase 3.2.1 `FlowMatchingModule` | CFM training loss + ODE inference |

**Embedding-level injection:**

> **CRITICAL INTEGRATION POINT.** The interaction between custom `inputs_embeds` and the backbone's native `pixel_values` processing is the riskiest technical step in Phase 3.2. Many HuggingFace VLMs (including Qwen-VL variants) have internal vision-language fusion pipelines that may not cleanly accept both simultaneously. The steps below describe the *intended* approach, but the actual injection strategy must be validated against the real backbone behavior before committing. See the decision framework below.

The backbone (Qwen3.5-4B via HuggingFace) natively processes `input_ids` + `pixel_values` through its own embedding layer and vision encoder. To inject state and action tokens, `VLAModel` operates at the embedding level:

1. **Text embeddings**: `backbone.model.embed_tokens(input_ids)` → `(B, seq_text, H)`
2. **Vision embeddings**: produced internally by backbone's vision encoder from `pixel_values`
3. **State embeddings**: `self.state_projector(robot_state)` → `(B, n_segments, H)`
4. **Action embeddings**: `self.action_projector(noisy_actions, t)` → `(B, n_action_tokens, H)`
5. **Assembly**: concatenate all embeddings in the correct sequence order → `(B, seq_total, H)`
6. **Forward**: pass assembled embeddings via `inputs_embeds` parameter through backbone transformer layers

**Multimodal injection decision — must resolve before implementing forward pass:**

The core question is: can the backbone accept `inputs_embeds` (pre-assembled, including custom state/action tokens) **and** `pixel_values` (for vision encoder) in the same forward call? Three possible outcomes:

| Scenario | Evidence | Action |
|---|---|---|
| **A. Compatible passthrough** | Backbone processes `pixel_values` through vision encoder, produces vision embeddings, and allows `inputs_embeds` to coexist (vision embeddings inserted at image token positions within `inputs_embeds`) | Use `inputs_embeds` directly — assemble text + state + action embeddings, let backbone handle vision internally |
| **B. Mutually exclusive** | Backbone ignores `pixel_values` when `inputs_embeds` is provided (common HuggingFace pattern: `inputs_embeds` bypasses the entire embedding stage including vision) | Must run vision encoder separately: extract vision embeddings via `backbone.vision_model(pixel_values)`, then assemble all four modalities (text + vision + state + action) into `inputs_embeds` manually |
| **C. EO-1 custom path** | EO-1 has its own token fusion mechanism that does not use the standard HuggingFace `inputs_embeds` interface | Follow EO-1's approach — adapt its fusion code rather than fighting the HuggingFace API |

**How to resolve:** In the first execution step of 3.2.4, before writing any model code:
1. Inspect EO-1's model class for how it handles multimodal + action embedding injection
2. If no EO-1 guidance: call `backbone.forward(inputs_embeds=dummy, pixel_values=dummy)` and verify (a) no error, (b) vision embeddings appear at expected positions in the output hidden states, (c) custom embeddings at state/action positions are preserved
3. Document which scenario (A/B/C) applies and choose the injection strategy accordingly

This is not a blocker for the Phase 3.2 document — the component designs (projectors, flow matching, output head) are injection-strategy-agnostic. But it **is** the first thing to resolve during 3.2.4 implementation, before writing `assemble_sequence()` or `forward()`.

This requires the Phase 3.1 backbone wrapper to support `inputs_embeds` passthrough (and possibly separate vision encoding, if Scenario B). If Phase 3.1's `get_hidden_states()` does not already support this parameter, add a minimal extension: `get_hidden_states(..., inputs_embeds=None)` that passes `inputs_embeds` to the underlying HuggingFace model.

**Training forward pass — `VLAModel.forward(batch) → dict`:**

*Note: The pseudocode below illustrates the intended forward structure assuming Scenario A (compatible passthrough). The exact multimodal fusion call — particularly step 6 (`get_hidden_states`) — must follow the resolved Scenario A/B/C integration path. Under Scenario B, vision embeddings would be extracted separately and assembled into `full_embeds` before the backbone call, and `pixel_values` would not be passed.*

```python
def forward(self, batch: dict) -> dict:
    """
    batch keys:
      - input_ids: (B, seq_text) — text token IDs
      - pixel_values: images for backbone vision encoder
      - attention_mask: (B, seq_total) — causal attention mask
      - robot_states: (B, n_segments, 52) — per-segment normalized states
      - action_chunks: (B, n_chunks, 16, 17) — ground truth action chunks
      - chunk_masks: (B, n_chunks, 16) — valid step mask (1=real, 0=padding)
      - token_type_ids: (B, seq_total) — per-position type: TEXT/IMAGE/STATE/ACTION
      - text_labels: (B, seq_text) — next-token targets for AR loss (-100 = ignore)
    """
    # 1. Sample flow matching timestep
    t = self.flow_matching.sample_timestep(B)             # (B, 1, 1)

    # 2. Sample noise and compute noisy actions
    noise = torch.randn_like(action_chunks_flat)           # (B, n_total_action_tokens, 17)
    noisy_actions = self.flow_matching.interpolate(
        action_chunks_flat, noise, t)                      # (B, n_total_action_tokens, 17)
    target_velocity = self.flow_matching.target_velocity(
        action_chunks_flat, noise)                         # (B, n_total_action_tokens, 17)

    # 3. Project state and action tokens
    state_embeds = self.state_projector(robot_states)      # (B, n_segments, H)
    action_embeds = self.action_projector(
        noisy_actions.view(B, -1, 17), t.squeeze())        # (B, n_total_action_tokens, H)

    # 4. Get text embeddings from backbone
    text_embeds = self.backbone.get_text_embeddings(input_ids)  # (B, seq_text, H)

    # 5. Assemble full sequence in correct order
    full_embeds = self.assemble_sequence(
        text_embeds, state_embeds, action_embeds,
        batch["sequence_layout"])                          # (B, seq_total, H)

    # 6. Forward through backbone transformer
    hidden_states = self.backbone.get_hidden_states(
        inputs_embeds=full_embeds,
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"])                # (B, seq_total, H)

    # 7. Compute AR text loss at text positions
    text_logits = self.backbone.lm_head(
        hidden_states[:, text_positions, :])               # (B, seq_text, vocab_size)
    text_loss = F.cross_entropy(
        text_logits.view(-1, vocab_size),
        text_labels.view(-1), ignore_index=-100)

    # 8. Compute FM action loss at action positions
    velocity_pred = self.action_head(
        hidden_states[:, action_positions, :])             # (B, n_total_action_tokens, 17)
    action_loss = self.flow_matching.loss(
        velocity_pred, target_velocity, chunk_masks_flat)

    # 9. Combine losses
    total_loss = (self.cfg.loss.lambda_text * text_loss
                + self.cfg.loss.lambda_action * action_loss)

    return {
        "total_loss": total_loss,
        "text_loss": text_loss,
        "action_loss": action_loss,
    }
```

**Inference — `VLAModel.predict_actions(observation, robot_state, K=10) → Tensor`:**

```python
@torch.no_grad()
def predict_actions(self, observation: dict, robot_state: torch.Tensor,
                    K: int = 10) -> torch.Tensor:
    """
    observation: dict with input_ids, pixel_values, attention_mask (text + images)
    robot_state: (1, 52) normalized
    Returns: (1, 16, 17) denoised action chunk
    """
    # 1. Prepare context embeddings (text + vision + state) — computed once
    state_embed = self.state_projector(robot_state)        # (1, 1, H)

    # 2. Define velocity prediction function for ODE solver
    def predict_velocity(noisy_actions, t):
        action_embeds = self.action_projector(noisy_actions, t)
        full_embeds = self.assemble_sequence(
            context_embeds, state_embed, action_embeds, layout)
        hidden = self.backbone.get_hidden_states(
            inputs_embeds=full_embeds,
            attention_mask=mask,
            pixel_values=observation["pixel_values"])
        return self.action_head(hidden[:, action_positions, :])

    # 3. Run ODE integration from noise to data
    action_chunk = self.flow_matching.denoise(
        predict_fn=predict_velocity,
        shape=(1, 16, 17),
        K=K)                                               # (1, 16, 17)

    return action_chunk
```

**Note on inference cost:** Each of the K denoising steps requires a full backbone forward pass. For K=10 at inference, this is 10× the cost of a single forward pass. At batch_size=1 with seq_length ~1000 tokens (single observation + state + 16 action tokens), a single Qwen3.5-4B forward pass on A100 is ~10–20 ms. K=10 denoising steps = ~100–200 ms per action chunk — comfortably within the 50 ms × 8 = 400 ms window if executing 8 of 16 predicted steps before re-predicting.

**KV cache optimization (optional, deferred):** During K denoising iterations, the context tokens (text + vision + state) are identical. Their KV cache entries could be computed once and reused, reducing each subsequent step to only the 16 action token positions. This optimization is valuable but not required for correctness — defer to Phase 3.3 if inference latency is acceptable without it.

### Execution checklist
- **Inspect EO-1 codebase** for the top-level VLA model class, sequence assembly logic, dual loss routing, and inference loop. Document what exists
- Reuse/adapt EO-1's model class if available. Only implement from scratch where necessary
- **CRITICAL — Resolve multimodal injection strategy first:** Test backbone with `inputs_embeds` + `pixel_values` to determine Scenario A/B/C (see decision table above). Document which scenario applies. This determines the `assemble_sequence()` and `forward()` implementation
- Verify Phase 3.1 backbone supports `inputs_embeds` parameter (and separate vision encoding if Scenario B). If not, add minimal extension
- Implement or adapt the VLA model `__init__()` (EO-1-derived or new): compose backbone + all action head components
- Implement or adapt the VLA model `forward()` (EO-1-derived or new): training forward pass with dual loss
- Implement or adapt the VLA model `predict_actions()` (EO-1-derived or new): inference with ODE denoising
- Implement `assemble_sequence()`: correct token ordering per the 3.2.0 layout contract
- Extend `get_model()` in `models/utils.py` with `"vla"` routing (lazy import, alongside existing `"transformer"` and `"vlm"` branches)
- Add `configs/model/vla.yaml` that composes `vlm.yaml` defaults + `action_head.yaml`
- Integration test: full forward pass with synthetic batch (random tensors, correct shapes) → returns `{total_loss, text_loss, action_loss}` with finite values and no NaN
- Integration test: `predict_actions()` with synthetic observation → returns `(1, 16, 17)` tensor with finite values
- Integration test: backward pass on `total_loss` produces non-zero gradients on action head parameters
- Integration test: backward pass does NOT produce gradients on frozen backbone parameters (verify freeze)
- Integration test: sequence assembly produces correct token ordering (verify positions match token_type_ids)
- Measure action head VRAM overhead: compare `torch.cuda.max_memory_allocated()` before/after forward pass, verify within Phase 3.1 budget

### Milestone (minimum success criteria)
- `VLAModel` (EO-1-derived or new) composes the Phase 3.1 backbone with all Phase 3.2 components. Training forward pass with synthetic data returns finite losses (both `text_loss` and `action_loss`). Inference produces `(1, 16, 17)` denoised action chunks with finite values. Gradients flow to action head parameters but not to frozen backbone weights. Token ordering in the assembled sequence matches the 3.2.0 contract. VRAM overhead is within Phase 3.1's measured budget.

---

## 3.2.5) End-to-End Validation & Profiling

### What we will do
Create a comprehensive test suite and standalone validation script that verify all Phase 3.2 components individually and integrated, profile the action head's VRAM and compute overhead on target hardware, and confirm that the complete VLA model is ready for Phase 3.3 training integration.

### Why this matters
The action head contains multiple interacting components (flow matching, projectors, output head, sequence assembly) that are individually correct but may fail at integration boundaries. The validation script provides a single command that confirms end-to-end functionality. Memory profiling ensures that the combined backbone + action head fits the A100 training budget before committing to expensive training runs in Phase 3.3.

### Design

**Test file** — `tests/test_action_head.py`:

| Test class | Scope | Markers | Count |
|---|---|---|---|
| `TestActionChunkContract` | Chunk shapes, masking, context budget | `action_head` | ~5 |
| `TestFlowMatching` | CFM math: interpolation, velocity, loss, ODE solver | `action_head` | ~8 |
| `TestRobotStateProjector` | Shape, gradients, numerical stability | `action_head` | ~4 |
| `TestNoisyActionProjector` | Shape, timestep embedding, gradients | `action_head` | ~5 |
| `TestActionOutputHead` | Shape, gradients, round-trip compatibility | `action_head` | ~3 |
| `TestVLAModelTraining` | Forward pass, dual loss, gradient routing | `action_head`, `vlm`, `gpu`, `slow` | ~6 |
| `TestVLAModelInference` | Denoising, output shape, finite values | `action_head`, `vlm`, `gpu`, `slow` | ~4 |
| `TestBackwardCompatibility` | Existing models unaffected | (none) | ~2 |

New `action_head` pytest marker in `tests/conftest.py` — auto-skipped when `transformers` is not installed, consistent with the `vlm` marker from Phase 3.1. Tests marked only `action_head` (not `vlm`/`gpu`) run on CPU with mock backbone for CI compatibility.

**Validation script** — `scripts/validate_action_head.py`:

Following the established `scripts/validate_*.py` pattern, runs 10 sequential checks and produces artifacts to `logs/action_head/`.

| Check | Input | Verifies |
|---|---|---|
| 1. Config parsing | `action_head.yaml` + `vla.yaml` | Configs parse without error |
| 2. Component instantiation | Config | All modules instantiate with correct param counts |
| 3. Flow matching math | Synthetic data | Interpolation boundary values, velocity correctness |
| 4. Projector shapes | Random tensors | State projector: (B, 52) → (B, 1, H); action projector: (B, 16, 17) + t → (B, 16, H) |
| 5. Output head shape | Random tensors | (B, 16, H) → (B, 16, 17) |
| 6. VLA training forward | Synthetic batch | Returns finite losses, correct shapes |
| 7. VLA inference | Synthetic observation | Returns (1, 16, 17) denoised chunk, finite values |
| 8. Gradient routing | Synthetic batch | Action head gradients non-zero; frozen backbone gradients zero |
| 9. Numerical stability | Edge cases (t=0, t=1, zero state, large actions) | No NaN/Inf in any output |
| 10. Memory overhead | Forward + backward on GPU | Peak VRAM within Phase 3.1 budget |

**Artifacts:** `logs/action_head/validation_report.json` (all check results), `logs/action_head/memory_overhead.json` (VRAM delta from backbone-only), `logs/action_head/param_counts.json` (per-component parameter counts).

**Memory profiling** (within validation script):

Measure VRAM delta by comparing backbone-only forward pass vs full VLA model forward pass:
```
VRAM_backbone_only = measure(backbone.forward(synthetic_input))
VRAM_vla_model     = measure(vla_model.forward(synthetic_batch))
VRAM_action_head_overhead = VRAM_vla_model - VRAM_backbone_only
```

Expected overhead: ~20.5M parameters × 2 bytes (bf16) ≈ 41 MB for weights + activations for 16 extra action tokens and 1 state token per segment. The overhead should be well under 1 GB even with gradients.

**SLURM job template** — `infra/gilbreth/job_templates/09_validate_action_head.sh`:

Single A100, 30 minutes, runs the validation script with full VLM backbone loaded. Requires weights pre-cached (job 07).

### Execution checklist
- Add `action_head` marker to `tests/conftest.py` with auto-skip logic
- Write tests in `tests/test_action_head.py` covering all component classes + integration
- Create mock backbone for CPU-only tests (returns random hidden states with correct shapes)
- Create `scripts/validate_action_head.py` with 10 checks and artifact output
- Create `infra/gilbreth/job_templates/09_validate_action_head.sh`
- Run validation script on lab PC with `model=vla_dev` (mock or small backbone)
- Run validation script on A100 with `model=vla` (full Qwen3.5-4B)
- Verify all 10 checks pass on both environments
- Verify VRAM overhead is within Phase 3.1 budget (expected < 1 GB)
- Verify all existing tests (`test_models.py`, `test_vlm_backbone.py`) pass unchanged
- Save artifacts to `logs/action_head/`

### Milestone (minimum success criteria)
- Test suite passes: ~37 tests across 8 test classes. Validation script exits with 10/10 checks passed. Action head VRAM overhead measured and documented (expected < 1 GB). All existing tests pass unchanged. Artifacts saved to `logs/action_head/`.

---

# Downstream Contract with Phase 3.3

Phase 3.3 (End-to-End Training Integration) will consume Phase 3.2's VLA model as follows:

| Phase 3.3 Need | Phase 3.2 Provider | Interface |
|---|---|---|
| Complete VLA model for training | VLA model class (EO-1-derived or new) | `model.forward(batch) → {total_loss, text_loss, action_loss}` |
| Action prediction for evaluation | VLA model `predict_actions()` | `(observation, state, K) → (B, 16, 17)` action chunk |
| Training batch format spec | 3.2.0 contract + `VLAModel.forward()` signature | Batch dict with all required keys and shapes |
| Loss components for logging | `forward()` return dict | Separate `text_loss` and `action_loss` for W&B tracking |
| Action head config | `configs/model/action_head.yaml` | Hydra-composable config for hyperparameter sweeps |
| Model construction | `get_model(cfg)` with `architecture.type: "vla"` | Returns fully assembled `VLAModel` |
| Freeze/unfreeze control | `VLAModel` exposes backbone freeze methods | Phase 3.3 controls LoRA and fine-tuning schedule |
| Memory profile | `logs/action_head/memory_overhead.json` | Measured VRAM for batch size planning |

**Key design constraint:** The VLA model class owns the model architecture (embedding assembly, loss computation, inference loop). The Phase 3.3 training loop owns the optimizer, scheduler, distributed strategy, data loading, and checkpoint management. The model's `forward()` returns loss tensors — it does not call `loss.backward()`.

**Dataloader contract:** Phase 3.3's dataloader must produce batches matching the VLA model's `forward()` signature defined in 3.2.4. This includes resolving Phase 2.3 JSONL references to actual HDF5 data, tokenizing text, chunking actions, sampling flow matching noise, and computing token type masks. Phase 3.2 defines the batch format; Phase 3.3 implements the dataloader that produces it.

---

# Startup-Grade Outputs (deliverables by end of 3.2)
- **Flow matching module** — EO-1-derived or new `FlowMatchingModule` with CFM training loss, velocity parameterization, and configurable ODE solver (Euler/midpoint/RK4). Provenance documented
- **Robot state projector** — EO-1-derived or new `RobotStateProjector` (52-D → hidden_dim, single token). Provenance documented
- **Noisy action projector + output head** — EO-1-derived or new `NoisyActionProjector` (17-D + timestep → hidden_dim) and `ActionOutputHead` (hidden_dim → 17-D velocity). Provenance documented
- **VLA model class** — EO-1-derived or newly implemented, composing backbone + all action head components, with Transfusion-style dual loss (AR + FM) training forward pass and multi-step denoising inference. Multimodal injection strategy (Scenario A/B/C) resolved and documented
- **Hydra configs** — `configs/model/action_head.yaml` (action head parameters) and `configs/model/vla.yaml` (full VLA model composition). Adapted from EO-1 config structure where applicable
- **Test suite** — ~37 tests across 8 test classes covering components, integration, and backward compatibility
- **Validation script** — 10-check `scripts/validate_action_head.py` confirming end-to-end functionality and memory profile
- **Memory profile** — Measured VRAM overhead of action head on A100, verified within Phase 3.1 budget

---

# Files Inventory

### EO-1 reuse expectations

The file list below is **provisional** — it assumes EO-1 does not already provide a directly usable module for each item. During implementation, the first step of each subphase is to inspect the EO-1 codebase. The actual outcome for each file will be one of:

| Category | Meaning | Example |
|---|---|---|
| **Reused directly** | EO-1 module imported or copied as-is (action dim updated in config, not in code) | Flow matching math utilities |
| **Adapted from EO-1** | EO-1 module copied and modified (action dim, chunk size, projector sizes, loss routing) | VLA model class, action projectors |
| **Newly implemented** | No EO-1 equivalent found; written from scratch following project conventions | Validation scripts, SLURM job templates |

### New or adapted files (up to 9)

| File | Purpose | Expected EO-1 reuse |
|---|---|---|
| `models/flow_matching.py` | CFM module: interpolation, velocity, loss, ODE solver | **Adapt** from EO-1 FM module if available |
| `models/action_head.py` | State projector, noisy action projector, output head | **Adapt** from EO-1 action head modules if available |
| `models/vla_model.py` | VLA model assembly: composes backbone + action head, dual loss, inference | **Adapt** from EO-1 model class if available |
| `configs/model/action_head.yaml` | Action head configuration | **Adapt** from EO-1 config if available |
| `configs/model/vla.yaml` | Full VLA model config (composes vlm.yaml + action_head.yaml) | **Adapt** from EO-1 config structure |
| `configs/model/vla_dev.yaml` | Lightweight VLA config for local dev/CI (mock or small backbone, CPU-compatible) | **New** (follows `vlm.yaml`/`vlm_dev.yaml` pattern from Phase 3.1) |
| `tests/test_action_head.py` | Unit + integration tests (~37 tests) | **New** (project-specific test suite) |
| `scripts/validate_action_head.py` | Standalone 10-check validation | **New** (project-specific, uses our sim images) |
| `infra/gilbreth/job_templates/09_validate_action_head.sh` | SLURM validation job | **New** (cluster-specific) |

### Modified files (4)

| File | Change | Expected EO-1 reuse |
|---|---|---|
| `models/utils.py` | Add `"vla"` branch to `get_model()` (lazy import) | **Adapt** from EO-1 model factory if it has one |
| `models/vlm_backbone.py` | Ensure `get_hidden_states()` supports `inputs_embeds` param | **Minor extension** if not already present |
| `tests/conftest.py` | Add `action_head` pytest marker + auto-skip | **New** (project-specific CI gating) |
| `CLAUDE.md` | Add action head module docs, commands, test markers | **New** (project-specific) |

---

# Phase 3.2 Definition of Done

Phase 3.2 is complete when:
- All action head components (flow matching module, state projector, action projector, output head) are implemented and individually tested.
- The VLA model class (EO-1-derived or new) composes the Phase 3.1 backbone with all action head components.
- Training forward pass with synthetic data returns finite `text_loss` and `action_loss`.
- Inference produces `(B, 16, 17)` denoised action chunks via K-step ODE integration.
- Gradients flow to action head parameters (state projector, action projector, output head) but not to frozen backbone weights.
- Flow matching module (EO-1-derived or new) correctly implements CFM: interpolation, velocity computation, masked loss, and ODE solver.
- The multimodal injection strategy (Scenario A/B/C from 3.2.4) is resolved and documented.
- Action head VRAM overhead is measured and within Phase 3.1's A100 budget (expected < 1 GB).
- All existing tests (`test_models.py`, `test_vlm_backbone.py`, etc.) continue to pass — zero regressions.
- `get_model(cfg)` correctly routes to the VLA model for `architecture.type: "vla"`, to `VLMBackbone` for `"vlm"`, and to `TransformerModel` for `"transformer"`.
- Validation script (`scripts/validate_action_head.py`) exits with 10/10 checks passed on both lab PC and A100.
- Hydra configs (`action_head.yaml`, `vla.yaml`) parse correctly and compose with existing configs.
- `CLAUDE.md` is updated with action head module docs, commands, and test markers.

---

# Verification Plan

### Local (lab PC, RTX 4090)
```bash
# CPU-only tests (mock backbone, fast)
pytest tests/test_action_head.py -v -m "action_head and not vlm and not gpu"

# Full suite (requires GPU + transformers)
pytest tests/test_action_head.py -v

# Backward compat
pytest tests/test_models.py tests/test_vlm_backbone.py -v

# Validation script (dev config)
python scripts/validate_action_head.py --model-config vla_dev
```

### Cluster (Gilbreth A100)
```bash
# Full validation with Qwen3.5-4B backbone
sbatch infra/gilbreth/job_templates/09_validate_action_head.sh

# Or interactively
python scripts/validate_action_head.py
```

### CI (no changes needed)
Existing CI installs `.[ci]` which excludes `transformers`. Tests marked `action_head` without `vlm`/`gpu` run with mock backbone on CPU. VLM-dependent tests are auto-skipped. Verify with:
```bash
pytest tests/ -m "not slow and not gpu and not mujoco and not vlm" -v
```
