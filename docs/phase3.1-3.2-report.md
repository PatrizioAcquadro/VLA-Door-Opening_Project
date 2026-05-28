# Phase 3.1–3.2 Completion Report: VLM Backbone Integration & Action Head Implementation

**Project:** VLA-LEGO — Vision-Language-Action System for Bimanual Robotic LEGO Assembly
**Author:** Euge
**Date:** March 2026
**Hardware:** Lab PC (RTX 4090 24 GB, CUDA 12.8) and Gilbreth HPC (A100 80 GB)

---

## Executive Summary

Phases 3.1 and 3.2 establish the neural architecture of the VLA-LEGO system. Phase 3.1 integrates a pretrained 4.54-billion-parameter vision-language model (Qwen3.5-4B) as the frozen backbone, providing multimodal understanding of simulation images and text instructions. Phase 3.2 attaches a 20.5-million-parameter flow-matching action head that produces continuous 17-D robot action trajectories through iterative ODE denoising. Together, they compose a complete Vision-Language-Action model with Transfusion-style dual loss — autoregressive cross-entropy on text positions and flow matching velocity MSE on action positions — ready for Phase 3.3 training integration.

The entire architecture has been validated end-to-end on GPU with 160 automated tests and two standalone validation scripts (18 checks total, all passing). The action head adds only 0.75 GB of VRAM overhead on top of the backbone's 8.55 GB baseline, leaving ample room for training on the A100.

---

## 1. Phase 3.1 — VLM Backbone Integration

### 1.1 Objective

Load and validate a pretrained vision-language model as the VLA backbone, confirming that it accepts the project's simulation images (320x320, 4 camera views), produces usable hidden-state representations for downstream action prediction, and fits within the A100 80 GB training memory budget.

### 1.2 Model Selection: Qwen3.5-4B

We selected Qwen3.5-4B (`Qwen/Qwen3.5-4B`) for the following reasons:

- **Native multimodal architecture.** Qwen3.5 uses early vision-language fusion — the vision encoder is built into the model, not bolted on via an adapter. This eliminates the ViT alignment problem and simplifies the embedding pipeline.
- **Parameter-count parity with EO-1.** The reference architecture (EO-1, arXiv:2508.21112) uses Qwen2.5-VL-3B. Our 4B variant is the closest available in the Qwen3.5 family, sharing the same HuggingFace API surface while benefiting from the newer architecture.
- **Hardware compatibility.** The model fits both hardware tiers: inference on the lab PC (RTX 4090, 24 GB) and full training on the A100 (80 GB). A config-level switch to the 9B variant is trivial for future scale-up.
- **Shared model family.** Phase 2.2's annotation pipeline already uses Qwen3.5, enabling code reuse for tokenization and image preprocessing.

### 1.3 Architecture Constants (Qwen3.5-4B)

| Property | Value |
|----------|-------|
| Total parameters | 4.54 B |
| Hidden size | 2,560 |
| Transformer layers | 32 |
| Vocabulary size | 248,320 |
| Vision encoder | ~334 M params (`model.model.visual`) |
| Working dtype | bfloat16 |
| HuggingFace class | `Qwen3_5ForConditionalGeneration` |

### 1.4 Implementation (4 Subphases)

**3.1.0 — Dependencies & Configuration.** Added a `vlm` dependency group to `pyproject.toml` (`transformers>=4.49.0`, `accelerate>=0.30.0`, `Pillow`, `sentencepiece`, `torchvision`). Created two Hydra model configs: `vlm.yaml` for A100 production (flash_attention_2, 8192 max sequence length) and `vlm_dev.yaml` for RTX 4090 development (SDPA attention, 4096 max sequence).

**3.1.1 — Backbone Loading Pipeline.** Implemented `VLMBackbone(nn.Module)` in `models/vlm_backbone.py`, wrapping the HuggingFace model with a clean interface for downstream consumers. Key API:

- `get_hidden_states(input_ids, attention_mask, pixel_values, ...) -> (B, seq, 2560)` — last-layer hidden states for action head input
- `freeze_backbone()` / `unfreeze_backbone()` / `freeze_vision()` — training-time control
- `verify_backbone()` — automated parameter count, dtype, and NaN/Inf verification
- `.to()` override — safe no-op when `device_map="auto"` is active (prevents double-move errors in the training loop)

Extended `get_model(cfg)` in `models/utils.py` with architecture type routing: `"transformer"` (existing placeholder), `"vlm"` (backbone-only), and later `"vla"` (full model).

**3.1.2 — Tokenizer & Processor.** Implemented multimodal preprocessing utilities:

- `ProcessorInfo` (frozen dataclass) — vocabulary metadata, special token IDs, measured vision tokens per image
- `estimate_vision_tokens(backbone, width, height)` — measures actual token count by running a dummy image through the processor (not analytically estimated)
- `preprocess_images(backbone, images, text, device)` — converts NumPy uint8 RGB arrays (MuJoCo camera output) or PIL images into model-ready tensors via `processor.apply_chat_template()`
- `compute_context_budget(vision_tokens_per_image, ...)` — pure-function context window breakdown (vision + text + remaining for actions)

**3.1.3 — Inference Sanity Check.** Created `scripts/validate_vlm_backbone.py` — 8-check standalone validation producing artifacts to `logs/vlm_backbone/`. Checks include: model loading, processor, text-only forward, single-image forward, multi-view forward (4 images), hidden state extraction, text generation, and numerical sanity. All 8 checks pass on the lab PC.

**3.1.4 — Memory Profiling.** Created `scripts/profile_vlm_memory.py` — automated VRAM sweep with GPU tier auto-detection. Key findings on the RTX 4090:

| Configuration | Peak VRAM | Status |
|---------------|-----------|--------|
| Inference, seq=1024, 4 images | 9.2 GB | OK |
| Inference, seq=4096, 4 images | 11.2 GB | OK |
| Training, seq=2048, 4 images | 22.6 GB | OOM |

**Conclusion:** The RTX 4090 is suitable for inference and debugging. Full backbone fine-tuning requires the A100 or LoRA. Weight-only VRAM baseline is 8.5 GB (bfloat16).

### 1.5 Phase 3.1 Test Coverage

43 tests in `tests/test_vlm_backbone.py`:

| Test Class | Count | Scope |
|------------|-------|-------|
| `TestVLMBackboneConfig` | 5 | Config parsing (CPU) |
| `TestVLMBackboneLoading` | 8 | Model loading (GPU) |
| `TestVLMBackboneInfo` | 2 | Dataclass contracts |
| `TestBackwardCompatibility` | 2 | Existing models unaffected |
| `TestResolveDtype` | 4 | Dtype resolution logic |
| `TestProcessorInfo` | 2 | Tokenizer metadata (CPU) |
| `TestContextBudget` | 4 | Context window math (CPU) |
| `TestProcessorFunctions` | 8 | Image preprocessing (GPU) |
| `TestVLMInference` | 7 | End-to-end inference (GPU) |

19 tests run on CPU (CI-compatible); 24 require GPU + cached weights.

---

## 2. Phase 3.2 — Action Head Implementation

### 2.1 Objective

Implement the flow matching action head that attaches to the Phase 3.1 backbone and produces continuous 17-D robot action trajectories. The design follows the EO-1 architecture: a unified decoder-only transformer processes vision, language, robot state, and noisy action tokens in a single sequence, then a lightweight MLP decodes the hidden states at action positions into velocity predictions for ODE-based denoising.

### 2.2 Architectural Decisions

**Time convention.** EO-1 uses reversed time (t=0 is data, t=1 is noise). We adopt the standard OT-CFM convention (Lipman et al., 2023): t=0 is noise, t=1 is data. This aligns with the broader flow matching literature and makes the code more intuitive.

**Multimodal injection strategy.** The Qwen3.5-4B backbone enforces a strict XOR constraint: it accepts either `input_ids` or `inputs_embeds`, never both simultaneously. We therefore use **Scenario C (full embedding control)**: the VLA model manually assembles `inputs_embeds` by extracting text embeddings, scattering vision features at image-token positions, and concatenating state and action tokens — then passes the assembled sequence to the backbone with `input_ids=None`. This gives us complete control over the sequence layout.

**Float32 action head.** Following EO-1, all action head components (projectors, output head) operate in float32 for numerical stability, even though the backbone runs in bfloat16. Embeddings are cast to bfloat16 for sequence assembly, then back to float32 for velocity prediction.

**Action chunk size.** 16 steps at 20 Hz = 0.8 seconds of robot motion per prediction. This matches the project's control contract (Phase 1.1.5) and provides sufficient temporal coverage for LEGO pick-and-place motions.

### 2.3 Implementation (6 Subphases)

**3.2.0 — Action Chunk Contract.** Defined frozen constants and data structures in `models/action_head.py`:

| Constant | Value | Source |
|----------|-------|--------|
| `ACTION_CHUNK_SIZE` | 16 | 0.8 s at 20 Hz |
| `ACTION_DIM` | 17 | Phase 1.1.5 frozen action space |
| `STATE_DIM` | 52 | Phase 1.1.6 frozen robot state |
| `TOKENS_PER_ACTION_STEP` | 1 | One VLM token per action step |
| `TOKENS_PER_STATE` | 1 | One VLM token for full 52-D state |

Implemented `TokenType(IntEnum)` for per-position loss routing (`TEXT=0, IMAGE=1, STATE=2, ACTION=3`), `ActionChunkConfig` frozen dataclass, and chunking utilities (`chunk_actions`, `chunk_actions_batched`, `compute_action_context_tokens`).

Verified context budget: a single-placement episode (16 images, ~200 vision tokens each) requires ~3,661 sequence tokens; the worst-case Level 3 curriculum (32 images) requires ~7,261 — both fit within the 8,192 maximum sequence length.

**3.2.1 — Flow Matching Module.** Implemented `FlowMatchingModule(nn.Module)` — a parameter-free module encapsulating all OT-CFM math:

- `sample_timestep(B, device)` — Beta(1.5, 1.0) time sampling (EO-1 convention, biases toward noisier timesteps)
- `interpolate(x_data, noise, t)` — straight-line OT path: `x_t = (1-t)*noise + t*x_data`
- `target_velocity(x_data, noise)` — constant velocity field: `u_t = x_data - noise`
- `loss(pred_v, target_v, mask)` — per-step MSE averaged over action dimensions, masked mean over valid positions
- `denoise(predict_fn, shape, K)` — ODE integration from t=0 to t=1 with configurable solver (Euler, midpoint, RK4)

The identity property — that all three solvers recover exact data from noise when the velocity field is the true constant field — is verified in the test suite.

**3.2.2 — Robot State Projector.** `RobotStateProjector(nn.Module)`: projects the normalized 52-D robot state into a single hidden-dimension token for VLM sequence injection.

Architecture: `LayerNorm(52) -> Linear(52, 2560) -> SiLU -> Linear(2560, 2560) -> unsqueeze -> (B, 1, 2560)`

We use a 2-layer MLP with LayerNorm (rather than EO-1's simple linear projection) because the 52-D state vector has heterogeneous components — joint positions, angular velocities, quaternions, Cartesian end-effector poses — that benefit from normalization before projection. Parameter count: **6,691,944**.

**3.2.3 — Noisy Action Projector & Output Head.**

`NoisyActionProjector(nn.Module)`: embeds noisy action chunks and the flow matching timestep into hidden-dimension tokens.

Architecture: sinusoidal timestep embedding `(B, 1) -> (B, 256)` -> expand across chunk steps -> concatenate with noisy actions `(B, 16, 17+256)` -> 2-layer MLP -> `(B, 16, 2560)`. Parameter count: **7,257,600**.

`ActionOutputHead(nn.Module)`: decodes backbone hidden states at action positions to velocity predictions.

Architecture: `Linear(2560, 2560) -> SiLU -> Linear(2560, 17)`. Parameter count: **6,599,697**.

**Combined action head: 20,549,241 parameters (20.5 M) — 0.45% of the 4.54 B backbone.**

**3.2.4 — VLA Model Assembly.** Implemented `VLAModel(nn.Module)` in `models/vla_model.py`, composing the backbone with all action head components:

Sequence layout (per segment):
```
[text_tokens + scattered_vision | state_token(1) | action_tokens(16)]
<---------- seq_text ----------> <-- n_seg -----> <-- n_action ------>
```

Training forward pass (`forward(batch)`):
1. Sample flow matching timestep t ~ Beta(1.5, 1.0)
2. Interpolate noisy actions: `x_t = (1-t)*noise + t*x_data`
3. Extract text embeddings, scatter vision features at image token positions
4. Project state to hidden-dim token, project noisy actions + t to hidden-dim tokens
5. Assemble full sequence, forward through backbone (Scenario C: `inputs_embeds` only)
6. Compute dual loss: AR cross-entropy on text positions + FM velocity MSE on action positions
7. Return `{total_loss, text_loss, action_loss}` — model does NOT call `backward()`

Inference (`predict_actions(input_ids, attention_mask, robot_state, ...)`):
1. Compute context embeddings (text + vision + state) once
2. Sample initial noise `x_0 ~ N(0, I)` of shape `(B, 16, 17)`
3. Iterate K=10 ODE steps from t=0 to t=1, reusing context embeddings at each step
4. Return denoised action chunk `(B, 16, 17)` in float32

Created Hydra configs: `vla.yaml` (A100, flash_attention_2, seq=8192) and `vla_dev.yaml` (4090, SDPA, seq=4096).

**3.2.5 — End-to-End Validation & Profiling.** Created `scripts/validate_action_head.py` — 10-check standalone validation with dual-mode operation (real backbone on GPU, mock backbone on CPU).

### 2.4 Validation Results (Lab PC, RTX 4090)

#### Validation Script: 10/10 Checks Passed

| Check | Result | Detail |
|-------|--------|--------|
| Config parsing | PASS | `architecture.type=vla`, chunk_size=16, action_dim=17 |
| Component instantiation | PASS | 20.549 M params total, 47 ms init |
| Flow matching math | PASS | Boundary values exact, loss(v,v)=0 |
| Projector shapes | PASS | State: (B,52)->(B,1,H); Action: (B,16,17)+(B,1)->(B,16,H) |
| Output head shape | PASS | (B,16,H)->(B,16,17), all finite |
| VLA training forward | PASS | total_loss=19.38, text_loss=16.75, action_loss=2.63 |
| VLA inference | PASS | Shape (1,16,17), float32, all finite, 195 ms |
| Gradient routing | PASS | 14/14 action head params with grad; backbone clean |
| Numerical stability | PASS | Edge cases (t near 0, t near 1, zero state, 100x actions) all clean |
| **Memory overhead** | **PASS** | **Baseline 8.55 GB, peak 9.30 GB, overhead 0.750 GB** |

#### VRAM Budget Summary

| Component | VRAM (GB) |
|-----------|-----------|
| Backbone weights (bf16) | 8.55 |
| Action head forward + backward overhead | 0.75 |
| **Total VLA model** | **9.30** |
| A100 capacity | 80.00 |
| **Remaining for optimizer + batch** | **~70.70** |

The action head's 0.75 GB overhead is well within the 1.0 GB budget established during Phase 3.1 memory profiling. On the A100, the full VLA model leaves ~70 GB for optimizer states (Adam: ~16 GB for backbone, negligible for the 20.5 M action head), activations, and batched data — more than sufficient for training.

### 2.5 Phase 3.2 Test Coverage

117 tests across two test files:

**`tests/test_action_head.py` — 88 tests (all CPU-only):**

| Test Class | Count | Scope |
|------------|-------|-------|
| `TestActionChunkContract` | 13 | Frozen constants, config, chunking |
| `TestContextBudget` | 3 | Context window budget math |
| `TestTensorInterfaceShapes` | 6 | Shape contracts for all components |
| `TestFlowMatchingModule` | 27 | CFM math: interpolation, velocity, loss, solvers, identity |
| `TestRobotStateProjector` | 11 | Shape, NaN, gradients, magnitude, param count |
| `TestSinusoidalTimestepEmbedding` | 6 | Embedding correctness, edge cases |
| `TestNoisyActionProjector` | 12 | Shape, timestep, gradients, from_cfg variants |
| `TestActionOutputHead` | 10 | Shape, gradients, round-trip, from_cfg variants |

**`tests/test_vla_model.py` — 29 tests (20 CPU + 9 GPU):**

| Test Class | Count | Markers | Scope |
|------------|-------|---------|-------|
| `TestVLAModelConstruction` | 4 | CPU | Component composition, param counts |
| `TestSequenceAssembly` | 3 | CPU | Sequence layout, ordering, dtype |
| `TestVLAModelForward` | 4 | CPU | Loss structure, finite values |
| `TestVLAModelGradients` | 4 | CPU | Gradient routing, freeze/unfreeze |
| `TestVLAModelInference` | 3 | CPU | predict_actions shape, finite values |
| `TestBackwardCompatibility` | 2 | CPU | Existing model paths unaffected |
| `TestVLAModelGPU` | 9 | GPU | Real Qwen3.5-4B: loading, forward, gradients, inference |

CPU tests use a lightweight `MockVLMBackbone` (hidden_size=64) for fast CI execution. GPU tests load the real 4.54 B parameter backbone and validate the complete pipeline end-to-end.

---

## 3. Relationship to EO-1

The implementation is architecturally inspired by EO-1 (arXiv:2508.21112) but is entirely new code. Key adaptations:

| Aspect | EO-1 | VLA-LEGO |
|--------|------|----------|
| Backbone | Qwen2.5-VL-3B (2048 hidden) | Qwen3.5-4B (2560 hidden) |
| Action dim | Up to 32 | 17 (frozen, bimanual) |
| Chunk size | 50 steps | 16 steps (0.8 s at 20 Hz) |
| State dim | Varies by task | 52 (frozen, dual-arm EE poses + velocities) |
| Time convention | Reversed (t=0=data, t=1=noise) | Standard OT-CFM (t=0=noise, t=1=data) |
| State projector | Linear(state_dim, H) | LayerNorm + 2-layer MLP (heterogeneous state) |
| Action head dtype | float32 | float32 (adopted) |
| Time sampling | Beta(1.5, 1.0) | Beta(1.5, 1.0) (adopted) |
| Multimodal injection | Qwen2.5-VL native | Scenario C: manual embedding assembly (Qwen3.5 XOR constraint) |
| Loss | Flow matching MSE only | Transfusion dual loss (AR text + FM action) |

---

## 4. Deliverables Summary

### Source Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `models/vlm_backbone.py` | VLM backbone wrapper (Qwen3.5-4B) | ~650 |
| `models/action_head.py` | Action head components (FM, projectors, output head, chunking) | ~850 |
| `models/vla_model.py` | VLA model assembly (dual loss, ODE inference) | ~520 |
| `configs/model/vlm.yaml` | VLM config (A100) | 36 |
| `configs/model/vlm_dev.yaml` | VLM config (4090) | 30 |
| `configs/model/action_head.yaml` | Action head config | 50 |
| `configs/model/vla.yaml` | Full VLA config (A100) | 73 |
| `configs/model/vla_dev.yaml` | Full VLA config (4090) | 67 |
| `tests/test_vlm_backbone.py` | VLM backbone tests (43 tests) | ~800 |
| `tests/test_action_head.py` | Action head tests (88 tests) | ~950 |
| `tests/test_vla_model.py` | VLA model tests (29 tests) | ~680 |
| `scripts/validate_vlm_backbone.py` | 8-check VLM validation | ~410 |
| `scripts/profile_vlm_memory.py` | VRAM profiling sweep | ~500 |
| `scripts/validate_action_head.py` | 10-check action head validation | ~490 |
| `infra/gilbreth/job_templates/07_download_vlm_weights.sh` | SLURM: pre-cache weights | 215 |
| `infra/gilbreth/job_templates/09_validate_action_head.sh` | SLURM: action head validation | ~130 |

### Test Results

| Suite | Tests | Status |
|-------|-------|--------|
| `test_vlm_backbone.py` (CPU) | 19 | All pass |
| `test_vlm_backbone.py` (GPU) | 24 | All pass |
| `test_action_head.py` (CPU) | 88 | All pass |
| `test_vla_model.py` (CPU) | 20 | All pass |
| `test_vla_model.py` (GPU) | 9 | All pass |
| **Total** | **160** | **All pass** |

### Validation Scripts

| Script | Checks | Status |
|--------|--------|--------|
| `validate_vlm_backbone.py` | 8/8 | All pass (lab PC) |
| `validate_action_head.py` | 10/10 | All pass (lab PC) |

---

## 5. What Phase 3.3 Receives

Phase 3.3 (End-to-End Training Integration) inherits a fully tested VLA model with the following interface:

| Need | Interface |
|------|-----------|
| Model construction | `get_model(cfg)` with `architecture.type: "vla"` |
| Training forward pass | `model.forward(batch) -> {total_loss, text_loss, action_loss}` |
| Action prediction | `model.predict_actions(input_ids, attn_mask, robot_state) -> (B, 16, 17)` |
| Loss logging | Separate `text_loss` and `action_loss` for W&B tracking |
| Freeze control | `freeze_backbone()`, `unfreeze_backbone()`, `freeze_vision()` |
| Batch format | Documented in `VLAModel.forward()` docstring: `input_ids`, `pixel_values`, `attention_mask`, `robot_states`, `action_chunks`, `chunk_masks`, `text_labels` |
| Memory budget | 9.30 GB total on RTX 4090; ~70 GB remaining on A100 |

Phase 3.3's responsibility is to build the dataloader that produces batches matching this contract, wire the VLA model into the existing Trainer class, and implement the training loop with appropriate optimizer, scheduler, and checkpointing.
