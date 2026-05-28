# Phase 3.1 — VLM Backbone Integration (4 days)

**Goal:** Load and validate a pretrained Qwen3.5-4B Vision-Language Model as the VLA backbone — reusing and adapting the EO-1 backbone integration code wherever that code is available in the project copy, rather than building a parallel custom stack from scratch — confirming that it accepts our simulation images (320×320, 4 views), produces usable hidden states for downstream action prediction, and fits within the A100 80 GB training memory budget — so that Phase 3.2 can attach the action head with a known, profiled foundation.

**Fixed upstream decisions (from 1.2 + 2.1–2.3):**
- **Sim engine:** MuJoCo (MJCF-first, headless via EGL)
- **Robot:** IHMC Alex upper-body fixed-base, **17-D action space** (Δq arms + gripper), **52-D state vector**
- **Views:** 4 frozen cameras (overhead, left_wrist_cam, right_wrist_cam, third_person), 320×320, 20 Hz
- **Data format:** HDF5 episodes (Phase 2.1) + JSONL annotations (Phase 2.2) → reference-based interleaved sequences (Phase 2.3)
- **Training infrastructure:** Hydra configs, PyTorch DDP/DeepSpeed, Gilbreth A100 80 GB (2/node), lab PC RTX 4090 24 GB
- **Model family:** Phase 2.2 establishes Qwen3.5 (4B dev, 9B production) for annotation — Phase 3.1 uses the same family for the VLA backbone
- **Target architecture & codebase (EO-1):** Unified decoder-only transformer backbone + autoregressive action decoding + flow matching for continuous actions. EO-1 serves as both the architectural template and the preferred reusable codebase — implementation should reuse EO-1 code wherever it is available in the project copy

**Key Phase 3.1 stance:**
- **EO-1 code reuse first.** EO-1 is both the architectural template **and** the preferred reusable codebase. If EO-1 implementation code for backbone loading, processor setup, model wrapping, inference, or training-time integration is already available in the repository, it is the default implementation base. Phase 3.1 should inspect, reuse, and adapt that code to the chosen VLM wherever practical, and only introduce new standalone components where the inspected EO-1 code does not provide a realistic integration path. The first step of every subphase is to check whether EO-1 already solves the problem.
- **Qwen3.5-4B as the VLA backbone.** Qwen3.5 (released March 2026) is natively multimodal via early fusion — vision is built into the model, no separate ViT adapter required. The 4B variant is closest to EO-1's 3B parameter count, shares the model family with Phase 2.2's annotation pipeline (code reuse for loading, tokenization, processor), and fits both hardware tiers: lab PC (inference) and A100 (training). Config-level switch to 9B is trivial for scale-up experiments. Where EO-1 loads Qwen 2.5 VL, this phase adapts the same loading path for Qwen3.5-4B — the HuggingFace API surface is nearly identical.
- **Backbone-only scope.** This phase loads, validates, and profiles the pretrained VLM. It does **not** add action heads, modify the training loop, or integrate with the data pipeline. Those are Phase 3.2 (action head) and Phase 3.3 (training integration) concerns.
- **Backward compatible.** The existing placeholder `TransformerModel` continues to work for all current tests and configs. The new VLM loading path is a separate code branch triggered by `architecture.type: "vlm"` in the Hydra config.

**Critical gap this phase closes:**
The project currently has zero VLM infrastructure — no HuggingFace dependencies, no pretrained weight loading, no multimodal tokenization, no vision processing. The existing `TransformerModel` is a 25M-parameter placeholder that accepts continuous state vectors. The EO-1 codebase (if present in the project copy) already solves many of these problems for Qwen 2.5 VL — this phase's job is to identify which EO-1 modules can be reused directly, which need adaptation for Qwen3.5, and which gaps require new code. Without Phase 3.1, there is no foundation for the action head (Phase 3.2) or the training pipeline (Phase 3.3) to build on.

---

## 3.1.0) Dependencies & Model Configuration

### What we will do
Add the HuggingFace ecosystem to the project dependencies and create Hydra configurations for the Qwen3.5-4B VLM backbone, with separate profiles for production (A100) and development (lab PC).

### Why this matters
Every subsequent subphase depends on `transformers` being importable and the model configuration being parseable. Getting the dependency group and config structure right upfront prevents cascading rework. The two-tier config pattern (production vs dev) follows the established `base.yaml`/`large.yaml` convention and ensures the backbone works on both hardware tiers from day one.

### Design

**New dependency group** in `pyproject.toml`:

```toml
vlm = [
    "transformers>=4.49.0",
    "Pillow>=10.0.0",
    "accelerate>=0.30.0",
    "sentencepiece>=0.2.0",
]
```

Also append these four packages to the `dev` group. `torch`, `safetensors`, and `einops` are already core dependencies — no duplication needed.

**Production config** — `configs/model/vlm.yaml` (A100 target):

```yaml
name: vlm

architecture:
  type: "vlm"                          # routes get_model() to VLM path

vlm:
  model_id: "Qwen/Qwen3.5-4B"
  revision: null                       # pin after initial validation
  torch_dtype: "bfloat16"
  device_map: "auto"
  trust_remote_code: true
  attn_implementation: "flash_attention_2"
  max_seq_length: 8192
  cache_dir: "${oc.env:HF_HOME,${paths.root}/cache/huggingface}"

  processor:
    image_resolution: 320              # frozen camera contract
    padding: "longest"
    truncation: true

  freeze:
    backbone: true                     # frozen by default; Phase 3.3 controls unfreezing
    vision_encoder: true

_estimated_params_billions: 4.0
_estimated_memory_gb_bf16: 8.0
```

**Dev config** — `configs/model/vlm_dev.yaml` (RTX 4090 target):

Same structure, but with `attn_implementation: "sdpa"` (no FlashAttention-2 dependency), `max_seq_length: 4096` (shorter context for 24 GB headroom).

### Execution checklist
- Inspect EO-1 codebase for existing dependency declarations and config patterns for VLM backbone loading
- Add `vlm` dependency group to `pyproject.toml` (reuse EO-1's dependency list if available, adapting versions as needed); update `dev` group to include VLM deps
- Create `configs/model/vlm.yaml` (production) and `configs/model/vlm_dev.yaml` (dev) — adapt from EO-1 config structure if one exists, otherwise follow established `base.yaml`/`large.yaml` pattern
- Verify `pip install -e ".[vlm]"` succeeds and `import transformers` works
- Verify both configs parse via Hydra compose without error
- Verify existing `model=base` and `model=large` configs continue to parse

### Milestone (minimum success criteria)
- `pip install -e ".[vlm]"` installs all dependencies. `python -c "import transformers; print(transformers.__version__)"` succeeds. Both VLM Hydra configs parse. Existing model configs are unaffected.

---

## 3.1.1) Pretrained Weight Loading Pipeline

### What we will do
Inspect the EO-1 codebase for its backbone loading, model wrapping, and weight verification code. If EO-1 provides a clean loading path (e.g., a backbone wrapper module, a `get_model()` factory, or a weight verification routine), reuse and adapt it for Qwen3.5-4B. Only create a new `models/vlm_backbone.py` (or equivalent local wrapper) where the inspected EO-1 code does not already expose a usable integration point. Extend the existing `get_model()` factory to route `architecture.type: "vlm"` to the loading path (EO-1-derived or new) without breaking the placeholder `TransformerModel`.

### Why this matters
This is the foundation of the entire VLA model. Every downstream phase (action head, training, evaluation) depends on a correctly loaded, verified backbone with known hidden dimensions and dtype. Getting weight verification and cache management right avoids silent corruption and download failures on the cluster. EO-1 likely already solves most of this for Qwen 2.5 VL — adapting that proven code path is far safer than reimplementing from scratch.

### Design

**Implementation strategy — EO-1 reuse first:**

Before writing any new code, inspect the EO-1 codebase for:
- A backbone wrapper class (analogous to `VLMBackbone` below) that loads a pretrained VLM and exposes hidden states
- A model factory or loading function that handles HuggingFace `AutoModel` instantiation, dtype, device placement, freeze policy
- Weight verification or sanity checks
- Processor/tokenizer setup code

If EO-1 provides these, reuse them directly (importing or copying into `models/`) and adapt the model ID from `Qwen/Qwen2.5-VL-3B-Instruct` (or equivalent) to `Qwen/Qwen3.5-4B`. If EO-1's wrapper has a different interface than the one described below, prefer the EO-1 interface to minimize divergence — only extend it where our project has additional requirements (e.g., two-tier config, specific freeze granularity).

**If EO-1 does not provide a reusable loading path**, create `models/vlm_backbone.py` with these components:

| Component | Purpose |
|---|---|
| `VLMBackboneInfo` (frozen dataclass) | Architecture metadata: model_id, param counts, hidden_size, num_layers, vocab_size, dtype, vision_token_count |
| `VLMBackbone(nn.Module)` | Wraps HuggingFace model; exposes `get_hidden_states()`, `forward()`, `freeze_backbone()`, `unfreeze_backbone()`, `freeze_vision()` |
| `load_vlm_backbone(cfg) → VLMBackbone` | End-to-end loading: download/cache → dtype enforcement → device placement → freeze policy → verification |
| `verify_backbone(backbone) → bool` | Checks: param count in expected range, dtype correct, no NaN/Inf, hidden_size matches config |

**Key interface for Phase 3.2** (whether EO-1-derived or newly implemented):

```python
# Phase 3.2 will do:
hidden = backbone.get_hidden_states(input_ids, attention_mask, pixel_values, ...)
# → (B, seq_len, hidden_size)
# Then: action_logits = action_head(hidden[:, action_token_positions, :])
```

The backbone (EO-1-derived or new) must expose:
- `backbone.hidden_size` — for sizing the action head input dimension
- `backbone.info` — architecture metadata for logging and verification
- `backbone.processor` — for preparing multimodal inputs

**`models/utils.py`** modification — extend `get_model()`:

If EO-1 already has a model factory that routes by architecture type, adapt it. Otherwise, add:

```python
if model_type == "vlm":
    from models.vlm_backbone import load_vlm_backbone
    return load_vlm_backbone(cfg)
```

Lazy import avoids requiring `transformers` for non-VLM model types. The existing `"transformer"` branch remains unchanged.

**HuggingFace cache placement:**
- Gilbreth: `HF_HOME=${VLA_SCRATCH_ROOT}/cache/huggingface` (set in job scripts)
- Lab PC: defaults to `${paths.root}/cache/huggingface`

**SLURM weight download job** — `infra/gilbreth/job_templates/07_download_vlm_weights.sh`:

Standalone download job (no GPU needed, `--gres=gpu:0`) that pre-caches the ~8 GB model weights on scratch before training jobs start. Prevents download timeouts during GPU-allocated training runs.

**Test file** — `tests/test_vlm_backbone.py`:

| Test class | Scope | Markers | Count |
|---|---|---|---|
| `TestVLMBackboneConfig` | Config parsing (CPU-safe) | `vlm` | ~5 |
| `TestVLMBackboneLoading` | Actual model loading | `vlm`, `gpu`, `slow` | ~8 |
| `TestVLMBackboneInfo` | Dataclass validation | `vlm` | ~2 |
| `TestBackwardCompatibility` | Existing model unaffected | (none) | ~2 |

New `vlm` pytest marker in `tests/conftest.py` — auto-skipped when `transformers` is not installed, ensuring CI (which installs only `.[ci]`) is unaffected.

### Execution checklist
- **Inspect EO-1 codebase** for backbone wrapper, model loading, weight verification, and model factory code. Document what exists and what can be reused
- Reuse/adapt EO-1's backbone loading code for Qwen3.5-4B. Only introduce a new local wrapper module (e.g., `models/vlm_backbone.py`) where the inspected EO-1 code does not already expose a clean integration point
- Extend `get_model()` in `models/utils.py` with `"vlm"` routing (adapt EO-1's factory if it has one, otherwise add lazy import)
- Add `vlm` marker to `tests/conftest.py` with auto-skip logic
- Write tests in `tests/test_vlm_backbone.py` covering config, loading, verification, backward compat
- Create `infra/gilbreth/job_templates/07_download_vlm_weights.sh`
- Verify: `get_model(cfg)` with `model=vlm` returns the backbone (EO-1-derived or new); with `model=base` returns `TransformerModel`
- Verify: all existing `test_models.py` tests pass unchanged

### Milestone (minimum success criteria)
- The backbone loading path (EO-1-derived or new) loads Qwen3.5-4B on GPU with bf16 dtype. `backbone.info.param_count_total` is approximately 4B (within 10%). All parameters are bf16 with no NaN/Inf. `backbone.hidden_size` returns the model's hidden dimension. `get_model()` routes correctly for both `"vlm"` and `"transformer"` types.

---

## 3.1.2) Tokenizer & Processor Configuration

### What we will do
Reuse EO-1's processor/tokenizer setup code if available (adapting model ID and resolution as needed), or configure and validate the Qwen3.5 processor from scratch. The processor bundles text tokenizer + image preprocessor for our specific use case: 320×320 MuJoCo camera images and robot manipulation text. Determine the exact vision token count per image at our resolution — this number is critical for context window budgeting in Phase 3.2.

### Why this matters
The processor is the bridge between raw simulation data (numpy uint8 images, text strings) and model inputs (token IDs, pixel tensors). If the processor silently resizes, pads, or truncates our 320×320 images in an unexpected way, downstream training will fail or produce degraded results. The vision token count per image directly determines how many timesteps fit within the model's context window — a miscalculation here cascades into wrong sequence lengths for the entire training pipeline. EO-1 likely already has processor setup and vision token counting code for Qwen 2.5 VL — that code is the starting point.

### Design

**Implementation strategy — EO-1 reuse first:**

Inspect EO-1 for processor configuration, image preprocessing functions, and vision token counting utilities. If EO-1 already exposes these (e.g., a preprocessing pipeline, a token budget analysis, or a `ProcessorInfo`-equivalent), reuse them with model ID and resolution adjustments. Only create new functions where EO-1 does not cover the requirement.

**Functions (EO-1-derived or new) to be available in `models/vlm_backbone.py` (or equivalent EO-1 module):**

| Function | Purpose |
|---|---|
| `ProcessorInfo` (frozen dataclass) | Vocabulary size, special token IDs (image, BOS, EOS, pad), vision start/end tokens, `estimated_vision_tokens_per_image` |
| `get_processor_info(backbone) → ProcessorInfo` | Extracts all metadata; determines vision token count by running a dummy 320×320 image through the processor |
| `preprocess_images(backbone, images, text) → dict` | Takes numpy uint8 RGB arrays (MuJoCo format) + text prompt → returns model-ready tensors (`input_ids`, `attention_mask`, `pixel_values`, etc.) |
| `estimate_vision_tokens(backbone, width, height) → int` | Measures actual vision token count for a given resolution by counting tokens between vision start/end markers |

**Context window budget analysis** (key output of this subphase):

The estimated budget for an 8192-token context window (single-placement VLA sequence):

| Component | Token estimate |
|---|---|
| 4 images × ~N vision tokens/image | ~4N |
| Task description | ~40 |
| 8 step narrations × ~25 tokens | ~200 |
| Outcome text | ~5 |
| Action tokens (Phase 3.2, TBD) | Remaining budget |
| **Total text + vision** | **~4N + 245** |

The exact value of N (vision tokens per 320×320 image) will be measured empirically. For Qwen VL models with 14×14 patches and spatial merging, N is typically 130–260 tokens per image. This puts total vision+text at roughly 765–1,285 tokens, leaving 6,900–7,400 tokens for action sequences — ample for ~200 steps of 17-D actions.

### Execution checklist
- **Inspect EO-1 codebase** for processor setup, image preprocessing, and vision token counting code. Reuse/adapt where available
- Implement or adapt `ProcessorInfo`, `get_processor_info()`, `preprocess_images()`, `estimate_vision_tokens()` — placing them in the EO-1-derived module if reusing, or in `models/vlm_backbone.py` if new
- Run `estimate_vision_tokens(backbone, 320, 320)` and record the actual number
- Verify image preprocessing: 320×320 uint8 numpy array → valid `pixel_values` tensor (correct shape, non-zero)
- Verify text tokenization with robot manipulation vocabulary ("gripper", "baseplate", "stud", "2×4 brick", "press-fit")
- Verify multi-image preprocessing: 4 images (quad view) + text prompt → valid combined input
- Compute and log the full context window budget breakdown
- Add processor tests to `tests/test_vlm_backbone.py` (processor loading, vision token count, text tokenization, image preprocessing, multi-image)

### Milestone (minimum success criteria)
- `ProcessorInfo.estimated_vision_tokens_per_image` populated with the measured value for 320×320 input. `preprocess_images()` accepts numpy uint8 arrays (MuJoCo render output) and returns valid model inputs. Context window budget logged: "4 views × N tokens = M total vision tokens per step, leaving K tokens for text+actions within 8192 context."

---

## 3.1.3) VLM Inference Sanity Check

### What we will do
Run end-to-end forward passes with actual MuJoCo simulation images and robot manipulation prompts — verifying that the loaded backbone (EO-1-derived or new) produces valid logits, non-NaN hidden states, and coherent text generation. If EO-1 includes inference scripts or test utilities, adapt them for our sim images and prompts. Create a standalone validation script following the established `scripts/validate_*.py` pattern.

### Why this matters
Loading weights and configuring the processor is necessary but not sufficient. We must confirm the full pipeline works with our actual domain data before investing in training infrastructure. A forward pass that produces NaN, crashes on our image resolution, or generates incoherent text indicates a silent integration bug that would waste GPU-hours in later phases. EO-1 may already have inference validation code that can be adapted rather than rewritten.

### Design

**Validation script** — `scripts/validate_vlm_backbone.py`:

Following the pattern of `scripts/validate_cameras.py` and `scripts/validate_lego_task.py`, this script runs 8 sequential checks and produces artifacts to `logs/vlm_backbone/`.

| Check | Input | Verifies |
|---|---|---|
| 1. Model loading | Config | Correct dtype, param count, no NaN in weights |
| 2. Processor | 320×320 dummy image | Image preprocessing produces valid tensors |
| 3. Text-only forward | Text prompt | Logit shape = (1, seq, vocab), no NaN |
| 4. Single-image forward | 1 overhead sim image + prompt | Valid logits with vision+text input |
| 5. Multi-view forward | 4 sim images + task instruction | Valid logits with quad-view input |
| 6. Hidden state extraction | Same as check 5 | `get_hidden_states()` returns (1, seq, hidden_size) |
| 7. Text generation | 1 overhead sim image + manipulation prompt | Non-empty generated text, valid token IDs |
| 8. Numerical sanity | All outputs from checks 3–7 | No NaN/Inf in any output tensor |

**Sim image acquisition:** Load `alex_upper_body` scene via `load_scene()`, step to `rest` keyframe, render 4 views at 320×320 using `MultiViewRenderer`.

**Test prompt:** `"The robot is performing a LEGO assembly task. Describe what you see and what the robot should do next."`

**Artifacts:** `logs/vlm_backbone/validation_report.json` (all check results), `logs/vlm_backbone/sample_generation.txt` (generated text), `logs/vlm_backbone/hidden_states_shape.json` (shape metadata).

**Tests added to `tests/test_vlm_backbone.py`:**

`TestVLMInference` class (markers: `vlm`, `gpu`, `mujoco`, `slow`) — 7 tests covering forward passes with text-only, single-image, multi-view inputs; hidden state shape; output numerical validity; text generation.

### Execution checklist
- **Inspect EO-1 codebase** for inference scripts, forward-pass test utilities, or validation code. Adapt for our sim images/prompts if available
- Create `scripts/validate_vlm_backbone.py` with 8 checks and artifact output (reusing EO-1 inference patterns where applicable)
- Add `TestVLMInference` tests to `tests/test_vlm_backbone.py`
- Run validation script on lab PC with `model=vlm_dev` and on A100 with `model=vlm`
- Verify all 8 checks pass on both environments
- Inspect generated text for basic coherence (not gibberish)
- Save artifacts to `logs/vlm_backbone/`

### Milestone (minimum success criteria)
- Forward pass with 4 sim images + manipulation instruction completes without error on both hardware tiers. Output logits have shape `(1, seq_len, vocab_size)` with no NaN or Inf. `get_hidden_states()` returns `(1, seq_len, hidden_size)`. Generated text is non-empty and consists of valid tokens. Validation script exits with 8/8 checks passed.

---

## 3.1.4a) Lab PC Memory Profiling (RTX 4090 24 GB)

### What we will do
Build `scripts/profile_vlm_memory.py` and run a 4090-appropriate sweep. The script auto-detects GPU VRAM and selects the right sweep automatically. This validates the profiling infrastructure locally and produces useful dev-tier baseline numbers. The same script runs on A100 in 3.1.4b with zero rework.

### Why this matters
Developing and validating the profiling script locally avoids burning cluster allocation on script bugs. The 4090 numbers establish which configurations fit in 24 GB for local dev/debug workflows. The memory profile is the single most important input to Phase 3.2 (action head design) and Phase 3.3 (training hyperparameters).

### Design

**Profiling script** — `scripts/profile_vlm_memory.py`:

Auto-detects GPU via `torch.cuda.get_device_properties().total_mem`:
- <=32 GB → 4090-tier sweep (7 configs, max seq=4096, max batch=2 training)
- \>32 GB → A100-tier sweep (11 configs, adds seq=8192, batch=4 training)

**4090-tier sweep (7 configs):**

| Seq Length | Batch | Mode | Images | Purpose |
|---|---|---|---|---|
| 1024 | 1 | inference | 0 | Baseline text-only |
| 1024 | 1 | inference | 4 | Quad-view inference |
| 2048 | 1 | inference | 4 | Typical VLA sequence |
| 4096 | 1 | inference | 4 | Max dev context |
| 2048 | 1 | training | 4 | Training baseline |
| 2048 | 2 | training | 4 | Training batch=2 (may OOM) |
| 4096 | 1 | training | 4 | Long training (may OOM) |

Quick mode (`--quick`) runs only seq_length=2048, batch_size=1, both modes.

**OOM handling:** Each configuration is wrapped in try/except `torch.cuda.OutOfMemoryError`. On OOM, the script reports peak VRAM before crash, marks the config as "OOM", and continues to the next configuration.

**Metrics per configuration:**
- Peak VRAM: `torch.cuda.max_memory_allocated()` (after cache clear + forward + optional backward)
- Forward/backward wall time (ms)
- Remaining VRAM: total GPU memory − peak
- KV cache analytical estimate: `2 × num_layers × batch × seq × num_kv_heads × head_dim × dtype_bytes`

**Training mode:** Temporarily unfreezes backbone, runs forward + backward (full gradient computation). This is worst-case memory — does NOT include optimizer states (Adam would add ~2x model size). LoRA fine-tuning would use less memory.

**Artifacts** saved to `logs/vlm_memory/`:
- `memory_profile.json` — full structured results
- `memory_table.txt` — formatted ASCII table
- `action_head_budget.json` — remaining VRAM for each training config

### Execution checklist
- Implement `scripts/profile_vlm_memory.py` with auto-detect sweep, OOM handling, metrics collection, and report generation
- Run on RTX 4090 with `--model-config vlm_dev` — confirm all inference configs complete without OOM
- Confirm at least training batch=1 at seq=2048 completes
- Verify artifacts are produced in `logs/vlm_memory/`
- Update `CLAUDE.md` with profiling commands and artifact documentation

### Milestone (minimum success criteria)
- Profiling script runs on RTX 4090 and produces JSON report + ASCII table. Weight-only VRAM measured. All inference configs complete. At least one training config completes. Artifacts saved to `logs/vlm_memory/`.

---

## 3.1.4b) A100 Memory Profiling & Production Characterization

### What we will do
Run the validated profiling script (from 3.1.4a) on A100 with the full production sweep. Compute the action head VRAM budget that Phase 3.2 depends on. Create a SLURM job template.

### Why this matters
The A100 numbers are the authoritative input to Phase 3.2 (action head sizing) and Phase 3.3 (training batch size, gradient checkpointing decisions). The script is already validated from 3.1.4a — this is purely execution + analysis.

### Design

**A100-tier sweep (11 configs)** — includes all 7 4090-tier configs (for direct comparison) plus:

| Seq Length | Batch | Mode | Images | Purpose |
|---|---|---|---|---|
| 8192 | 1 | inference | 4 | Max context |
| 2048 | 4 | training | 4 | Training batch=4 |
| 4096 | 2 | training | 4 | Long training batch=2 |
| 4096 | 4 | training | 4 | Long training batch=4 |

The script auto-detects A100 (>32 GB) and runs the extended sweep automatically.

**SLURM job template** — `infra/gilbreth/job_templates/08_profile_vlm_memory.sh`:

Single A100, 1 hour, standby QOS. Requires weights pre-cached via job 07.

### Execution checklist
- Create `infra/gilbreth/job_templates/08_profile_vlm_memory.sh`
- Run on A100 — confirm all 11 configurations complete without OOM
- Verify KV cache analytical estimate is within 20% of measured overhead
- Compute action head VRAM budget for target training configurations (seq=2048/4096, batch=2/4)
- Produce 4090 vs A100 comparison notes
- Save artifacts to `logs/vlm_memory/`

### Milestone (minimum success criteria)
- Full 11-config sweep on A100 completes. Action head VRAM budget quantified for seq={2048,4096}, batch={2,4}. Report saved to `logs/vlm_memory/`. SLURM job template created.

---

# Downstream Contract with Phase 3.2

Phase 3.2 (Action Head) will consume Phase 3.1's backbone as follows:

| Phase 3.2 Need | Phase 3.1 Provider | Interface |
|---|---|---|
| Hidden states at action positions | backbone wrapper `.get_hidden_states()` (or EO-1 equivalent) | `(B, seq, hidden_size)` tensor |
| Hidden dimension for head sizing | backbone wrapper `.hidden_size` (or EO-1 equivalent) | int property |
| Multimodal input preparation | `preprocess_images()` (or EO-1 preprocessing path) | numpy → model tensors |
| Freeze/unfreeze for LoRA | `freeze_backbone()` / `unfreeze_backbone()` (or EO-1 equivalent) | method calls |
| Context budget for action tokens | processor info `estimated_vision_tokens_per_image` + context budget analysis | int + documented budget |
| Memory budget for action head | `logs/vlm_memory/action_head_budget.json` | measured VRAM headroom |
| Model config reference | `configs/model/vlm.yaml` | Hydra config |

**Key design constraint:** Phase 3.1's backbone wrapper (whether EO-1-derived or newly implemented) is a pure wrapper — it does not modify the pretrained model's architecture. Phase 3.2 will add new `nn.Module` components (action head MLP, flow matching decoder) that consume the backbone's hidden states. This separation ensures the pretrained weights remain intact and verifiable.

**EO-1 provenance note:** If the backbone wrapper, processor utilities, or factory routing are adapted from EO-1 code, the implementation should document the EO-1 source files and what was changed (model ID, resolution, config keys, etc.). This makes it straightforward to pull in future EO-1 improvements.

---

# Startup-Grade Outputs (deliverables by end of 3.1)
- **VLM backbone module** — EO-1-derived or new `models/vlm_backbone.py` (or equivalent) with backbone wrapper, verified weight loading, hidden state extraction, and multimodal preprocessing. Provenance documented (which parts from EO-1, which newly written)
- **Two-tier Hydra configs** — `configs/model/vlm.yaml` (A100 production) and `configs/model/vlm_dev.yaml` (lab PC), adapted from EO-1 config structure where applicable
- **Vision token characterization** — measured vision tokens per 320×320 image, context window budget analysis for VLA sequence design
- **Inference validation** — 8-check validation script confirming end-to-end forward pass with real MuJoCo sim images
- **A100 memory profile** — measured VRAM table across sequence lengths and batch sizes, with quantified action head budget for Phase 3.2
- **Cluster support** — SLURM jobs for weight pre-caching and memory profiling on Gilbreth

---

# Files Inventory

### EO-1 reuse expectations

The exact file list below is **provisional** — it assumes EO-1 does not already provide a directly usable module for each item. During implementation, the first step of each subphase is to inspect the EO-1 codebase. The actual outcome for each file will be one of:

| Category | Meaning | Example |
|---|---|---|
| **Reused directly** | EO-1 module imported or copied as-is (model ID updated in config, not in code) | Model factory routing, weight download script |
| **Adapted from EO-1** | EO-1 module copied and modified (model ID, resolution, config keys, additional methods) | Backbone wrapper, processor setup, preprocessing |
| **Newly implemented** | No EO-1 equivalent found; written from scratch following project conventions | Validation scripts with MuJoCo sim images, SLURM job templates for our cluster |

The implementer should document the actual category for each file at implementation time.

### New or adapted files (up to 8)

| File | Purpose | Expected EO-1 reuse |
|---|---|---|
| `models/vlm_backbone.py` | Core VLM backbone module | **Adapt** from EO-1 backbone wrapper if available; new only if EO-1 has no equivalent |
| `configs/model/vlm.yaml` | Production VLM config (A100) | **Adapt** from EO-1 model config if available |
| `configs/model/vlm_dev.yaml` | Dev VLM config (lab PC) | **Adapt** (two-tier pattern is project-specific, but config keys may come from EO-1) |
| `tests/test_vlm_backbone.py` | VLM backbone tests (~25 tests) | **New** (project-specific test suite, but test patterns may mirror EO-1 tests) |
| `scripts/validate_vlm_backbone.py` | Standalone 8-check validation script | **New** (uses our MuJoCo sim images; may reuse EO-1 inference patterns) |
| `scripts/profile_vlm_memory.py` | Memory profiling script | **Adapt** from EO-1 profiling code if available; **new** otherwise |
| `infra/gilbreth/job_templates/07_download_vlm_weights.sh` | SLURM weight pre-download | **New** (cluster-specific) |
| `infra/gilbreth/job_templates/08_profile_vlm_memory.sh` | SLURM memory profiling | **New** (cluster-specific) |

### Modified files (5)

| File | Change | Expected EO-1 reuse |
|---|---|---|
| `pyproject.toml` | Add `vlm` dependency group; update `dev` group | **Adapt** dependency list from EO-1 if available |
| `models/utils.py` | Add `"vlm"` branch to `get_model()` (lazy import) | **Adapt** from EO-1 model factory if it has one |
| `models/__init__.py` | Add docstring note about lazy VLM import | **New** (minor) |
| `tests/conftest.py` | Add `vlm` pytest marker + auto-skip when transformers missing | **New** (project-specific CI gating) |
| `CLAUDE.md` | Add VLM commands, architecture docs, dependency group, pytest marker | **New** (project-specific) |

---

# Phase 3.1 Definition of Done

Phase 3.1 is complete when:
- The `vlm` dependency group is defined and `pip install -e ".[vlm]"` succeeds.
- Qwen3.5-4B loads from HuggingFace with bf16 dtype on both lab PC and A100, with verified parameter count and no NaN/Inf.
- The Qwen3.5 processor correctly preprocesses 320×320 uint8 images (MuJoCo format) and robot manipulation text.
- Vision tokens per image are measured and the context window budget is documented.
- End-to-end forward pass with 4 sim camera views + manipulation prompt produces valid logits and hidden states.
- Text generation from sim images produces non-empty, coherent output.
- A100 memory profile exists with measured VRAM for inference and training modes across sequence lengths and batch sizes.
- Action head VRAM budget is quantified for Phase 3.2.
- All existing tests (`test_models.py`, etc.) continue to pass — zero regressions.
- `get_model(cfg)` correctly routes to the EO-1-derived or newly implemented VLM backbone path for `model=vlm` and to `TransformerModel` for `model=base`.
- Validation script (`scripts/validate_vlm_backbone.py`) exits with 8/8 checks passed.
- SLURM job templates for weight download and memory profiling are tested on Gilbreth.
- `CLAUDE.md` is updated with VLM commands, module docs, and dependency information.

---

# Verification Plan

### Local (lab PC, RTX 4090)
```bash
pip install -e ".[vlm]"
python -c "import transformers; print(transformers.__version__)"
pytest tests/test_vlm_backbone.py -v -m "not slow"          # config + dataclass tests
pytest tests/test_vlm_backbone.py -v                          # full suite (requires GPU)
pytest tests/test_models.py -v                                # backward compat
python scripts/validate_vlm_backbone.py --model-config vlm_dev
python scripts/profile_vlm_memory.py --quick
```

### Cluster (Gilbreth A100)
```bash
sbatch infra/gilbreth/job_templates/07_download_vlm_weights.sh   # once
sbatch infra/gilbreth/job_templates/08_profile_vlm_memory.sh
python scripts/validate_vlm_backbone.py
python scripts/profile_vlm_memory.py
```

### CI (no changes needed)
Existing CI installs `.[ci]` which excludes `transformers`. All VLM tests are auto-skipped via `@pytest.mark.vlm`. Verify with:
```bash
pytest tests/ -m "not slow and not gpu and not mujoco" -v     # existing CI command
```
