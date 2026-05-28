# Phase 4.2 — Distributed Training Setup (3 days)

**Goal:** Upgrade the training infrastructure from the current limited distributed scaffold (basic DDP, rank-0-only checkpointing, no DeepSpeed, no bf16, no gradient accumulation) to a production-quality distributed training pipeline — integrating DeepSpeed ZeRO-1, bf16 mixed precision, gradient accumulation, distributed-safe checkpointing, and W&B experiment tracking with the VLA model interface from Phases 3.2 and 4.1 — validated at 8-GPU scale (4 nodes x 2 A100 80 GB) on Gilbreth with measured throughput baselines, so that the subsequent training phase can launch full-scale VLA pre-training runs with confidence in the distributed mechanics.

**Fixed upstream decisions (from 1.1-1.2 + 2.1-2.3 + 3.1-3.2 + 4.1):**
- **Action space:** Frozen 17-D `[dq_spine(1), dq_left_arm(7), dq_right_arm(7), gripper_left(1), gripper_right(1)]`, arm actions normalized `[-1, 1]`, gripper actions absolute `[0, 1]`
- **Robot state:** Frozen 52-D normalized vector `[q(15), q_dot(15), gripper(2), left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), left_ee_vel(3), right_ee_vel(3)]`
- **VLM backbone:** Qwen3.5-4B loaded in Phase 3.1, ~4B params, bf16, `get_hidden_states()` -> `(B, seq_len, hidden_size)`
- **Action head:** Phase 3.2 FlowMatchingModule, chunk_size=16, action_dim=17
- **VLAModel:** Phase 4.1 `forward(batch)` returns `{total_loss, text_loss, action_loss, metrics}` — four keys only (Phase 3.2.4 base contract + Phase 4.1 `metrics` dict)
- **Loss modules:** Phase 4.1 `VLATextLoss`, `VLAActionLoss`, `VLACombinedLoss` — loss for backprop is `output["total_loss"]`, a scalar tensor with grad
- **Batch format:** Phase 3.2.4 dict: `{input_ids, pixel_values, attention_mask, robot_states, action_chunks, chunk_masks, token_type_ids, text_labels}`
- **Training data:** Phase 2.3 reference-based JSONL -> HDF5 (not yet implemented; validate with synthetic data matching the batch format)
- **Cluster:** Gilbreth — 4 nodes x 2 A100 80 GB, 100 Gbps InfiniBand, SLURM, `srun + torchrun` launcher
- **Distributed stack:** DeepSpeed ZeRO-1 (decided in `configs/cluster/gilbreth.yaml`), NCCL backend
- **Tracking:** W&B via `ExperimentTracker` (rank-0 only), GPU monitoring, `ThroughputTracker`
- **Optimizer:** AdamW, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, cosine schedule with 1000-step warmup
- **Precision:** bf16 (decided in `configs/trainer/default.yaml` and `configs/deepspeed/zero1.json`)

**Key Phase 4.2 stance:**
- **Adapt existing infrastructure.** The project already has a `Trainer` class, DeepSpeed configs, SLURM templates, and W&B tracking modules. Phase 4.2 upgrades these existing components rather than building a parallel training stack. No EO-1 source code exists in the repository; the project's own infrastructure is the codebase to build on.
- **Synthetic VLA batches for validation.** Phase 2.3 training data does not yet exist. Phase 4.2 validates all distributed mechanics (sharding, checkpointing, throughput) using a `SyntheticVLADataset` that produces batches matching the Phase 3.2.4 format. The real VLA dataloader plugs in later without changes to the distributed plumbing.
- **DeepSpeed as production default, DDP as local fallback.** DeepSpeed ZeRO-1 is required for 8-GPU training of a 4B model (optimizer state sharding: ~32 GB total sharded to ~4 GB per GPU). Local development and CI continue to use vanilla DDP or single-GPU mode. The config switch (`cluster=gilbreth` vs `cluster=local`) selects the mode — no code changes needed.
- **Correctness before performance.** Distributed training bugs (wrong sharding, corrupted checkpoints, inconsistent data sampling) are silent and catastrophic. Every subphase prioritizes verifiable correctness over throughput optimization.

**Critical gap this phase closes:**
The current `Trainer` class in `train/trainer.py` has basic DDP initialization and a working training loop, but lacks the distributed features required for production VLA training: it saves checkpoints only on rank 0 (incompatible with DeepSpeed sharded optimizer state), has no gradient accumulation, no bf16 mixed precision, no warmup scheduler (TODO at line 153), and no W&B tracking integration. The SLURM template `04_smoke_8gpu_deepspeed.sh` runs a standalone dummy model — it does not use the project's `Trainer`, `VLAModel`, or data pipeline. Without Phase 4.2, there is no tested path from "model + loss modules work on 1 GPU" to "training runs correctly at 8-GPU scale."

**EO-1 code reuse assessment:**
No EO-1 source code is present in the local repository. However, EO-1 is an open-source project and its public implementation may be consulted or adapted for distributed training patterns (e.g., DeepSpeed integration, checkpoint format, data pipeline structure) if external reuse is permitted. EO-1 is referenced throughout this project as the architectural template (Qwen backbone, Transfusion dual loss, flow matching). The primary implementation baseline for Phase 4.2 is the project's own existing codebase; EO-1's public code serves as a secondary reference for patterns not yet established locally:

| Component | Status | Phase 4.2 action |
|---|---|---|
| `train/trainer.py` | Exists (basic DDP, 414 lines) | **Adapted** — add DeepSpeed, bf16, grad accum, W&B, warmup, collective checkpointing |
| `data/loader.py` | Exists (`DistributedSampler`, 81 lines) | **Adapted** — add collate_fn parameter |
| `configs/deepspeed/zero1.json` | Exists (ZeRO-1, bf16, auto batch) | **Adapted** — add scheduler section, Hydra overlay pattern |
| `configs/cluster/gilbreth.yaml` | Exists (full cluster config) | **Unchanged** — consumed as-is |
| `tracking/experiment.py` | Exists (complete ExperimentTracker) | **Unchanged** — wired into Trainer |
| `tracking/metrics.py` | Exists (ThroughputTracker, grad norm) | **Unchanged** — consumed via tracker |
| `infra/gilbreth/job_templates/04_*` | Exists (standalone 8-GPU smoke test) | **Reference pattern** — new templates use same launcher structure |
| `data/synthetic_vla.py` | Does not exist | **Newly implemented** — no upstream equivalent |
| `data/collator.py` | Does not exist | **Newly implemented** — no upstream equivalent |
| Benchmark/validation scripts | Do not exist | **Newly implemented** — follow `scripts/validate_*.py` project pattern |

---

## 4.2.0) DeepSpeed-Integrated Training Loop

### What we will do
Upgrade the `Trainer` class to support DeepSpeed ZeRO-1 initialization, bf16 mixed precision, gradient accumulation, warmup + cosine LR scheduler, W&B `ExperimentTracker` integration, and epoch-aware distributed sampling — while preserving backward compatibility with the existing DDP path for local development.

### Why this matters
The current Trainer uses vanilla DDP only. A 4B VLM at bf16 uses ~8 GB for model weights, but AdamW optimizer state (fp32 master weights + momentum + variance) adds ~32 GB. Without ZeRO-1 to shard this across 8 GPUs (~4 GB per GPU), optimizer state alone exceeds a single A100's working memory budget when combined with activations and gradients. Additionally, gradient accumulation is essential for reaching effective batch sizes that stabilize VLA training without exceeding per-GPU memory limits. The existing W&B `ExperimentTracker` is fully implemented but not wired into the training loop — metrics are currently written only to Python logging.

### Design

**Mode selection via config — no code changes to switch modes:**
```
if cfg.cluster.distributed.deepspeed.enabled:
    # DeepSpeed: model_engine owns optimizer, scheduler, gradient accumulation
    model_engine, optimizer, _, scheduler = deepspeed.initialize(...)
else:
    # DDP: existing behavior (local dev, single GPU, CI)
    model = DDP(model, device_ids=[local_rank])
```

The cluster config (`cluster=local` vs `cluster=gilbreth`) drives this switch. This is a critical branching point — DeepSpeed wrapping and DDP wrapping are **mutually exclusive**.

**DeepSpeed initialization in `Trainer.setup()`:**
- Load `VLAModel` (or placeholder `TransformerModel` for smoke tests) to device
- When DeepSpeed is enabled: **skip** `_create_optimizer()` and `_create_scheduler()` — DeepSpeed owns both
- Call `deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())` which returns `(model_engine, optimizer, _, scheduler)`
- Store `self.model_engine` — all forward/backward/step calls route through it
- When DeepSpeed is **not** enabled: use existing DDP wrapping + new warmup scheduler

**`_build_deepspeed_config()` helper (bridges Hydra and DeepSpeed):**
- Reads `configs/deepspeed/zero1.json` as base template
- Overlays Hydra config values: `train_micro_batch_size_per_gpu` from `cfg.trainer.training.batch_size_per_device`, `gradient_accumulation_steps` from `cfg.trainer.gradient.accumulation_steps`, optimizer params from `cfg.trainer.optimizer`, scheduler params from `cfg.trainer.scheduler`
- Returns a dict for `deepspeed.initialize(config=...)`
- Eliminates the "two sources of truth" problem between Hydra and DeepSpeed JSON configs

**Training loop modifications:**
- **Forward pass**: change from `self.model(batch["input_ids"], batch["attention_mask"])` to `output = model(batch)` (Phase 4.1 VLAModel contract). Extract `total_loss`, `text_loss`, `action_loss`, `metrics`. Fall back to current MSE path if model returns old format (backward compat).
- **Backward pass**: DeepSpeed: `model_engine.backward(output["total_loss"])` then `model_engine.step()`. DDP: `output["total_loss"].backward()` then `optimizer.step()`.
- **Gradient clipping**: DeepSpeed handles it internally (`"gradient_clipping": 1.0` in config) — disable the Trainer's manual `clip_grad_norm_()` in DeepSpeed mode to avoid double-clipping.
- **Gradient accumulation**: DeepSpeed handles accumulation natively via `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps`. DDP path: manual accumulation with `loss = loss / accumulation_steps`, step optimizer every N micro-batches.
- **Epoch-aware sampler**: add `sampler.set_epoch(epoch)` at the start of each epoch loop (missing in current code — critical for proper shuffling in distributed mode).
- **W&B tracking**: call `tracker.log_training_step(loss=..., step=..., batch_size=..., optimizer=..., model=..., loss_ar=..., loss_fm=..., extra_metrics=output["metrics"])` at the logging interval. Call `tracker.start_throughput_tracking()` at training start and `tracker.finish()` at end.
- **Optimizer zero_grad**: skip `self.optimizer.zero_grad()` in DeepSpeed mode — the engine handles it.

**Warmup + cosine LR scheduler:**
- DeepSpeed path: use DeepSpeed's built-in `WarmupDecayLR` configured via the DeepSpeed config JSON. This correctly handles gradient accumulation step counting (steps per optimizer step, not per micro-batch).
- DDP path: `torch.optim.lr_scheduler.SequentialLR` composing `LinearLR` (warmup from `lr * 0.01` to `lr` over `warmup_steps`) + `CosineAnnealingLR` (decay to `min_lr_ratio * lr`). This closes the TODO at `trainer.py` line 153.

**Backward compatibility:**
- `model=base cluster=local`: existing behavior unchanged (DDP, TransformerModel, MSE loss)
- `model=vlm cluster=gilbreth`: DeepSpeed + VLAModel path
- All existing tests pass without modification

### Execution checklist
- Add DeepSpeed initialization branch to `Trainer.setup()` — create `model_engine` via `deepspeed.initialize()`; skip manual optimizer/scheduler creation
- Implement `_build_deepspeed_config()` to bridge Hydra config values into DeepSpeed JSON format
- Refactor training loop: route forward/backward/step through `model_engine` in DeepSpeed mode; consume `VLAModel.forward(batch)` output dict
- Disable manual `clip_grad_norm_()` and `optimizer.zero_grad()` when DeepSpeed is active
- Add gradient accumulation support (DeepSpeed-native; manual division for DDP path)
- Add bf16 `torch.autocast` for DDP path (DeepSpeed handles via config)
- Integrate `ExperimentTracker` — create via `create_tracker(cfg)` in `setup()`, log in training loop, finish at end
- Implement warmup scheduler: `WarmupDecayLR` for DeepSpeed, `SequentialLR` for DDP
- Add `sampler.set_epoch(epoch)` call at each epoch start
- Create `configs/cluster/local_deepspeed.yaml` — local multi-GPU config with DeepSpeed enabled, loopback network, no SLURM dependencies (portable for lab PC testing)
- Create `configs/trainer/distributed.yaml` — production overrides (batch_size=4, accumulation=4, checkpoint every 1000 steps)
- Create `configs/deepspeed/zero1_vla.json` — extends `zero1.json` with scheduler section and Hydra overlay placeholders
- Smoke test: `python -m train.trainer trainer=debug cluster=local` still works (DDP path, no regressions)
- DeepSpeed test: 2-GPU with `torchrun --nproc_per_node=2 -m train.trainer trainer=debug cluster=local_deepspeed data.dataset.name=synthetic_vla` — loss decreases, W&B logs appear, LR follows warmup+cosine curve

### Milestone (minimum success criteria)
- `Trainer.setup()` correctly initializes DeepSpeed ZeRO-1 when `cluster.distributed.deepspeed.enabled=true`. Training loop completes 100 steps with gradient accumulation and bf16 on 2 GPUs. W&B logs show loss (total, text, action), learning rate (warmup ramp visible), gradient norm, and throughput metrics. Existing `trainer=debug cluster=local` smoke test still passes unchanged.

---

## 4.2.1) Distributed Data Pipeline

### What we will do
Create a `SyntheticVLADataset` that generates batches matching the Phase 3.2.4 format with variable-length sequences, implement a `VLACollator` for padding, verify distributed data sharding correctness with `DistributedSampler`, and define the interface for the real VLA dataloader that will consume Phase 2.3 training data.

### Why this matters
Correct distributed data sharding is a hard correctness requirement — not an optimization. If two ranks see the same data, effective batch size is wrong and training is silently degraded. If sharding is correct but epoch shuffling is not (the current bug — no `sampler.set_epoch()` call), the model sees data in the same order every epoch, hurting generalization. The synthetic VLA dataset is a blocking prerequisite for Phase 4.2.0: the current `DummyDataset` produces `{input_ids, labels, attention_mask}` (3 keys, wrong shapes), but `VLAModel.forward()` expects 8 keys in the Phase 3.2.4 format. Without a compatible dataset, the training loop cannot execute a single step.

### Design

**`SyntheticVLADataset` (new file: `data/synthetic_vla.py`):**

Produces random tensors with correct shapes, dtypes, and realistic value ranges:

| Key | Shape | Dtype | Values |
|---|---|---|---|
| `input_ids` | `(S,)` | int64 | random in `[0, vocab_size)`, where `vocab_size=151936` (Qwen3.5) |
| `pixel_values` | `(4, 3, 320, 320)` | float32 | normalized `[-1, 1]` (4 frozen camera views) |
| `attention_mask` | `(S,)` | int64 | ones with random trailing zeros for padding variation |
| `robot_states` | `(n_segments, 52)` | float32 | random in `[-1, 1]`, `n_segments` ≥ 1 (one per action segment in sequence) |
| `action_chunks` | `(n_chunks, 16, 17)` | float32 | random in `[-1, 1]`; `n_chunks` random in `[1, max_chunks]`, each chunk is `(chunk_size=16, action_dim=17)` |
| `chunk_masks` | `(n_chunks, 16)` | float32 | per-step validity within each chunk (1=real, 0=padding); last chunk may have trailing zeros |
| `token_type_ids` | `(S,)` | int64 | Phase 4.1 `TokenType` enum: TEXT=0, IMAGE=1, STATE=2, ACTION=3; structurally consistent (see below) |
| `text_labels` | `(S,)` | int64 | copy of `input_ids` at TEXT positions, `-100` at all IMAGE/STATE/ACTION/padding positions |

Sequence length `S` drawn deterministically from a configurable range (default 512-2048) per sample using `idx` as seed. This exercises variable-length padding in the collator.

**Structural consistency constraints on `token_type_ids`:**

The synthetic batch must validate the real forward-path structure, not just tensor shapes. Each sample's `token_type_ids` is constructed to satisfy the following invariants (matching the Phase 3.2 interleaved sequence layout):

- The number of ACTION positions in `token_type_ids` equals exactly `n_chunks * 16` (chunk_size). This ensures `action_chunks` and the ACTION span in the sequence are dimensionally consistent.
- The number of STATE positions equals exactly `n_segments`. This ensures `robot_states` and the STATE span are consistent.
- `text_labels` has real token IDs only at TEXT positions; all IMAGE, STATE, and ACTION positions are set to `-100` (matching the `ignore_index` masking used by `VLATextLoss`).
- `attention_mask` is 1 at all non-padding positions and 0 at trailing padding. Padding positions have `token_type_ids` = 0 (TEXT) with `text_labels` = `-100`, so they are masked out by both `attention_mask` and `ignore_index`.
- The position ordering follows the Phase 3.2.0 layout: `[TEXT...] [IMAGE...] [TEXT...] [STATE] [ACTION × 16]...` segments interleaved. A simplified but structurally valid ordering is sufficient for distributed training validation.

**No separate `sequence_layout` metadata required:** Phase 3.2's forward pseudocode references a `sequence_layout` object for `assemble_sequence()`, but Phase 4.1 freezes the batch contract at exactly 8 keys. The finalized forward path derives all position routing from the padded tensors themselves: `text_labels` encodes text positions via `-100` masking (non-text → ignored by `VLATextLoss`), `chunk_masks` encodes valid action steps (passed directly to `VLAActionLoss`), and `token_type_ids` serves as a structural validator — not a runtime routing input to the loss modules. No 9th key or layout descriptor is added to the synthetic batch or collator.

**`VLACollator` (new file: `data/collator.py`):**

Pads variable-length VLA sequences into uniform batches:
- Pad `input_ids`, `attention_mask`, `token_type_ids`, `text_labels` to max sequence length in batch (right-pad: 0 for inputs, -100 for labels)
- Pad `action_chunks` to `(B, max_n_chunks, 16, 17)` and `chunk_masks` to `(B, max_n_chunks, 16)` — pad along the `n_chunks` dimension to max chunk count in batch
- Pad `robot_states` to `(B, max_n_segments, 52)` — pad along the `n_segments` dimension
- Stack fixed-shape tensors directly (`pixel_values`)
- ~100 lines of collation code

**`data/loader.py` modification:**
- Add optional `collate_fn` parameter to `create_dataloader()`
- Pass through to `DataLoader(collate_fn=...)`
- Default to `None` (PyTorch default collation) for backward compatibility with `DummyDataset`

**Multi-dataset interleaved sampling (interface spec for Phase 2.3):**

Phase 2.3 produces 4 training subsets with different mixing ratios. Define the config interface here for future implementation:

```yaml
# configs/data/vla.yaml (interface spec — consumed when Phase 2.3 data exists)
dataset:
  name: "vla"
  subsets:
    interleaved: {weight: 0.5, path: "${paths.data}/training_sequences/v2.3.0/interleaved/train.jsonl"}
    temporal:     {weight: 0.2, path: "${paths.data}/training_sequences/v2.3.0/temporal/train.jsonl"}
    spatial:      {weight: 0.2, path: "${paths.data}/training_sequences/v2.3.0/spatial/train.jsonl"}
    free_chat:    {weight: 0.1, path: "${paths.data}/training_sequences/v2.3.0/free_chat/train.jsonl"}
```

The actual `VLADataset` will produce the same batch dict as `SyntheticVLADataset` — same 8 keys, same dtypes. The distributed training pipeline sees no difference.

**Sharding correctness verification:**
- 2-process test via `torch.multiprocessing.spawn`
- Each rank collects its sample indices from the `DistributedSampler`
- Assert: no overlap between rank 0 and rank 1 indices
- Assert: different epochs produce different orderings (verifies `set_epoch()`)
- Assert: union of both ranks' indices covers the full dataset

### Execution checklist
- Implement `SyntheticVLADataset` in `data/synthetic_vla.py` matching all 8 Phase 3.2.4 batch keys
- Implement `VLACollator` in `data/collator.py` for variable-length padding
- Add `collate_fn` parameter to `create_dataloader()` in `data/loader.py`
- Add `data.dataset.name: "synthetic_vla"` routing in `Trainer._create_dataloaders()`
- Create `configs/data/synthetic_vla.yaml` with VLA-specific dataset parameters
- Create `configs/data/vla.yaml` (interface spec for Phase 2.3 — not yet functional)
- Write batch format test: verify all 8 keys present with correct shapes and dtypes
- Write collator test: variable-length input produces correctly padded rectangular batch
- Write sharding correctness test: 2-rank test verifying disjoint sample indices
- Write epoch shuffling test: different epochs produce different orderings
- Verify: training with `SyntheticVLADataset` on 2 GPUs produces correct effective batch size

### Milestone (minimum success criteria)
- `SyntheticVLADataset` produces batches with all 8 Phase 3.2.4 keys and correct shapes/dtypes. `VLACollator` pads variable-length sequences into rectangular batches with correct masks. Distributed training on 2 GPUs with `DistributedSampler` produces no sample overlap between ranks (verified by test). Epoch-aware shuffling produces different orderings across epochs.

---

## 4.2.2) Distributed Checkpoint & Resume

### What we will do
Replace the rank-0-only `torch.save` checkpointing with DeepSpeed collective checkpointing (all ranks participate), implement DDP fallback, persist W&B run ID and RNG states for seamless resume across training restarts, and verify checkpoint correctness via roundtrip tests.

### Why this matters
DeepSpeed ZeRO-1 shards optimizer state across ranks — every rank holds a different slice. The current `Trainer._save_checkpoint()` only saves on rank 0 (`torch.save`), which captures the model weights but misses the distributed optimizer shards. Loading this checkpoint into a DeepSpeed run corrupts training. Additionally, without RNG state preservation, data shuffling and stochastic operations (dropout, flow matching noise sampling) produce different sequences after resume, making debugging across restart boundaries non-deterministic. Checkpoint correctness is the highest-stakes correctness requirement in distributed training — a silent bug here wastes days of GPU-hours.

### Design

**Checkpoint contents by mode:**

| Component | DDP mode | DeepSpeed mode |
|---|---|---|
| Model weights | rank-0 `model.module.state_dict()` | Collective via `model_engine.save_checkpoint()` |
| Optimizer state | rank-0 `optimizer.state_dict()` | Collective — per-rank shards saved automatically |
| Scheduler state | rank-0 `scheduler.state_dict()` | Included in DeepSpeed engine state |
| Training state | rank-0: `{step, epoch, config}` | `client_state`: `{step, epoch, config, wandb_run_id, rng_states}` |
| RNG states | Python, NumPy, PyTorch, CUDA states | Same, via `client_state` |
| W&B run ID | In checkpoint dict | In `client_state` dict |

**DeepSpeed checkpoint format (managed by DeepSpeed):**
```
checkpoints/
  step_5000/
    global_step5000/
      mp_rank_00_model_states.pt
      zero_pp_rank_0_mp_rank_00_optim_states.pt
      zero_pp_rank_1_mp_rank_00_optim_states.pt
      ...
    latest
```

**DDP checkpoint format (unchanged from current):**
```
checkpoints/
  step_5000.pt
  step_10000.pt
  final.pt
```

**`_save_checkpoint()` redesign:**
- DeepSpeed mode (all ranks call): `model_engine.save_checkpoint(ckpt_dir, tag=f"step_{step}", client_state={step, epoch, config, wandb_run_id, rng_states})`
- DDP mode (rank-0 only): existing `torch.save()` logic, enhanced with `wandb_run_id` and `rng_states`
- Both modes: call `tracker.log_checkpoint()` for W&B artifact (rank-0 only, handled internally)

**`_load_checkpoint()` redesign:**
- DeepSpeed mode (all ranks call): `_, client_state = model_engine.load_checkpoint(ckpt_dir, tag)` — restores model, optimizer, scheduler automatically. Extract `global_step`, `epoch`, `wandb_run_id`, `rng_states` from `client_state`.
- DDP mode: existing `torch.load()` on rank-0, broadcast model state. Enhanced with RNG state restoration.
- W&B resume: extracted `wandb_run_id` passed to `create_tracker(resume_checkpoint={"wandb_run_id": ...})` during `setup()`.

**RNG state capture/restore helpers:**
- `_capture_rng_states()` -> dict: captures `random.getstate()`, `np.random.get_state()`, `torch.random.get_rng_state()`, `torch.cuda.get_rng_state_all()`
- `_restore_rng_states(states)`: restores all captured states

**`_find_latest_checkpoint()` helper:**
- DeepSpeed: scans for `step_*` directories, reads `latest` tag file
- DDP: scans for `step_*.pt` files, extracts step number from filename
- Returns path/tag of newest checkpoint, or None

**Checkpoint pruning (keep-last-N):**
- Config: `trainer.checkpoint.keep_last_n: 3`
- After each save, delete older checkpoints (directories for DeepSpeed, files for DDP)
- Rank-0 only for cleanup — barrier after to synchronize

**World-size constraint:**
DeepSpeed ZeRO-1 checkpoints are world-size-dependent (optimizer shard count matches rank count). Resume requires the same world-size. This is documented explicitly. If world-size change is needed, DeepSpeed provides `zero_to_fp32.py` to consolidate shards into a single-file checkpoint (model weights only, optimizer state discarded).

### Execution checklist
- Refactor `_save_checkpoint()` to dispatch between DeepSpeed (collective) and DDP (rank-0) modes
- Implement DeepSpeed save with `client_state` containing `global_step`, `epoch`, `config`, `wandb_run_id`, `rng_states`
- Implement DeepSpeed load with `client_state` extraction and RNG restoration
- Implement `_capture_rng_states()` and `_restore_rng_states()` helper methods
- Implement `_find_latest_checkpoint()` for both DeepSpeed and DDP layouts
- Implement checkpoint pruning (keep-last-N) for both modes
- Store `wandb_run_id` in checkpoint; pass to `create_tracker()` for experiment resume
- Write RNG state roundtrip test: capture -> restore -> verify same random sequence
- Write checkpoint `client_state` test: all required keys present after save
- Write `_find_latest_checkpoint()` test: correctly selects highest step from multiple checkpoints
- Write DDP backward compatibility test: old-format `step_*.pt` files still loadable
- Write DeepSpeed collective checkpoint test (2-GPU): save at step N -> load -> training loss at step N+1 is within tolerance
- Write resume continuity test: train 50 steps -> save -> restart -> verify loss trajectory has no discontinuity
- Write keep-last-N test: only N most recent checkpoints remain after save
- Document world-size constraint in CLAUDE.md

### Milestone (minimum success criteria)
- DeepSpeed checkpoint save/load works correctly on 2 GPUs: save at step 50, resume, loss trajectory continues without discontinuity. DDP checkpoint save/load still works for single-GPU mode — zero regressions. W&B run ID persists across restarts (verified by W&B showing continuous metrics). RNG states round-trip through save/load. Checkpoint pruning keeps only the last N checkpoints. World-size constraint is documented.

---

## 4.2.3) 8-GPU Production Scripts & Throughput Benchmarking

### What we will do
Create production-ready SLURM job templates that launch VLA training on 8 GPUs (4 nodes x 2 A100) using the project's actual `Trainer`, implement a throughput benchmarking script that measures samples/sec, peak GPU memory, and scaling efficiency, and establish performance baselines for the VLA model at distributed scale.

### Why this matters
The existing `04_smoke_8gpu_deepspeed.sh` runs a standalone dummy model that bypasses the project's Trainer, VLAModel, and data pipeline. It proves InfiniBand connectivity and DeepSpeed initialization work, but does not validate that the actual training pipeline runs correctly at scale. Throughput baselines are essential for estimating total training time (e.g., "10K episodes x 200 steps/episode at X samples/sec = Y GPU-hours"), tuning batch sizes and gradient accumulation, and detecting performance regressions in future phases.

### Design

**Production SLURM job template — `infra/gilbreth/job_templates/09_smoke_distributed_trainer.sh`:**

Follows the pattern of `04_smoke_8gpu_deepspeed.sh` but uses the project's Trainer:

```bash
srun --ntasks=$NNODES --ntasks-per-node=1 --export=ALL bash -c '
    export NODE_RANK=$SLURM_NODEID
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m train.trainer \
        cluster=gilbreth \
        trainer=distributed \
        data.dataset.name=synthetic_vla \
        trainer.training.max_steps=200 \
        trainer.checkpoint.save_every_n_steps=100
'
```

Key differences from template 04:
- Uses `python -m train.trainer` (the actual Trainer) instead of an inline dummy script
- Uses Hydra config composition (`cluster=gilbreth trainer=distributed`) instead of hardcoded values
- Tests checkpoint save at step 100, resume at step 100, train to step 200 (validates the full distributed pipeline)

**Throughput benchmarking script — `scripts/benchmark_throughput.py`:**

Sweep configurations at multiple world sizes:

| World Size | Batch/GPU | Grad Accum | Effective Batch | Purpose |
|---|---|---|---|---|
| 1 | 2 | 1 | 2 | Single-GPU baseline |
| 2 | 2 | 1 | 4 | Single-node (2 A100, NVLink) |
| 2 | 2 | 4 | 16 | Single-node + grad accum |
| 8 | 2 | 1 | 16 | Multi-node baseline |
| 8 | 2 | 4 | 64 | Multi-node + grad accum |
| 8 | 4 | 4 | 128 | Max effective batch (if memory allows) |

Each configuration runs 20 warmup steps (discard, JIT compilation and cache warming) + 100 measured steps.

**Metrics per configuration:**
- Samples/sec (global, across all ranks)
- Steps/sec (optimizer steps, accounting for accumulation)
- Step time (ms): mean, p50, p95
- Peak GPU memory per rank (GB): `torch.cuda.max_memory_allocated()`
- Scaling efficiency: `throughput_N / (throughput_1 * N) * 100%`

**Output format:**
```
VLA DISTRIBUTED TRAINING BENCHMARK
Model: VLAModel (Qwen3.5-4B + FlowMatching) | bf16 | ZeRO-1
Cluster: Gilbreth (A100 80GB, InfiniBand 100Gbps)

World | Batch/GPU | Accum | Eff.Batch | Samples/s | Peak VRAM | Scaling
------+-----------+-------+-----------+-----------+-----------+--------
  1   |     2     |   1   |     2     |   XX.X    |  XX.X GB  |  100%
  2   |     2     |   1   |     4     |   XX.X    |  XX.X GB  |   XX%
  8   |     2     |   1   |    16     |   XX.X    |  XX.X GB  |   XX%
  8   |     2     |   4   |    64     |   XX.X    |  XX.X GB  |   XX%
```

Artifacts saved to `logs/distributed_benchmark/`: `benchmark_report.json` (raw results), `benchmark_table.txt` (formatted table).

**Benchmark scope note:** These throughput numbers are infrastructure baselines for distributed mechanics (DeepSpeed communication overhead, gradient accumulation, bf16 compute) and model compute (forward/backward time for Qwen3.5-4B + FlowMatching). They are **not** final end-to-end throughput estimates for production training with the real Phase 2.3 data loader, which will introduce additional costs from HDF5 I/O, tokenization, image preprocessing, and multi-subset sampling. Production throughput characterization happens during Phase 4.3.

**SLURM job template for benchmarking — `infra/gilbreth/job_templates/10_throughput_benchmark.sh`:**
- Requests 4 nodes (maximum 8 GPUs) with 2-hour time limit
- Sequentially runs benchmark at 1, 2, and 8 GPU world sizes
- Artifacts to scratch: `${VLA_SCRATCH_ROOT}/logs/distributed_benchmark/`

**Effective batch size verification:**
- Count total samples processed across all ranks over 100 steps: `total = sum_over_ranks(local_batch_count * batch_size)`
- Assert: `total == 100 * batch_per_gpu * world_size * accumulation_steps`
- Log `effective_batch_size` metric to W&B for monitoring

**Standalone validation script — `scripts/validate_distributed.py`:**

8-check validation following the established `scripts/validate_*.py` pattern:

| Check | Input | Verifies |
|---|---|---|
| 1. DeepSpeed import | Python import | `import deepspeed` succeeds; version >= 0.10.0 |
| 2. Config composition | Hydra compose | `trainer=distributed cluster=gilbreth data=synthetic_vla` parses without error |
| 3. DeepSpeed config build | `_build_deepspeed_config()` | Valid JSON with correct optimizer params, batch sizes, scheduler |
| 4. Synthetic batch format | `SyntheticVLADataset` + `VLACollator` | All 8 keys, correct shapes and dtypes |
| 5. Single-GPU training | 20 steps, DeepSpeed on 1 GPU | Loss decreases, grad norm finite, LR follows warmup |
| 6. Checkpoint roundtrip | Save step 10, load, 10 more steps | Training continues, loss trajectory consistent |
| 7. Tracker integration | `ExperimentTracker` mock | Metrics logged at correct intervals, correct keys |
| 8. Memory within budget | 20 steps with target batch size | Peak VRAM < 70 GB per A100 (10 GB headroom) |

Artifacts: `logs/distributed_validation/validation_report.json` (all check results), `logs/distributed_validation/memory_profile.json` (peak VRAM measurements).

### Execution checklist
- Create `infra/gilbreth/job_templates/09_smoke_distributed_trainer.sh` — 8-GPU smoke test using actual Trainer with Hydra config
- Create `infra/gilbreth/job_templates/10_throughput_benchmark.sh` — scaling efficiency measurement at 1/2/8 GPUs
- Implement `scripts/benchmark_throughput.py` with sweep, warmup, metrics collection, and report generation
- Implement `scripts/validate_distributed.py` with 8 checks and artifact output
- Run 200-step smoke test on 8 GPUs: verify no errors, loss decreases, all ranks complete, checkpoint save/load works
- Run throughput benchmark: collect metrics across 1/2/8 GPU configurations
- Verify effective batch size matches `batch_per_gpu * world_size * accumulation_steps` (verified by sample counting)
- Verify scaling efficiency >= 70% at 8 GPUs (reasonable threshold for 4-node InfiniBand with 4B model)
- Verify peak GPU memory fits within 80 GB for target batch size (batch_size=4, accumulation=4)
- Save benchmark artifacts to `logs/distributed_benchmark/`
- Save validation artifacts to `logs/distributed_validation/`
- Add benchmark and validation commands to CLAUDE.md
- Update CLAUDE.md with distributed training documentation (trainer modes, checkpoint format, SLURM templates, config composition)

### Milestone (minimum success criteria)
- 8-GPU training completes 200 steps without error using the project's Trainer + DeepSpeed + `SyntheticVLADataset`. Checkpoint saved at step 100 and successfully resumed. Throughput benchmark produces measured samples/sec and peak VRAM for 1/2/8 GPU configurations. Scaling efficiency >= 70% at 8 GPUs. Effective batch size verified correct by sample counting. Validation script exits with 8/8 checks passed. Benchmark artifacts and SLURM templates are committed.

---

# Downstream Contract with Phase 4.3

Phase 4.3 (Production Training Runs) will consume Phase 4.2's distributed training setup as follows:

| Phase 4.3 Need | Phase 4.2 Provider | Interface |
|---|---|---|
| 8-GPU training launcher | `09_smoke_distributed_trainer.sh` SLURM template | `sbatch` with Hydra config overrides |
| Distributed trainer | `Trainer.setup()` + `Trainer.train()` | DeepSpeed ZeRO-1 via `cluster=gilbreth` |
| Checkpoint resume | `_load_checkpoint()` + DeepSpeed collective load | `trainer.checkpoint.resume_from: "latest"` |
| Real data integration | `configs/data/vla.yaml` interface spec + `VLACollator` | Swap `data.dataset.name` from `synthetic_vla` to `vla` |
| Performance baseline | `logs/distributed_benchmark/benchmark_report.json` | Expected throughput range for regression detection |
| Effective batch size | Verified formula: `batch_per_gpu * world * accum` | Config overrides in `trainer=distributed` |
| Loss logging | `ExperimentTracker.log_training_step()` | `total_loss`, `text_loss`, `action_loss`, per-component metrics to W&B |

**Key design constraint:** The distributed training setup is data-format-agnostic. When Phase 2.3 data becomes available, the only change is implementing a `VLADataset` that produces the same 8-key batch dict as `SyntheticVLADataset` and switching `data.dataset.name` in the Hydra config. No changes to the Trainer, DeepSpeed integration, checkpointing, or SLURM templates are needed.

---

# Startup-Grade Outputs (deliverables by end of 4.2)
- **DeepSpeed-integrated Trainer** — upgraded `train/trainer.py` with ZeRO-1, bf16, gradient accumulation, warmup+cosine LR scheduler, W&B `ExperimentTracker` integration, epoch-aware distributed sampling, and VLAModel forward/backward interface
- **Synthetic VLA dataset** — `SyntheticVLADataset` producing all 8 Phase 3.2.4 batch keys with variable-length sequences; `VLACollator` for padding
- **Distributed checkpointing** — DeepSpeed collective save/load + DDP fallback, RNG state preservation, W&B run resume, checkpoint pruning (keep-last-N)
- **8-GPU SLURM templates** — production `09_smoke_distributed_trainer.sh` (200-step smoke test with checkpoint resume) and benchmark `10_throughput_benchmark.sh` (scaling efficiency measurement)
- **Throughput benchmark** — measured samples/sec, peak VRAM, scaling efficiency across 1/2/8 GPU configurations on Gilbreth
- **Data pipeline interface** — `configs/data/vla.yaml` spec for Phase 2.3 training data integration; `configs/data/synthetic_vla.yaml` for distributed validation
- **Validation** — `scripts/validate_distributed.py` (8 checks) and `scripts/benchmark_throughput.py`
- **Test suite** — ~20 tests in `tests/test_distributed_trainer.py` covering config build, batch format, collation, sharding correctness, checkpoint roundtrip, resume continuity, RNG preservation

---

# Files Inventory

### New files (11)

| File | Purpose | Lines (est.) |
|---|---|---|
| `configs/cluster/local_deepspeed.yaml` | Local multi-GPU cluster config with DeepSpeed enabled — portable (no SLURM, no InfiniBand, loopback network) | ~40 |
| `data/synthetic_vla.py` | `SyntheticVLADataset` — Phase 3.2.4 batch format with variable-length sequences | ~180 |
| `data/collator.py` | `VLACollator` — pads variable-length VLA sequences into rectangular batches | ~80 |
| `configs/data/synthetic_vla.yaml` | Synthetic VLA dataset config (vocab_size, seq_length_range, chunk_size, etc.) | ~20 |
| `configs/data/vla.yaml` | Real VLA dataset interface spec for Phase 2.3 (subset weights, paths) | ~25 |
| `configs/trainer/distributed.yaml` | 8-GPU production trainer overrides (batch_size=4, accum=4, checkpoint every 1000) | ~25 |
| `configs/deepspeed/zero1_vla.json` | DeepSpeed config with warmup scheduler section and Hydra overlay pattern | ~50 |
| `tests/test_distributed_trainer.py` | Unit + integration tests (~20 tests; markers: gpu, slow) | ~400 |
| `scripts/benchmark_throughput.py` | Throughput profiling across GPU configurations | ~120 |
| `scripts/validate_distributed.py` | 8-check standalone validation script | ~200 |
| `infra/gilbreth/job_templates/09_smoke_distributed_trainer.sh` | 8-GPU Trainer smoke test (200 steps + checkpoint resume) | ~80 |

### Additional SLURM template (1)

| File | Purpose | Lines (est.) |
|---|---|---|
| `infra/gilbreth/job_templates/10_throughput_benchmark.sh` | Scaling efficiency benchmark at 1/2/8 GPUs | ~100 |

### Modified files (5)

| File | Change |
|---|---|
| `train/trainer.py` | DeepSpeed init, bf16, grad accum, warmup scheduler, `ExperimentTracker` integration, collective checkpointing, RNG state persistence, `sampler.set_epoch()`, VLAModel forward interface |
| `data/loader.py` | Add optional `collate_fn` parameter to `create_dataloader()` |
| `data/__init__.py` | Export `SyntheticVLADataset`, `VLACollator` |
| `configs/trainer/default.yaml` | Verify `gradient.accumulation_steps` and `training.precision` fields are consumed |
| `CLAUDE.md` | Add distributed training docs: trainer modes, DeepSpeed config, checkpoint format, world-size constraint, SLURM templates, benchmark/validation commands |

---

# Phase 4.2 Definition of Done

Phase 4.2 is complete when:
- `Trainer.setup()` correctly initializes DeepSpeed ZeRO-1 when `cluster.distributed.deepspeed.enabled=true`, with bf16 mixed precision and gradient accumulation. DDP path remains functional for `cluster=local`.
- Training loop integrates W&B `ExperimentTracker` with `total_loss`, `text_loss`, `action_loss`, learning rate, gradient norm, and throughput logging (rank-0 only).
- Warmup + cosine LR scheduler is active — warmup ramp visible in logged LR curve.
- `SyntheticVLADataset` produces batches with all 8 Phase 3.2.4 keys and correct shapes/dtypes. `VLACollator` pads variable-length sequences correctly.
- `DistributedSampler` with `set_epoch()` produces non-overlapping sample indices across ranks (verified by test).
- DeepSpeed checkpoint save/load roundtrip works on 2 GPUs: save at step N, resume at step N+1, loss trajectory is consistent (no discontinuity).
- DDP checkpoint save/load still works for local single-GPU mode — zero regressions.
- W&B run ID is persisted in checkpoint and restoring it resumes the same experiment run.
- RNG states round-trip through save/load (verified by test).
- Checkpoint pruning keeps only the last N checkpoints.
- 8-GPU training completes 200 steps without error on Gilbreth (4 nodes x 2 A100), with checkpoint save at step 100 and successful resume.
- Throughput benchmark produces measured samples/sec and peak VRAM for 1/2/8 GPU configurations.
- Scaling efficiency >= 70% at 8 GPUs.
- Effective batch size matches `batch_per_gpu * world_size * accumulation_steps` (verified by sample counting).
- All existing tests (`test_models.py`, etc.) pass unchanged — zero regressions.
- ~20 new tests pass in `tests/test_distributed_trainer.py`.
- Validation script (`scripts/validate_distributed.py`) exits with 8/8 checks passed.
- `CLAUDE.md` is updated with distributed training documentation.

---

# Verification Plan

### Local (lab PC, single GPU or CPU)
```bash
# Existing smoke test still works (zero regressions)
python -m train.trainer trainer=debug cluster=local

# Synthetic VLA dataset test
python -m train.trainer trainer=debug cluster=local data.dataset.name=synthetic_vla

# CPU-safe unit tests
pytest tests/test_distributed_trainer.py -v -m "not gpu and not slow"

# Backward compatibility
pytest tests/test_models.py -v
```

### Local (lab PC, 2 GPUs — if available)
```bash
# 2-GPU DDP test
torchrun --nproc_per_node=2 -m train.trainer trainer=debug cluster=local data.dataset.name=synthetic_vla

# 2-GPU DeepSpeed test (uses local_deepspeed config — portable, no SLURM/InfiniBand deps)
torchrun --nproc_per_node=2 -m train.trainer trainer=debug cluster=local_deepspeed data.dataset.name=synthetic_vla

# Distributed tests (requires 2 GPUs)
pytest tests/test_distributed_trainer.py -v
```

### Cluster (Gilbreth)
```bash
# 8-GPU smoke test with checkpoint resume
sbatch infra/gilbreth/job_templates/09_smoke_distributed_trainer.sh

# Throughput benchmark (1/2/8 GPUs)
sbatch infra/gilbreth/job_templates/10_throughput_benchmark.sh

# Standalone validation (8 checks)
python scripts/validate_distributed.py

# Standalone benchmark
python scripts/benchmark_throughput.py --output-dir logs/distributed_benchmark/
```

### CI (no changes needed)
Distributed tests marked `gpu`/`slow` are auto-skipped in CI. Existing CI installs `.[ci]` which excludes DeepSpeed. CPU-safe tests in `test_distributed_trainer.py` (config build, batch format, collator, RNG roundtrip) run in CI. Verify with:
```bash
pytest tests/ -m "not slow and not gpu and not mujoco and not vlm" -v
```
