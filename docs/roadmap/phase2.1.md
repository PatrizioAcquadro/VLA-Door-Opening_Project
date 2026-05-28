# Phase 2.1 — Robot Trajectory Data Generation (5 days)
**Goal:** Generate a large-scale dataset of **physically valid robot demonstrations** for VLA training — where the Alex robot physically grasps, transports, and places LEGO bricks through its actuated joints, producing synchronized `(observation, 17-D action)` pairs at every 20 Hz control timestep.

**Fixed upstream decisions (from 1.2):**
- **Sim engine:** MuJoCo (MJCF-first, headless via EGL)
- **Robot:** IHMC Alex upper-body fixed-base, **17-D action space** (Δq arms + gripper), **52-D state vector**
- **Views:** 4 frozen cameras (overhead, wrist-L, wrist-R, third-person), 320×320, 20 Hz
- **Control:** SimRunner at 20 Hz (25 substeps × 0.002 s), `AlexActionSpace` normalization
- **LEGO:** Soft press-fit contacts, procedural bricks {2×2, 2×4, 2×6}, 8×8 baseplate, workspace scene
- **Episodes:** `EpisodeManager` with deterministic seeded reset, 3 curriculum levels
- **Tasks:** `generate_assembly_goal()` → `PlacementTarget` sequences, `check_placement()` for success

**Key Phase 2.1 stance:**
- **Pure Jacobian control** for all demonstrations in the base dataset. No `xfrc_applied` force-assist. If insertion accuracy is insufficient after tuning, a separately labeled assisted subset may be generated outside the clean dataset.
- **Progressive generation** — 5K single-brick episodes first, validate end-to-end, then 5K mixed-complexity.
- **All modalities** — RGB + Depth + Segmentation stored per frame (~550 GB total for 10K episodes).
- **Single-arm pick-and-place** — closest arm picks (Y>0 → left, Y<0 → right), other arm at rest. Bimanual is an explicit next-step extension.

**Critical gap this phase closes:**
The existing `ScriptedAssembler` (Phase 1.2.6) bypasses the robot entirely — it teleports bricks via `qpos` writes and applies external forces via `xfrc_applied`. The robot arms never move. For VLA training, we need a **demonstration controller** that produces 17-D actions through the real control pipeline (`SimRunner.step()`), so that every recorded `(state, action, image)` tuple is something a learned policy could reproduce.

---

## 2.1.1) Task-Space Demonstration Controller
### What we will do
Implement a **Jacobian-based resolved-rate controller** that produces 17-D actions from end-effector position targets. This is the core new component — without it, we cannot generate physically valid robot demonstrations.

### Why this matters
The entire data generation pipeline depends on this controller. It must produce smooth, physically realistic joint-space actions that move the robot's end-effectors to desired poses while respecting joint limits and velocity bounds.

### Design
- Use `mujoco.mj_jacSite()` on `left_tool_frame` / `right_tool_frame` to compute the 3×N position Jacobian
- Extract columns for the active arm's joints only (8 joints: spine + 7 arm)
- Resolved-rate control law: `dq = J^T (J J^T + λI)^{-1} · Kp · (x_target - x_current)` (damped least squares, λ=0.01)
- Clip `dq` to `delta_q_max` bounds (from `AlexActionSpace`), normalize via `AlexActionSpace.normalize(dq_15, gripper_cmd)` → 17-D action
- Execute via `SimRunner.step(action)` → `RobotState`
- Fine-approach zone: halve gain when position error < 5 mm to prevent overshoot
- Active arm selection: Y>0 → left arm, Y<0 → right arm (Alex Y-left convention)
- Idle arm: zero deltas in the 17-D action vector

### Execution checklist
- Implement `DemoController(model, data, action_space)` with `step_toward(target_pos, gripper_cmd)` → 17-D action
- Implement `execute_waypoints(waypoints)` → list of `(action, state)` tuples at each 20 Hz step
- Damped pseudoinverse with configurable λ (Tikhonov regularization to avoid singularity explosion)
- Fine-approach gain reduction within configurable radius
- Per-waypoint convergence check (position error < tolerance) and timeout (max steps)
- Unit tests: reaching to fixed targets, normalize roundtrip, convergence within N steps

### Milestone (minimum success criteria)
- Controller moves EE to 10 random workspace positions with < 2 mm final error and no joint-limit violations.

---

## 2.1.2) Grasp Planning & Pick-and-Place Waypoints
### What we will do
Implement a **waypoint generator** that converts a `PlacementTarget` (from `sim/lego/task.py`) into a concrete sequence of waypoints for the demo controller. This defines the physical manipulation strategy for single-arm LEGO pick-and-place.

### Why this matters
This bridges the task system (`generate_assembly_goal()`) and the controller. Without a well-defined waypoint sequence, the robot cannot reliably grasp, transport, and place bricks.

### Design — 8-phase waypoint sequence for single brick placement
1. **Pre-grasp**: EE to `(bx, by, bz + 0.05)`, gripper open (1.0)
2. **Approach**: EE to `(bx, by, bz + grasp_z_offset)`, gripper open (1.0), slower gain
3. **Close gripper**: Hold position, ramp gripper 1.0 → 0.0 over ~10 steps (0.5 s)
4. **Lift**: EE to `(bx, by, bz + 0.08)`, gripper closed (0.0)
5. **Transport**: EE to `(tx, ty, tz + 0.05)`, gripper closed (0.0)
6. **Lower/Insert**: EE to `(tx, ty, tz + grasp_z_offset)`, gripper closed (0.0), slow gain
7. **Release**: Hold position, ramp gripper 0.0 → 1.0 over ~10 steps
8. **Retract**: EE to `(tx, ty, tz + 0.05)`, gripper open (1.0)

- Brick position read from `data.xpos[body_id]` at execution time (handles settle drift)
- Target position from `compute_target_position()` in `sim/lego/task.py`
- `grasp_z_offset` = empirical offset from tool_frame site to finger contact point (~2–3 cm, tunable)
- Top-down approach perpendicular to table surface (identity orientation)

### Execution checklist
- `Waypoint` frozen dataclass: position, gripper_cmd, tolerance, max_steps, gain_scale
- `generate_pick_place_waypoints()` produces 8-phase sequence from `PlacementTarget`
- Grasp height calibration: determine `grasp_z_offset` by measuring tool_frame ↔ finger contact distance
- Integration test: full pick-and-place of one 2×2 brick from table to baseplate
- Success rate test: 50 random single-brick placements, target ≥ 80% success
- Video validation: render EE trajectory for visual inspection

### Milestone (minimum success criteria)
- Single-arm pick-and-place of a 2×2 brick succeeds ≥ 80% over 50 random episodes (seed-deterministic).

---

## 2.1.3) Episode Data Recording & HDF5 Storage
### What we will do
Define the **HDF5 per-episode storage format** and implement the recording pipeline that captures synchronized (state, action, images, metadata) at 20 Hz.

### Why this matters
The training pipeline needs efficiently loadable trajectory data with all modalities temporally aligned. The format must support random access (for batching), compression (for storage), and metadata (for reproducibility).

### HDF5 schema (per episode file)
```
episode_{seed:06d}.hdf5
├── metadata/                    (group attributes)
│   seed, level, brick_types, n_timesteps, success,
│   assembly_result (JSON), goal (JSON), spawn_poses (JSON),
│   control_hz, version ("2.1.1"), timestamp_created (ISO 8601)
├── state/
│   raw: float64 (T, 52)        # raw 52-D state vectors
│   normalized: float32 (T, 52) # normalized for training
│   timestamps: float64 (T,)    # sim timestamps
├── action/
│   normalized: float32 (T, 17) # normalized 17-D actions
├── images/
│   overhead/
│     rgb: uint8 (T, 320, 320, 3)
│     depth: float32 (T, 320, 320)
│     segmentation: int32 (T, 320, 320, 2)
│   left_wrist_cam/  ...
│   right_wrist_cam/ ...
│   third_person/    ...
├── camera_metadata/             (group attributes)
│   intrinsics: JSON per camera (constant, stored once)
│   extrinsics_{cam}: float64 (T, 12)   # [pos(3) + mat(9)] per step
└── labels/
    phase: uint8 (T,)           # waypoint phase index (0–7)
    grasp_state: float32 (T,)   # gripper command per step
    episode_outcome: str        # "success" | failure type
    perturbation_type: str      # "none" | perturbation type
    failure_timestep: int       # -1 if no failure
    has_recovery: bool          # true if episode contains a retry sequence
    recovery_start_timestep: int # step where recovery begins (-1 if none)
```

### Storage layout
```
data/demonstrations/v2.1.1/
├── episodes/
│   ├── episode_000000.hdf5
│   └── ...
├── train_manifest.json
├── val_manifest.json
├── test_manifest.json
└── dataset_stats.json          # per-feature mean/std for normalization
```

### Storage estimates
- RGB images: JPEG quality 90 (~30 KB per 320×320 frame)
- Depth: float32 with gzip compression
- Segmentation: int32 with gzip compression
- Per episode (200 steps, all modalities): ~55 MB
- 10K episodes: ~550 GB

### Execution checklist
- `EpisodeRecorder.record_step(state, action, frame, phase)` buffers in memory
- `EpisodeRecorder.save(path, metadata)` flushes to HDF5 with compression
- JPEG compression for RGB, gzip for depth/segmentation
- Read-back validation: write → read → assert shapes and values match
- `DemoDataset.__getitem__(idx)` PyTorch Dataset returns `{images, state, action, metadata}` tensors
- Schema validation test: verify all groups/datasets exist with correct dtypes
- Add `h5py>=3.8.0` to `pyproject.toml` (`sim` and `dev` dependency groups)

### Milestone (minimum success criteria)
- Write and read-back a 200-step episode with all modalities. File size ~50–60 MB. `DemoDataset` loads it correctly.

---

## 2.1.4) Single-Episode Demo Pipeline (End-to-End)
### What we will do
Build the **single-episode orchestrator** that ties together EpisodeManager → goal generation → waypoint planning → demo controller → camera capture → HDF5 recording into one callable function.

### Why this matters
This is the first end-to-end integration of all components. It must work reliably before parallelizing. Every bug found here saves hours of wasted parallel generation time.

### Pipeline
```python
def generate_episode(seed, level, output_dir, config) -> EpisodeResult:
    # 1. EpisodeManager.reset(seed, level)
    # 2. generate_assembly_goal(info, baseplate, ...)
    # 3. For each PlacementTarget in goal:
    #    a. generate_pick_place_waypoints(target)
    #    b. DemoController.execute_waypoints(waypoints)
    #    c. At each step: capture 4-view frame, record (state, action, frame)
    # 4. evaluate_assembly() for success metrics
    # 5. EpisodeRecorder.save() to HDF5
    # 6. Return EpisodeResult (metadata + success + timing)
```

- `MultiViewRenderer` created once per episode, shared across all timesteps
- Controller computes action → `SimRunner.step()` executes → capture frame → record (strict per-timestep ordering)
- Failed grasps/insertions: episode still recorded with failure labels (useful for failure dataset)
- Success evaluation via `check_placement()` from `sim/lego/task.py`

### Execution checklist
- `generate_episode()` produces valid HDF5 from a single seed
- Determinism: same seed → identical state/action sequences
- Timing: single episode < 60 seconds on one CPU core
- Video export: render saved HDF5 images back to MP4 for visual inspection
- Success rate: ≥ 80% on 50 random Level 1 episodes
- Failure episodes still recorded with correct labels
- Standalone debug script: `scripts/generate_single_demo.py`

### Milestone (minimum success criteria)
- 50 Level 1 episodes generated, ≥ 80% success rate, all HDF5 files valid, deterministic replay confirmed.

---

## 2.1.5) Parallel Data Collection & 10K Generation
### What we will do
Build a **multi-process data collection pipeline** that generates 10K episodes with fault tolerance, resumption, and progress tracking.

### Why this matters
Single-process generation would take ~170 hours. Parallel collection is required for practical generation times (~10–20 hours with 8–16 workers).

### Design
- Python `multiprocessing` — each worker owns its own `EpisodeManager` + `MultiViewRenderer` + `SimRunner` (MuJoCo is not thread-safe)
- Seed management: worker `w` processes seeds `[w * block_size, (w+1) * block_size)` — reproducible regardless of worker count
- Fault tolerance: per-episode try/except, failures logged to `failures.jsonl` with seed + traceback; 5-minute watchdog timeout per episode
- Resumption: on restart, check which HDF5 files exist, skip completed seeds
- Progress: tqdm in master process via shared counter (`multiprocessing.Value`)
- Default workers: `min(N_CPU - 1, 16)`, ~1 GB RAM per worker

### Progressive generation strategy
1. **Wave 1**: Generate 5K single-brick episodes (Level 1/2, seeds 0–4999)
2. **Validate**: Run QC checks, measure success rate, inspect sample videos
3. **Wave 2**: Generate 5K mixed-complexity episodes (Level 1–3, seeds 5000–9999)
4. **Merge**: Combine manifests, recompute dataset statistics

### Estimated wall time
| Workers | 5K episodes | 10K episodes |
|---------|-------------|--------------|
| 8       | ~10 hours   | ~20 hours    |
| 16      | ~5 hours    | ~10 hours    |

### Execution checklist
- Small-scale test: 100 episodes with 4 workers, all HDF5 files valid
- Determinism: same seeds produce identical outputs regardless of worker count
- Resumption: kill at 50%, restart, only remaining episodes generated
- Manifests: train/val/test splits (80/10/10) are disjoint and cover all episodes
- Dataset stats: per-feature mean/std computed correctly → `dataset_stats.json`
- Wave 1: generate 5K single-brick, success rate ≥ 80%
- Wave 2: generate 5K mixed, validate and merge
- Add generation commands to `CLAUDE.md`

### Milestone (minimum success criteria)
- 10K episodes generated, manifests created, `dataset_stats.json` computed, all files validated.

---

## 2.1.6) Failure Cases, Recovery & Quality Control
### What we will do
(a) Generate **intentional failure demonstrations** with labeled perturbations and recovery attempts.
(b) Run a **quality control filtering pipeline** on the full dataset.
(c) Produce **dataset statistics** and a summary report.

### Why this matters
Failure/recovery demonstrations improve policy robustness. QC filtering ensures dataset integrity — one corrupted episode in training can waste GPU-hours. Statistics provide confidence in the dataset's coverage and quality.

### Failure taxonomy
| Type | Fraction | Perturbation |
|------|----------|-------------|
| Misalignment | 30% | 2–5 mm lateral offset on insertion waypoint |
| Premature release | 20% | Open gripper 3–5 cm before target height |
| Grasp miss | 15% | 5–10 mm offset on grasp approach |
| Collision/knock | 15% | Transport waypoint near existing structure |
| Drop during transport | 10% | Brief gripper open (1–2 steps at cmd=0.5) |
| Incomplete insertion | 10% | Retract before engagement |

- Perturbations modify the waypoint sequence (not controller gains) via `PerturbationConfig` dataclass
- Recovery: after detected failure, generate re-approach + retry. Full trajectory recorded including failure + recovery, labeled `"has_recovery": true`
- Target composition: 7K success + 2K failure + 1K recovery = 10K total

### Quality control pipeline
- **Physics**: reject NaN in state/action, penetration > 5 mm, energy > 1000 J
- **Actions**: reject if any action exceeds [-1.05, 1.05] (5% tolerance for numerical noise)
- **Images**: reject if mean pixel < 10 (black) or > 245 (white)
- **State continuity**: reject if consecutive state jumps exceed bounds (no teleportation artifacts)
- **Episode length**: reject if < 50 or > 1000 steps
- **Duplicates**: verify no two episodes share the same seed

### Dataset statistics report
- Episode length distribution (histogram)
- Per-joint action distribution (mean, std, min, max)
- Success rate by brick type and curriculum level
- EE position coverage heatmap (XY workspace)
- Failure type distribution
- Total disk usage

### Execution checklist
- Each perturbation type produces a detectable failure
- Recovery episodes contain full fail + retry sequence
- QC filter catches injected bad episodes (NaN, out-of-range)
- Statistics report covers all metrics
- Final dataset manifest reflects filtered, labeled episodes

### Milestone (minimum success criteria)
- Cleaned 10K-episode dataset with success/failure/recovery labels, QC report, and statistics.

---

# Startup-Grade Outputs (deliverables by end of 2.1)
- **Jacobian-based demonstration controller** — produces policy-compatible 17-D actions through real physics
- **10K trajectory dataset** — HDF5 per-episode with RGB+Depth+Segmentation, state, action, metadata
- **Parallel generation pipeline** — multi-process, fault-tolerant, resumable, ~10 hours on 16 cores
- **Failure/recovery demonstrations** — labeled perturbation types, recovery trajectories
- **Quality control pipeline** — automated validation, filtering, and dataset statistics
- **PyTorch Dataset** — `DemoDataset` for direct training integration

---

# Phase 2.1 Definition of Done
Phase 2.1 is complete when:
- The Jacobian demo controller reliably picks and places LEGO bricks (≥ 80% success rate on single-brick).
- 10K episodes are generated, validated, and stored in HDF5 with all modalities.
- Train/val/test manifests exist with 80/10/10 split and dataset statistics.
- Failure and recovery episodes are included with correct labels.
- QC filtering has been run and a clean dataset manifest exists.
- `DemoDataset` loads episodes correctly for training.
