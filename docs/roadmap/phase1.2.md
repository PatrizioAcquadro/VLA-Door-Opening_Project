# Phase 1.2 — LEGO Environment Creation (5 days)
**Goal:** Build a MuJoCo LEGO assembly environment that is **real-world-relevant for IHMC Alex**, supports **contact-rich press-fit behavior (soft-realism)**, is **stable enough to generate large-scale datasets**, and is **engineered to a startup-grade standard** (reproducible, measurable, scalable, debuggable).

**Fixed upstream decisions (from 1.1):**
- **Sim engine:** MuJoCo (MJCF-first, code-first workflow; viewer for debugging)
- **Robot:** Existing **IHMC Alex** model (no proxy), **upper-body fixed-base**
- **Action:** 16-D **Δq** (bimanual 14) + **gripper** (2), policy rate ~20 Hz
- **State:** core (q, q̇, gripper) + recommended (EE pose + EE velocity)
- **Views:** 4 cameras from day 1 (overhead, wrist-L, wrist-R, third-person)

**Key Phase 1.2 stance (your decision 0):**
- We target **Real press-fit** as the long-term objective, but implement a **Soft Press-Fit** version now:
  - **Geometrically accurate connectors** (studs/tubes) and **physically plausible contacts**
  - **Controlled compliance + tolerances** to avoid solver instability
  - **No “magic snap constraints” as the primary mechanism** (we may keep a *diagnostic fallback* path only for debugging, not for dataset ground-truth)

---

## 1.2.0) Soft Real Press-Fit Definition (the “definitive” version for this project)
### What we will do
Implement a **Soft Real Press-Fit** LEGO connection model: studs and tubes are represented with accurate geometry/topology, but the physics is made robust via **carefully designed compliance, tolerances, and contact conditioning** rather than hard interference.

### Why this matters
True LEGO assembly is defined by **insertion forces**, **alignment**, and **mechanical retention**. A soft-real press-fit preserves these phenomena while keeping the simulator stable enough to generate high-quality datasets and support future sim-to-real transfer.

### Execution checklist
- **Define press-fit realism targets**
  - Specify target behaviors:
    - “Stud enters tube only if aligned”
    - “Insertion requires force and produces resistance”
    - “Connection remains stable under moderate perturbations”
  - Define acceptable simplifications:
    - Allow small compliance (material/connector elasticity approximation)
    - Allow tolerance bands for manufacturing variation
- **Design contact conditioning strategy**
  - Use soft-contact parameters (solver softness/impedance) to prevent numerical explosions.
  - Avoid hard mesh-mesh interference; model *effective* interference via compliant contacts.
- **Define tolerances and clearances (explicit)**
  - Specify insertion clearance (microscopic) and a “capture envelope” for alignment.
  - Define retention behavior (resistance to pull-off) via friction + geometry + compliance.
- **Define measurable success criteria**
  - Minimum insertion success rate under controlled approach trajectories
  - Maximum allowed jitter/penetration artifacts
  - Retention test under lateral/vertical disturbance

### Milestone (minimum success criteria)
- A standardized press-fit specification exists and is applied consistently to brick-brick connections:
  - Connect only when aligned within tolerance
  - Requires non-trivial insertion effort (observable resistance)
  - Stays connected under moderate disturbances without solver instability

---

## 1.2.1) Procedural LEGO Brick Models (2×2, 2×4, 2×6) — A+C Fusion
### What we will do
Create **procedural (parametric) LEGO bricks** that generate:
- **Accurate visual meshes with studs**
- **Connector metadata** (stud/tube locations, orientations, contact surfaces)
- **Collision geometry** intentionally chosen for stable soft press-fit contacts

### Why this matters
Procedural assets provide:
- Scale and extensibility (add 1×N, plates, special parts later)
- Perfect consistency for dataset generation (same canonical geometry and connector map)
- A strong “startup signal”: engineered assets, not ad-hoc meshes

### Execution checklist (define the “work” clearly)
- **Parameterization spec (brick family)**
  - Define canonical LEGO units:
    - stud pitch (grid spacing)
    - brick height (in LEGO units)
    - stud dimensions and tube socket dimensions (parametric)
  - Define brick type parameters:
    - `(studs_x, studs_y, height_type)` for 2×2 / 2×4 / 2×6
- **Procedural geometry generation**
  - Generate:
    - outer shell
    - studs (top features)
    - tubes/sockets (bottom features)
  - Ensure consistent coordinate frames and origins (center-of-mass and reference corner conventions).
- **Connector metadata generation (critical)**
  - For each stud:
    - position, axis direction, radius/shape descriptor
  - For each tube/socket:
    - position, axis direction, internal radius/shape descriptor
  - Define a connector indexing scheme (stable IDs).
- **Collision strategy for soft press-fit (explicit)**
  - Use collision primitives (cylinders/capsules) for studs and socket walls where possible.
  - Keep a separate high-detail **visual mesh** for rendering.
  - Ensure collision does not create extremely thin triangles/surfaces (instability risk).
- **Asset export & versioning**
  - Export MJCF-compatible geometry assets or embed generated geoms directly.
  - Pin generation parameters and versions (so any dataset can be reproduced).

### Milestone (minimum success criteria)
- Procedural generator produces 2×2, 2×4, 2×6 bricks with:
  - visually accurate studs,
  - stable collision geoms compatible with soft press-fit,
  - complete connector metadata (stud/tube maps) with stable IDs.

---

## 1.2.2) Stud/Tube Contact Physics — Soft Real Press-Fit Implementation
### What we will do
Implement and tune contact physics to achieve press-fit behavior using:
- physically plausible contact interactions,
- compliance and tolerances,
- stable solver conditioning for repeated insertion tasks.

### Why this matters
This is the core of LEGO assembly realism. If it fails, the environment either becomes unstable (no dataset) or unrealistic (no transfer).

### Execution checklist
- **Contact material model specification**
  - Define material-like parameters:
    - friction coefficients (static/dynamic)
    - compliance/softness parameters
    - damping terms to prevent oscillations
- **Connector interaction rules (physics-first, not “snap-first”)**
  - Stud insertion emerges from geometry + compliance:
    - misalignment should resist insertion
    - aligned approach should allow gradual insertion
- **Stability-first conditioning**
  - Introduce:
    - penetration caps / solver safeguards
    - bounded contact forces (avoid infinite impulses)
    - time-step / substep adjustments if needed
- **Calibration tests (must be scripted and repeatable)**
  - Single-stud insertion test:
    - approach trajectory
    - insertion depth vs force/effort curve proxy
  - Multi-stud insertion test (2×2 into baseplate region)
  - Pull-off retention test:
    - apply controlled upward force and check detachment threshold
- **Dataset relevance**
  - Ensure contact parameters remain stable across:
    - different brick sizes
    - multiple simultaneous contacts
    - repeated insert/remove cycles

### Milestone (minimum success criteria)
- Press-fit behavior is demonstrable and repeatable:
  - insertion succeeds when aligned,
  - fails or resists when misaligned,
  - connected bricks remain attached under moderate disturbance,
  - no frequent solver explosions or high-frequency jitter during insertion.

---

## 1.2.3) Baseplate & Workspace (Option A adapted for soft-real press-fit)
### What we will do
Create a baseplate that is **physically stable and press-fit compatible**, prioritizing realism while keeping the geometry and collisions solver-friendly.

### Why this matters
The baseplate anchors the entire assembly task. It defines the reference frame for “correct placement” and enables scalable multi-step builds.

### Execution checklist
- **Baseplate representation (A, realism-first)**
  - Use a visually accurate baseplate surface.
  - Implement studs as *press-fit compatible connector geometry* (consistent with brick studs/tubes).
- **Workspace layout**
  - Define a stable workspace frame relative to Alex torso:
    - baseplate pose
    - spawn regions
    - safety margins from robot self-collision
- **Surface/contact tuning**
  - Set baseplate friction and compliance to prevent “ice skating”.
  - Ensure studs do not cause catastrophic contact instabilities.
- **Validation tests**
  - Place a 2×2 brick on baseplate without connection (resting stability).
  - Insert/connect a brick to baseplate using controlled approach.
  - Verify retention and detach behavior.

### Milestone (minimum success criteria)
- Baseplate is stable and supports soft-real press-fit connections:
  - bricks can be placed and connected,
  - connections remain stable,
  - the environment remains numerically stable across repeated episodes.

---

## 1.2.4) Multi-View Rendering & What to Save (best-for-project + startup)
### What we will do
Implement a camera and data logging setup that maximizes:
- learning signal (multi-view perception),
- debugging capability (segmentation/depth),
- dataset utility for future methods and ablations.

### Why this matters
Startups care about **data** and **measurement**. Capturing richer modalities early (given abundant compute/storage) enables faster iteration, stronger results, and cleaner evaluation.

### Recommended logging (chosen for you)
**Per camera (4 views):**
- **RGB** (primary training signal)
- **Depth** (stabilizes geometric reasoning; great for insertion)
- **Segmentation / instance IDs** (debug + evaluation + potential auxiliary losses)

**Camera set (frozen):**
1) Overhead / workspace
2) Left wrist
3) Right wrist
4) Third-person

**Resolution & rate (balanced high-quality, not wasteful)**
- **Resolution:** 320×320 (or 384×384 if stable throughput is confirmed)
- **Frame rate:** match policy rate **20 Hz** (synchronized with state/action)
- **Compression:** lossless or high-quality (dataset-dependent), but keep raw availability for eval runs

**Alignment rules**
- All views are rendered at the **same sim step** (strict synchronization).
- Each frame includes metadata:
  - camera intrinsics/extrinsics
  - sim time / step index
  - episode ID and seed

### Execution checklist
- Implement four camera mounts and verify correct tracking (wrist cameras must be rigidly attached to EE frames).
- Implement RGB + depth + segmentation capture.
- Implement deterministic alignment with state/action logs.
- Add quick “dataset sanity viewer” to confirm:
  - camera correctness,
  - no swapped left/right views,
  - stable framing across resets.

### Milestone (minimum success criteria)
- Four-view synchronized recording works reliably headless.
- RGB/depth/segmentation are consistent and aligned with state/action.
- Recorded sequences are visually and geometrically meaningful for LEGO insertion.

---

## 1.2.5) Block Spawning & Reset — Best long-term choice (work upfront that pays off)
### What we will do
Implement a **high-reliability episode manager**: deterministic spawns, collision-free initialization, settle phase, curriculum hooks, and full reproducibility.

### Why this matters
This is one of the strongest “engineering signals” for startups: the ability to create scalable, reproducible, automated data generation pipelines.

### Execution checklist (best-practice approach)
- **Deterministic reset**
  - Every episode is generated from a seed; seed is logged.
  - Environment config is versioned (brick set, baseplate pose, camera params).
- **Constraint-based spawn sampling**
  - Spawn bricks within defined regions with:
    - min-distance constraints
    - orientation constraints (upright)
    - no initial intersections (collision checks)
- **Settle phase**
  - After spawning, run a settle period (no robot action) to let objects stabilize.
- **Curriculum scaffolding (even if simple in 1.2)**
  - Level 1: single brick placement
  - Level 2: single connection
  - Level 3: multi-step assembly
- **Reset reliability metrics**
  - Track:
    - % successful resets
    - average settle time
    - failure reasons (overlap, instability, out-of-bounds)

### Milestone (minimum success criteria)
- Reset success rate is high (target ≥ 95% over stress tests).
- Episodes are reproducible by seed.
- Spawning supports multi-brick scenes without frequent invalid starts.

---

## 1.2.6) MVP-3 Task (multi-step stacking/assembly) — adapted to soft real press-fit
### What we will do
Implement an MVP that demonstrates true assembly: **multi-step stacking** of multiple bricks on the baseplate with real(soft) press-fit connections—suitable for compelling demos and scalable dataset generation.

### Why this matters
MVP-3 is the strongest showcase: it demonstrates perception, precision, bimanual control, and contact-rich manipulation—exactly the profile that resonates with robotics startups.

### MVP-3 specification (concrete, consistent with decisions)
- **Brick set:** {2×2, 2×4, 2×6}
- **Assembly goal:** build a small structure with **2–4 bricks**:
  - Step 1: connect first brick to baseplate at target studs
  - Step 2: connect second brick onto first brick (press-fit)
  - Step 3+: optional additional stacking or bridging with a 2×4/2×6
- **Success criteria (episode-level)**
  - Each connection must be physically formed (not teleported):
    - studs inserted into sockets via soft press-fit
  - Final structure is stable for N seconds (no collapse) after the last placement
- **Evaluation signals (must be measurable)**
  - connection count and correctness (via connector metadata)
  - final pose errors (brick poses relative to targets)
  - stability duration without detachment

### Execution checklist
- Define target structures parametrically (connector map defines legal placements).
- Implement objective specification:
  - per-step subgoal definitions (which brick where)
  - success detection based on connector engagement + pose constraints
- Implement failure detection:
  - dropped brick, unstable oscillations, misaligned insertion
- Validate end-to-end scripted rollouts (not policy yet):
  - scripted “gold” trajectories that complete the build reliably

### Milestone (minimum success criteria)
- A scripted controller can complete an MVP-3 structure reliably (baseline feasibility).
- Connections form via soft press-fit (not snap).
- Success detection and metrics are accurate and reproducible.

---

# Startup-Grade Outputs (deliverables you should explicitly claim)
### What we will produce by end of 1.2
- **Procedural LEGO asset generator** (2×2/2×4/2×6) with connector metadata
- **Soft Real Press-Fit model** (documented parameters + calibration tests)
- **Baseplate + workspace** with reliable press-fit behavior
- **Four-view dataset logging** (RGB + depth + segmentation) synchronized with state/action
- **Episode manager** with deterministic seeding and reset reliability metrics
- **MVP-3 multi-step assembly scenario** + scripted feasibility rollouts
- **Short validation report**:
  - press-fit calibration results
  - reset reliability stats
  - example MVP-3 rollouts (video + metrics)

---

# Phase 1.2 Definition of Done
Phase 1.2 is complete when:
- Procedural bricks (2×2/2×4/2×6) and baseplate exist with connector metadata.
- Soft-real press-fit contacts are stable and validated via repeatable tests.
- Four cameras record RGB+depth+segmentation in sync with state/action.
- Spawning/reset is deterministic and reliable (high success rate).
- MVP-3 multi-step assembly is feasible via scripted rollouts and measurable success metrics.
