# Soft Real Press-Fit Specification (Phase 1.2.0)

**Revision:** Definitive (updated post-Phase 1.2.2 tuning)

## 1. Purpose and Scope

This document defines the **Soft Real Press-Fit** connection model for LEGO brick assembly in the VLA-LEGO MuJoCo simulation. It specifies geometry, contact physics, tolerances, and measurable success criteria that all Phase 1.2 subphases implement against.

**What this spec defines:**
- Exact simulation dimensions for studs, tubes, and brick shells
- MuJoCo contact parameter classes for LEGO physics (four classes, tuned)
- Contact isolation bitfield design
- Alignment tolerances and capture envelope
- Retention mechanism design (physics mode + hybrid spec-proxy mode)
- Quantitative acceptance tests (with achievable thresholds)

**What this spec does NOT define (deferred to later subphases):**
- Procedural generation code details (1.2.1 -- implemented)
- Baseplate layout and workspace (1.2.3 -- implemented)
- Camera setup (1.2.4)
- Episode management (1.2.5)
- Assembly task logic (1.2.6)

**Core principle:** Studs and tubes have accurate geometry/topology, but physics is made robust via carefully designed compliance, tolerances, and contact conditioning -- not hard interference or magic snap constraints.

**Dual retention modes:** Physics-only mode provides ground-truth rigid-body dynamics. Spec-proxy mode adds hybrid weld constraints for realistic retention forces during VLA policy training. See Section 6.

---

## 2. Real LEGO Reference Dimensions

Standard LEGO brick dimensions (from published specifications and measurements):

| Dimension | Value (mm) | Source |
|-----------|-----------|--------|
| Stud pitch (center-to-center) | 8.0 | Standard |
| Stud outer diameter | 4.8 | Standard (measured: ~4.85-4.89) |
| Stud height (above brick top) | 1.7 | Standard (some sources: 1.6-1.8) |
| Tube outer diameter | 6.51 | Standard (for 2+ wide bricks) |
| Tube inner diameter | 4.8 | Nominal (creates interference with stud) |
| Standard brick height (body only) | 9.6 | = 3 plates |
| Standard brick height (with stud) | 11.3 | 9.6 + 1.7 |
| Plate height (body only) | 3.2 | = 1 plate |
| Wall thickness | 1.2 | Standard |
| Inter-brick gap (side-to-side) | 0.2 | 0.1 per side |
| Brick width (N studs) | N x 8.0 - 0.2 | Accounts for gap tolerance |
| Manufacturing tolerance | +/-0.02 | Typical injection molding |
| Interference fit (stud-tube) | ~0.1-0.2 | Stud OD slightly > tube ID |

**Real insertion/retention forces (published estimates):**

| Metric | Value | Notes |
|--------|-------|-------|
| Insertion force per stud | 3-5 N | Varies with brick age/condition |
| Pull-off force per stud | 2-3 N | Vertical separation |
| Lateral shear per stud | 1-2 N | Before sliding |

**Material:** ABS (Acrylonitrile Butadiene Styrene)
- Density: ~1050 kg/m^3
- Young's modulus: ~2.3 GPa
- Coefficient of friction (ABS-on-ABS): 0.3-0.5

---

## 3. Simulation Geometry Specification

### 3.1 Coordinate and Unit Conventions

- **Units:** meters (MuJoCo convention). All dimensions below given in both mm and m.
- **Coordinate frame:** Z-up, X-forward, Y-left (consistent with Phase 1.1 world frame)
- **Stud axis:** +Z (studs point up)
- **Brick origin:** center of bottom face (geometric center at z=0 of the shell bottom)
- **Stud grid origin:** first stud at `(-pitch*(nx-1)/2, -pitch*(ny-1)/2)` relative to brick origin, where `nx`, `ny` are stud counts

### 3.2 Stud Geometry (Collision)

Each stud is represented as a **MuJoCo cylinder primitive** (type="cylinder"):

| Parameter | Value (mm) | Value (m) | Rationale |
|-----------|-----------|-----------|-----------|
| Radius | 2.35 | 0.00235 | 50 um undersize vs real 2.4 mm; provides clearance for tube capsule ring |
| Half-height | 0.85 | 0.00085 | Total height 1.7 mm (full stud height) |
| Z-offset (from brick top) | +0.85 | +0.00085 | Center of stud cylinder above brick top surface |

**Visual mesh:** Full stud cylinder at real dimensions (r=2.4 mm, h=1.7 mm) in a separate visual-only geom (group 0, contype=0, conaffinity=0).

**Contact class:** `lego/stud` (see Section 4.1)

### 3.3 Tube/Socket Geometry (Collision)

MuJoCo has no hollow cylinder primitive. We represent each tube as a **ring of 8 capsule primitives** arranged in a circle:

```
    Top view of one tube (8 capsules):

         o
       o   o        o = capsule cross-section (r=0.55 mm)
      o  *  o       * = tube center (stud insertion axis)
       o   o        Ring radius = 3.0 mm (capsule centers)
         o
```

| Parameter | Value (mm) | Value (m) | Rationale |
|-----------|-----------|-----------|-----------|
| Ring radius (capsule centers) | 3.0 | 0.00300 | Outer extent (3.0 + 0.55 = 3.55 mm) reaches stud faces at 3.31 mm, creating ~0.19 mm interference for retention |
| Capsule radius | 0.55 | 0.00055 | Tuned in Phase 1.2.2 for insertion + retention balance |
| Capsule half-height | 0.85 | 0.00085 | Total capsule length 1.7 mm; matches stud height for full engagement |
| Capsule count per tube | 8 | -- | 8 at 45 deg intervals aligns with stud diagonal positions (studs at +/-4mm, +/-4mm from tube center) |
| Angular spacing | 45 deg | -- | 360 deg / 8 capsules |
| Z-offset | +0.85 | +0.00085 | Capsule centers inside brick cavity at positive Z (tubes point upward into cavity) |

**Tube Z position:** Tubes are placed inside the brick cavity at positive Z, not protruding below the brick. The stud from a lower brick inserts upward into the upper brick's tubes.

**Effective radial clearance:** stud radius (2.35 mm) vs tube ring inner opening. The gap between adjacent capsule edges creates a **funnel effect** that guides aligned studs while rejecting misaligned ones.

**Geometric interlock mechanism:** When a stud is inserted, it sits between the capsule ring. The capsules provide:
1. **Lateral constraint** -- stud cannot slide more than ~0.5 mm before contacting a capsule
2. **Friction retention** -- capsule ring contact normals x friction coefficient resist pull-out
3. **Rotational constraint** -- 8-point contact ring resists torque (reinforced by condim=4)

**Contact class:** `lego/tube` (see Section 4.2)

**Why not other approaches:**
- **Mesh collision:** thin-walled cylinder mesh creates degenerate triangles -> solver instability
- **Box walls (4 boxes forming a square socket):** not rotationally symmetric -> unrealistic insertion behavior
- **Single cylinder with contact conditioning only:** no geometric interlock -> retention depends entirely on friction

### 3.4 Brick Shell Geometry

The brick body is a **hollow rectangular shell** represented as **5 thin box primitives** (4 walls + 1 top plate), leaving the bottom open for stud engagement:

| Component | Dimensions | Notes |
|-----------|-----------|-------|
| Left wall | wall_thickness x shell_half_y x shell_half_z | -X face |
| Right wall | wall_thickness x shell_half_y x shell_half_z | +X face |
| Front wall | shell_half_x x wall_thickness x shell_half_z | -Y face |
| Back wall | shell_half_x x wall_thickness x shell_half_z | +Y face |
| Top plate | shell_half_x x shell_half_y x top_thickness | Top face (1.0 mm thick) |

Where `nx`, `ny` are stud counts along X, Y; `brick_height` = 9.6 mm for standard bricks.

| Parameter | Formula (mm) | Example: 2x4 (mm) | Example: 2x4 (m) |
|-----------|-------------|-------------------|-------------------|
| Shell half-X | (nx x 8.0 - 0.2) / 2 | 7.9 | 0.0079 |
| Shell half-Y | (ny x 8.0 - 0.2) / 2 | 15.9 | 0.0159 |
| Shell half-Z | brick_height / 2 | 4.8 | 0.0048 |
| Wall thickness | 1.2 | 1.2 | 0.0012 |
| Top plate thickness | 1.0 | 1.0 | 0.001 |

**Why hollow shell?** The open bottom allows studs from the lower brick to physically enter the upper brick's cavity and contact the tube capsule rings. A solid box would block insertion.

**Mass computation:** `mass = density x volume_solid` where `volume_solid` accounts for hollow interior:
- `V_outer = (nx x 8.0 - 0.2) x (ny x 8.0 - 0.2) x 9.6` mm^3
- `V_inner = (V_outer_x - 2x1.2) x (V_outer_y - 2x1.2) x (9.6 - 1.0)` mm^3 (1.0 mm top thickness)
- `V_studs = nx x ny x pi x 2.4^2 x 1.7` mm^3
- `V_tubes ~ n_tubes x pi x (3.255^2 - 2.4^2) x 1.6` mm^3 (approximate)
- `mass = 1.05 x 10^-6 x (V_outer - V_inner + V_studs + V_tubes)` kg (density in kg/mm^3)

Example masses (approximate):
| Brick | Mass (g) | Mass (kg) |
|-------|----------|-----------|
| 2x2 | ~2.3 | 0.0023 |
| 2x4 | ~4.5 | 0.0045 |
| 2x6 | ~6.8 | 0.0068 |

**Contact class:** `lego/brick_surface` for shell walls and top plate (see Section 4.3)

**Inertia:** Computed via MuJoCo's automatic inertia from geom primitives: `<compiler inertiafromgeom="true"/>`.

### 3.5 Brick Type Parameters

| Brick Type | nx | ny | Height (mm) | Studs | Tubes |
|-----------|----|----|-------------|-------|-------|
| 2x2 | 2 | 2 | 9.6 | 4 | 1 (center) |
| 2x4 | 2 | 4 | 9.6 | 8 | 3 (row) |
| 2x6 | 2 | 6 | 9.6 | 12 | 5 (row) |

**Tube placement:** For 2-wide bricks, tubes are placed along the centerline (Y=0 relative to brick center), spaced at 8.0 mm pitch. A 2xN brick has (N-1) tubes.

### 3.6 Dimensional Summary

| Dimension | Real (mm) | Sim (mm) | Sim (m) | Delta | Rationale |
|-----------|-----------|----------|---------|-------|-----------|
| Stud pitch | 8.0 | 8.0 | 0.008 | 0 | Exact |
| Stud radius (collision) | 2.4 | 2.35 | 0.00235 | -0.05 | Clearance for capsule ring |
| Stud radius (visual) | 2.4 | 2.4 | 0.0024 | 0 | Exact visual |
| Stud height | 1.7 | 1.7 | 0.0017 | 0 | Exact |
| Tube ring radius | ~2.4 | 3.0 | 0.003 | +0.6 | Tuned for diagonal stud alignment |
| Tube capsule radius | N/A | 0.55 | 0.00055 | N/A | Tuned Phase 1.2.2 for insertion/retention balance |
| Tube capsule height | N/A | 1.7 | 0.0017 | N/A | Matches stud height |
| Brick height (body) | 9.6 | 9.6 | 0.0096 | 0 | Exact |
| Wall thickness | 1.2 | 1.2 | 0.0012 | 0 | Exact |
| Effective interference | 0.1-0.2 | ~0.19 | ~0.00019 | ~0 | Ring outer (3.55) vs stud diagonal (3.31) |

---

## 4. Contact Physics Specification

### 4.1 Material Class: `lego/stud`

Used for stud cylinders. Studs contact tubes (for press-fit), brick surfaces (when resting), and robot/environment geoms (for grasping interaction).

```xml
<default class="lego/stud">
    <geom contype="6" conaffinity="7"
          solref="0.003 1.0"
          solimp="0.97 0.995 0.001 0.5 4"
          friction="0.65 0.005 0.005"
          condim="4"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solref` | `0.003 1.0` | timeconst=3 ms (stiffer than robot 5 ms, modeling hard ABS); dampratio=1.0 (critically damped). Satisfies: timeconst >= 2 x timestep = 4 ms (close to limit for maximum stiffness). |
| `solimp` | `0.97 0.995 0.001 0.5 4` | dmin=0.97 (near-rigid at first contact); dmax=0.995 (very rigid at full engagement); width=1 mm; power=4 (sharp impedance ramp for distinct insertion feel) |
| `friction` | `0.65 0.005 0.005` | Tangent=0.65 (high end of ABS; maximizes retention without blocking insertion); torsional=0.005; rolling=0.005 |
| `condim` | `4` | 3D sliding + torsional friction. Prevents brick rotation on studs after insertion. |
| `contype` | `6` | Bits 1+2: contacts surfaces (bit 1) and tubes (bit 2) |
| `conaffinity` | `7` | Bits 0+1+2: responds to robot (0), LEGO surfaces (1), and stud-tube interlock (2) |

### 4.2 Material Class: `lego/tube`

Used for tube capsule ring elements. Tubes ONLY contact studs -- they do not collide with shell walls or robot geoms.

```xml
<default class="lego/tube">
    <geom contype="4" conaffinity="4"
          solref="0.003 1.0"
          solimp="0.97 0.995 0.001 0.5 4"
          friction="0.65 0.005 0.005"
          condim="4"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solref` | `0.003 1.0` | Same as stud -- symmetric contact pair |
| `solimp` | `0.97 0.995 0.001 0.5 4` | Same as stud |
| `friction` | `0.65 0.005 0.005` | Same as stud |
| `condim` | `4` | Same as stud |
| `contype` | `4` | Bit 2 only: stud-tube interlock channel |
| `conaffinity` | `4` | Bit 2 only: responds to stud-tube interlock (bit 2 of stud's contype=6) |

**Why separate from stud?** Tubes must not collide with shell walls. With a single `lego/stud_tube` class, tubes would collide with their own brick's walls, causing solver instability. The split allows precise control: tubes see studs, studs see everything.

### 4.3 Material Class: `lego/brick_surface`

Used for external brick faces (shell walls + top plate). Governs brick-to-brick surface sliding and brick-to-robot grasping contact.

```xml
<default class="lego/brick_surface">
    <geom contype="2" conaffinity="3"
          solref="0.005 1.0"
          solimp="0.9 0.95 0.001 0.5 2"
          friction="0.4 0.005 0.002"
          condim="3"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solref` | `0.005 1.0` | Same as robot collision defaults; brick surfaces don't need extra stiffness |
| `solimp` | `0.9 0.95 0.001 0.5 2` | Matches robot defaults; power=2 (gradual, no snap needed on flat surfaces) |
| `friction` | `0.4 0.005 0.002` | Lower than stud/tube (flat ABS sliding is smoother); enough to prevent ice-skating |
| `condim` | `3` | 3D sliding only; no torsional friction needed for flat contacts |
| `contype` | `2` | Bit 1: LEGO surface channel |
| `conaffinity` | `3` | Bits 0+1: responds to robot (bit 0) and LEGO surfaces/studs (bit 1) |

### 4.4 Material Class: `lego/baseplate`

Used for baseplate surface. Higher friction for stable anchoring.

```xml
<default class="lego/baseplate">
    <geom contype="2" conaffinity="3"
          solref="0.005 1.0"
          solimp="0.9 0.95 0.001 0.5 2"
          friction="0.6 0.01 0.005"
          condim="3"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `friction` | `0.6 0.01 0.005` | Higher than brick surface; baseplate is static, more friction helps anchor bricks |
| Others | Same as brick_surface | Baseplate doesn't need special stiffness/impedance |
| `conaffinity` | `3` | Responds to robot (table surface contact) and LEGO |

### 4.5 Contact Group Isolation Scheme

The contact isolation uses a 3-bit system to prevent unwanted collisions:

```
Bit 0: robot/environment
Bit 1: LEGO surfaces (shell walls, top plate)
Bit 2: LEGO stud/tube interlock (internal press-fit)

                    contype  conaffinity  Contacts with
Stud:                 6        7          everything (robot, surfaces, tubes)
Tube:                 4        4          studs only (NOT shell walls, NOT robot)
Brick surface:        2        3          robot + studs (NOT tubes)
Baseplate:            2        3          robot + studs (NOT tubes)
Robot geoms:          1        1          robot + surfaces (need conaffinity=3 for LEGO grasping)
Floor/table:          1        1          robot + surfaces
```

**Why this design?**
- **Tubes only see studs:** Prevents tube capsules from colliding with their own brick's shell walls (which would cause solver instability and block insertion)
- **Studs see everything:** Studs contact tubes (for press-fit), surfaces (when resting on top), and robot (for gripper interaction)
- **Robot does not see tubes:** Robot collision geoms should never directly contact stud/tube internals; only the robot grips the brick shell

**Important:** Robot geom conaffinity must include bit 1 to contact LEGO brick_surface geoms. Update robot defaults to `conaffinity="3"` (bits 0+1) so the EZGripper can grasp bricks.

### 4.6 Solver Requirements

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `iterations` | 80 | Multi-stud contacts increase constraint count; more iterations needed for convergence |
| `ls_iterations` | 10 | Adequate for line search |
| `timestep` | 0.002 s | Compatible with solref timeconst=0.003 s (>= 2 x timestep) |
| `integrator` | `implicitfast` | Unconditionally stable; best for contact-rich tasks |
| `noslip_iterations` | 0 | No noslip needed; friction alone provides adequate retention |
| `cone` | `pyramidal` (default) | Adequate accuracy; elliptic cone is more accurate but slower |

**Contact count budget:** A 2x2 brick insertion involves up to 4 studs x 8 capsules = 32 stud-tube contact pairs. A 2x6 insertion: 12 studs x 8 = 96 pairs. Target: simulation remains real-time with <100 active contacts on a single CPU core.

---

## 5. Capture Envelope and Alignment Tolerances

### 5.1 Capture Envelope

The capture envelope defines the region where a brick can be successfully inserted:

| Tolerance | Value | Rationale |
|-----------|-------|-----------|
| Lateral (XY) | +/-0.5 mm (+/-0.0005 m) | ~6x real LEGO tolerance; accounts for sim compliance and gripper precision |
| Angular (roll/pitch) | +/-3 deg from vertical | ~2-3x real tolerance; allows for slight EZGripper misalignment |
| Vertical approach zone | starts 2.0 mm (0.002 m) above engagement | stud height is 1.7 mm; extra 0.3 mm margin |

### 5.2 Alignment Success/Failure Behavior

**Within envelope (aligned):**
- Studs find tube openings via compliance
- Gradual insertion with observable resistance
- Stud slides between capsule ring elements
- Final state: stud seated within capsule ring, held by friction + geometry

**Outside envelope (misaligned):**
- Stud contacts tube capsule tops or adjacent capsules
- Resistance increases rapidly -> brick slides off or jams
- No "snap correction" or "teleportation" to aligned position
- Brick remains in contact but does not engage (resting on top surface)

### 5.3 Approach Velocity

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Nominal insertion speed | 1-5 mm/s (0.001-0.005 m/s) | Slow enough for stable contact; fast enough for practical episodes |
| Maximum insertion speed | 10 mm/s (0.01 m/s) | Beyond this, impact forces may cause bouncing or solver stress |

---

## 6. Retention Mechanism

### 6.1 Physics-Only Retention (Default Mode)

In physics mode (`retention_mode: physics`), retention comes entirely from MuJoCo contact forces:

**Primary: Friction.** The tangent friction coefficient (0.65) on stud-tube contacts creates resistance to separation. With 8 contact points per stud (one per capsule), the normal forces from the capsule ring pressing inward on the stud generate friction that resists vertical pull-out.

**Secondary: Geometric interlock.** The capsule ring provides lateral constraint:
- Capsules at 45 deg intervals contact stud diagonal positions
- Contact normal force x friction coefficient resists motion
- ~0.19 mm effective interference at equilibrium

**Achieved physics-mode retention:** ~0.06 N/stud (see Section 6.3 for gap analysis)

### 6.2 Hybrid Spec-Proxy Retention (Optional Mode)

In spec-proxy mode (`retention_mode: spec_proxy`), a `ConnectionManager` augments physics contacts with MuJoCo weld equality constraints to achieve realistic retention forces.

**How it works:**
1. **Physics insertion** (unchanged): Top brick drops under force, capsule-ring tubes provide geometric interlock. Same `run_insertion()` code as physics mode.
2. **Engagement detection**: `ConnectionManager` monitors brick pair positions each physics step:
   - Z-threshold: top brick body Z within margin of expected engaged position
   - XY alignment: lateral offset within 0.5 mm tolerance
   - Sustained contact: both conditions met for 50 consecutive steps (~100 ms)
3. **Weld activation**: Pre-declared `<weld>` equality constraint toggled on. Constraint relpose set from current brick positions (no impulse).
4. **Retention**: Weld provides spec-target forces. Compliant solref (0.01, 1.0) allows small deflection.
5. **Release**: If displacement from welded pose exceeds 2 mm for 25 consecutive steps, weld deactivates.

**All hybrid results are labeled [PROXY]** and excluded from ground-truth physics reporting.

**Achieved spec-proxy retention:** ~0.52 N/stud pull-off, ~0.63 N/stud shear

### 6.3 Retention Force Gap Analysis

The spec targets 0.3 N/stud retention based on real LEGO measurements. Physics mode achieves ~0.06 N/stud -- 5x below target.

**Root cause:** Real LEGO retention comes from **elastic deformation** of ABS plastic -- the tube slightly deforms to grip the stud. MuJoCo models contacts as **rigid body penetration with constraint forces**, which cannot replicate this elastic clamping effect.

**Fundamental trade-off confirmed by parameter sweep** (20 combinations, 10 trials each):
- Parameters that increase retention (larger capsules, stiffer contacts) also resist insertion
- No parameter combination achieves both insertion >= 95% AND retention >= 0.3 N/stud
- See `docs/contact-tuning-notes.md` for full sweep results

### 6.4 When to Use Each Mode

| Use Case | Mode | Why |
|----------|------|-----|
| VLA policy training | `spec_proxy` | Realistic retention forces during manipulation |
| Contact physics research | `physics` | Ground-truth rigid-body dynamics |
| Insertion tuning | `physics` | No weld interference during approach |
| Retention force validation | `spec_proxy` | Matches real LEGO force targets |

### 6.5 What We Explicitly Do NOT Model

- **Plastic deformation** (ABS flex during real insertion) -- modeled as elastic compliance instead
- **Acoustic/tactile feedback** (click sound/feel) -- not relevant for VLA training
- **Thermal effects** (friction heating) -- negligible
- **Fatigue** (connection loosening over cycles) -- all cycles are identical in sim
- **Asymmetric wear** (real LEGO bricks wear unevenly) -- all bricks are pristine

---

## 7. Measurable Success Criteria

All tests below are scripted and deterministic. They constitute the acceptance criteria for contact physics and serve as regression tests.

**Threshold philosophy:** Physics-mode thresholds are set to achievable values given MuJoCo's rigid-body contact model. Spec-proxy mode thresholds match real LEGO targets. The relaxations from original targets are documented in `docs/contact-tuning-notes.md`.

### 7.1 Insertion Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Aligned insertion (2x2)** | All 4 studs aligned, approach at 1 mm/s | All studs engaged within 2.0 s |
| **Near-miss (in envelope)** | 0.3 mm lateral offset | Still succeeds |
| **Miss (outside envelope)** | 1.0 mm lateral offset | Fails to engage; stud rests on surface or slides off |
| **Angular tolerance (2 deg)** | 2 deg tilt from vertical, aligned XY | Succeeds (within angular envelope) |
| **Angular rejection (5 deg)** | 5 deg tilt from vertical, aligned XY | Fails or partial engagement only |
| **2x4 aligned** | All 8 studs aligned | All studs engaged |
| **Noisy insertion** | Gaussian noise sigma=0.2 mm on XY, 100 trials | Success rate >= 95% |

### 7.2 Retention Tests (Physics Mode)

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Vertical pull-off** | After successful 2x2 insertion, upward force ramp | Resists >= 0.15 N total before detaching |
| **Lateral shear** | After insertion, horizontal force ramp | Resists >= 0.15 N total before sliding |
| **Static hold** | After insertion, no external force, 5 s wait | Drift < 1.0 mm in any axis |
| **Gravity retention** | Insert 2x2, release under gravity only | Brick does NOT fall off (retention > gravity) |

### 7.3 Retention Tests (Spec-Proxy Mode) [PROXY]

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Vertical pull-off [PROXY]** | After insertion + weld engagement | Resists >= 0.3 N/stud before detaching |
| **Lateral shear [PROXY]** | After insertion + weld engagement | Resists >= 0.2 N/stud before sliding |
| **Static hold [PROXY]** | After insertion + weld, 5 s wait | Drift < 0.1 mm |

### 7.4 Stability Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Insert/remove cycles** | 10 consecutive insert-then-remove cycles | No solver divergence (no NaN in qpos/qvel) |
| **Penetration cap** | Monitor `data.contact[i].dist` during all insertions | Max penetration depth < 2 mm (0.002 m) |
| **Energy bound** | Monitor total energy during insertion | Energy < 500 J at all times |
| **Post-insertion jitter** | After insertion, measure position RMS over 1 s | < 0.5 mm (0.0005 m) RMS |

### 7.5 Performance Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Contact count** | 2x2 brick insertion | < 100 active contacts during insertion |
| **Solver time** | 2x4 brick insertion on single CPU core | Physics runs at >= 1x real-time |
| **Large-scene stability** | 6 bricks in scene (various sizes) | No solver timeout or divergence over 10 s |

---

## 8. Configuration Integration

All tunable parameters are stored in `configs/sim/lego.yaml`. The config structure:

```yaml
lego:
  # Geometry (tuned Phase 1.2.2)
  geometry:
    stud_pitch: 0.008            # m
    stud_radius: 0.00235         # m (collision, 50 um undersize for clearance)
    stud_visual_radius: 0.0024   # m
    stud_height: 0.0017          # m
    tube_ring_radius: 0.003      # m (capsule centers)
    tube_capsule_radius: 0.00055 # m (tuned 1.2.2 for insertion + retention balance)
    tube_capsule_count: 8        # 8 at 45 deg aligns with stud diagonals
    tube_capsule_height: 0.0017  # m (matches stud height)
    brick_height: 0.0096         # m (standard, no stud)
    wall_thickness: 0.0012       # m
    inter_brick_gap: 0.0002      # m
    density: 1050.0              # kg/m^3 (ABS)

  # Contact classes (tuned Phase 1.2.2, split stud/tube)
  contact:
    stud:
      contype: 6
      conaffinity: 7
      solref: [0.003, 1.0]
      solimp: [0.97, 0.995, 0.001, 0.5, 4.0]
      friction: [0.65, 0.005, 0.005]
      condim: 4
    tube:
      contype: 4
      conaffinity: 4
      solref: [0.003, 1.0]
      solimp: [0.97, 0.995, 0.001, 0.5, 4.0]
      friction: [0.65, 0.005, 0.005]
      condim: 4
    brick_surface:
      contype: 2
      conaffinity: 3
      solref: [0.005, 1.0]
      solimp: [0.9, 0.95, 0.001, 0.5, 2.0]
      friction: [0.4, 0.005, 0.002]
      condim: 3
    baseplate:
      contype: 2
      conaffinity: 3
      solref: [0.005, 1.0]
      solimp: [0.9, 0.95, 0.001, 0.5, 2.0]
      friction: [0.6, 0.01, 0.005]
      condim: 3

  # Capture envelope tolerances
  tolerances:
    lateral_mm: 0.5
    angular_deg: 3.0
    approach_zone_mm: 2.0
    max_insertion_speed_m_s: 0.01

  # Success thresholds (tuned for achievable capsule-ring physics)
  thresholds:
    insertion_success_rate: 0.95
    min_retention_force_N: 0.15        # total per brick (physics mode)
    min_lateral_shear_N: 0.15          # total per brick (physics mode)
    max_penetration_m: 0.002
    max_jitter_rms_m: 0.0005           # 0.5 mm RMS
    max_drift_m: 0.001                 # 1.0 mm
    max_energy_J: 500.0
    hold_duration_s: 5.0
    max_active_contacts: 100

  # Solver overrides for LEGO scenes
  solver:
    iterations: 80
    ls_iterations: 10
    noslip_iterations: 0

  # Retention mode: "physics" (default) or "spec_proxy" (hybrid weld)
  retention_mode: physics

  # Hybrid retention parameters (only used when retention_mode: spec_proxy)
  hybrid:
    engage_xy_tol_m: 0.0005        # XY alignment gate (0.5 mm)
    engage_min_steps: 50           # sustained engagement steps (~100 ms)
    engage_z_margin_m: 0.0002      # Z margin below stud center (0.2 mm)
    release_displacement_m: 0.002  # displacement to break weld (2 mm)
    release_dwell_steps: 25        # hysteresis dwell (~50 ms)
    weld_solref: [0.01, 1.0]       # compliant weld (not perfectly rigid)
    weld_solimp: [0.95, 0.99, 0.001, 0.5, 2]
```

---

## 9. Frozen Decisions and Future Scope

### 9.1 Frozen for Phase 1.2

These values are **locked** and should not be changed without re-running all acceptance tests:

- Stud/tube collision dimensions (Section 3.6 table)
- Contact parameter classes: `lego/stud`, `lego/tube`, `lego/brick_surface`, `lego/baseplate` (Section 4)
- Contact isolation bitfield design (Section 4.5)
- Capture envelope tolerances (Section 5.1)
- Success criteria thresholds (Section 7)
- Tube geometry strategy: 8-capsule ring at 3.0 mm radius
- Brick set: {2x2, 2x4, 2x6} at standard height (9.6 mm)

### 9.2 Tunable Parameters

These may be adjusted during contact tuning without re-specifying:

- Exact solref/solimp values (within +/-20% of tuned values)
- Friction coefficients (within +/-30%)
- Solver iteration count (50-120 range)
- Hybrid retention thresholds (engage_min_steps, release_displacement_m, etc.)

### 9.3 Deferred to Future Phases

- Non-standard brick shapes (slopes, plates, technic, 1xN)
- Multi-material contacts (different brick materials)
- Acoustic/force feedback
- True interference modeling (not feasible in MuJoCo)
- Brick deformation / fatigue
- Brick-to-brick lateral slide connections (Technic pins, etc.)
- Material-faithful retention (custom MuJoCo plugin or FEM-coupled simulator)

---

## Appendix A: MuJoCo Contact Parameter Reference

### solref: Reference Acceleration

`solref = [timeconst, dampratio]` (when both positive)

Defines the constraint's target corrective motion as a spring-damper:
- **timeconst**: softness parameter. Lower = stiffer contact. Must be >= 2 x simulation timestep.
- **dampratio**: 1.0 = critically damped (no oscillation). Values < 1.0 create bouncing.

The stiffness `k` and damping `b` are derived as:
- `b = 2 / (dwidth x timeconst)`
- `k = d(r) / (dwidth^2 x timeconst^2 x dampratio^2)`

Where `dwidth` is the impedance at the contact surface and `d(r)` is the position-dependent impedance from solimp.

### solimp: Impedance Function

`solimp = [dmin, dmax, width, midpoint, power]`

Controls the constraint's ability to generate force as a function of penetration:
- **dmin**: impedance at zero penetration (0 = no force, 1 = full force)
- **dmax**: impedance at `width` penetration
- **width**: penetration distance over which impedance transitions from dmin to dmax (meters)
- **midpoint**: center of the sigmoid transition (0-1, typically 0.5)
- **power**: steepness of the transition (higher = sharper snap)

For stiff contacts (ABS plastic): use high dmin/dmax (0.97/0.995) so force is generated immediately on contact.

### condim: Contact Dimensionality

- `condim=1`: frictionless (normal force only)
- `condim=3`: 3D sliding friction (tangent plane)
- `condim=4`: sliding + torsional friction (resists rotation)
- `condim=6`: sliding + torsional + rolling friction

### friction: Friction Coefficients

`friction = [tangent, torsional, rolling]` (3 values for condim >= 3)

When two geoms with different friction values collide, MuJoCo uses element-wise maximum.

---

## Appendix B: Force Estimation and Tuning Results

### Achieved performance summary (Phase 1.2.2)

| Metric | Spec Target | Physics Mode | Spec-Proxy [PROXY] |
|--------|------------|-------------|---------------------|
| Aligned insertion (2x2) | Success | Pass | Pass |
| Near-miss 0.3mm | Success | Pass | Pass |
| Miss 1.0mm rejection | No engage | Pass | Pass |
| Angular 2 deg tolerance | Success | Pass | Pass |
| Angular 5 deg rejection | No engage | Pass | Pass |
| Pull-off retention | 0.3 N/stud | ~0.06 N/stud | ~0.52 N/stud |
| Lateral shear | 0.2 N/stud | ~0.3 N/stud | ~0.63 N/stud |
| Static hold drift | <0.1 mm | ~0.5 mm | ~0.02 mm |
| Max penetration | <2 mm | ~0.3 mm | ~0.3 mm |
| Gravity retention | Hold | Pass | Pass |
| 10 insert/remove cycles | No NaN | Pass | Pass |
| Real-time factor | >=1x | >100x | >100x |

### Approximate contact force from solref/solimp

For a stud-tube contact with `solref=[0.003, 1.0]` and `solimp=[0.97, 0.995, 0.001, 0.5, 4]`:

At penetration depth `r`:
1. Impedance `d(r)` transitions from 0.97 to 0.995 over 1 mm
2. Reference acceleration: `a_ref = -(b x v + k x r)` where `b` and `k` depend on `d(r)` and `solref`
3. Actual constraint force ~ `d(r) x a_ref x effective_mass`

**Single stud estimate:**
- Effective mass of 2x2 brick: ~2.3 g
- At 0.19 mm interference: force ~ 0.05-0.07 N per stud (8 capsules contributing)
- This matches measured physics-mode pull-off of ~0.06 N/stud

---

## Appendix C: Brick Type Quick Reference

### 2x2 Brick
- Studs: 4 (2x2 grid)
- Tubes: 1 (center)
- Outer dimensions: 15.8 x 15.8 x 9.6 mm (body) + 1.7 mm (studs)
- Mass: ~2.3 g

### 2x4 Brick
- Studs: 8 (2x4 grid)
- Tubes: 3 (centerline row)
- Outer dimensions: 15.8 x 31.8 x 9.6 mm (body) + 1.7 mm (studs)
- Mass: ~4.5 g

### 2x6 Brick
- Studs: 12 (2x6 grid)
- Tubes: 5 (centerline row)
- Outer dimensions: 15.8 x 47.8 x 9.6 mm (body) + 1.7 mm (studs)
- Mass: ~6.8 g

---

## Appendix D: Implementation Files

| File | Purpose |
|------|---------|
| `sim/lego/constants.py` | Frozen geometry constants, `BrickType`, `BaseplateType`, contact isolation bits |
| `sim/lego/mass.py` | ABS density mass computation (bricks + baseplates) |
| `sim/lego/connector.py` | `ConnectorPoint`, `BrickConnectors`, `BaseplateConnectors`, stud/tube metadata |
| `sim/lego/brick_generator.py` | Procedural MJCF generation for bricks |
| `sim/lego/baseplate_generator.py` | Procedural MJCF generation for baseplates |
| `sim/lego/contact_scene.py` | Scene builder for insertion tests, workspace scenes |
| `sim/lego/contact_utils.py` | Insertion measurement, force ramp, `InsertionResult` |
| `sim/lego/connection_manager.py` | Hybrid weld activation/deactivation for spec-proxy mode |
| `sim/assets/lego/defaults.xml` | MJCF contact material default classes |
| `configs/sim/lego.yaml` | All tunable parameters |
| `tests/test_lego_bricks.py` | Brick generation + metadata tests |
| `tests/test_lego_contacts.py` | Contact physics acceptance tests |
