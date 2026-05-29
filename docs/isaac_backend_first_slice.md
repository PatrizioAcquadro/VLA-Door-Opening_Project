# Isaac Backend First Slice

This repository still keeps MuJoCo as the reference scaffold. The first Isaac
slice adds an import-safe contract layer and a live proof script; it does not
port training, datasets, or the full MuJoCo scene.

## Backend Surface

`sim/isaac/` exposes:

- `reset()`
- `step(action_17d)`
- `get_robot_state_52d()`
- `get_door_metrics()`
- `render_views()`
- `record_episode()`

Importing `sim.isaac` does not import Isaac Sim, Isaac Lab, `omni`, or `pxr`.
Live calls fail with `IsaacRuntimeUnavailable` until a real Isaac adapter is
injected. Contract helpers enforce the frozen 17-D action, 52-D state, four
320x320 RGB views, language instruction episode surface, and door metric names.

## Live Proof

Run the proof from this repo root:

```bash
python3 scripts/run_isaac_proof.py
```

The driver uses:

```text
/home/pacquadr/Desktop/isaac_suitcase/bin/isaaclab-run
```

It writes machine-checkable evidence under:

```text
logs/isaac_proof/<timestamp>/
```

Expected artifacts:

- `command_result.json`
- `stdout.log`
- `stderr.log`
- `isaac_proof.json`
- `alex_urdf_import.usd` when URDF import succeeds

The proof records Isaac launch/import, Isaac Lab/Python status, selected Alex
asset path, attempted asset import/conversion, PhysX hinged-door angle samples,
programmatic camera frame shape/dtype/nonconstant check, and contact sensor
force/contact fields.

## Current Asset Selection

The required order is existing Alex USD, then URDF import, then MJCF conversion.
The proof inventories the current repo and `/home/pacquadr/Desktop/Alex-robot`
at runtime. It does not use stale paths.

The current validated run found no Alex robot USD asset. It selected and
successfully imported:

```text
/home/pacquadr/Desktop/Alex-robot/alex_models/alex_V1_description/rl_urdf/alex_v1.rlModel_fullBody_robotAccurate_torsoFootCollisions.urdf
```

The URDF import produced `alex_urdf_import.usd` in the proof run directory and
reported 183 USD prims with `/Alex/PELVIS_LINK` as the imported articulation
root. The inventory also found the existing door USD at:

```text
/home/pacquadr/Desktop/Alex-robot/scenes/HallwayScene/Objects/DoorObject.usda
```

MJCF remains a fallback only if USD and URDF cannot be used.

## Latest Validated Evidence

The latest host-visible proof run completed with return code 0 under the Isaac
Lab wrapper and recorded RTX 4090 visibility:

```text
logs/isaac_proof/20260528T233335Z/isaac_proof.json
logs/isaac_proof/20260528T233335Z/command_result.json
logs/isaac_proof/20260528T233335Z/alex_urdf_import.usd
```

The proof passed:

- Isaac Sim headless launch/import
- Isaac Lab/Python import
- Alex URDF import to USD
- simple PhysX hinged-door motion
- programmatic 320x320 RGB camera read
- programmatic contact sensor read

## Limits

This slice proves the first Isaac vertical path and contract boundary. It is not
a complete Alex door environment, not a training path, and not a MuJoCo
replacement. Isaac failures are recorded as evidence with exact command output
instead of falling back to MuJoCo or fake camera/contact data.
