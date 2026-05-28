# Phase 1.2 - Door-Opening Environment Creation

**Goal:** Build a MuJoCo door-opening environment for IHMC Alex that is stable enough for scripted baselines, dataset generation, and later VLA training.

## Scope

The active environment is articulated-object manipulation, not object assembly. The scene should model a door panel, hinge, handle, optional latch behavior, contact-rich interaction with the Alex end effectors, and measurable opening success.

## 1.2.0 Door Task Contract

Define the core task API and metrics:

- door hinge joint name and axis
- handle site or geom name
- closed angle, target angle, and maximum angle
- latch state and release threshold
- success metrics: handle contact, latch release, maximum angle, final angle error, force-limit violations, and recovery success

Config source: `configs/sim/door.yaml`.

## 1.2.1 Door MJCF Assets

Create solver-friendly assets for:

- simple hinged door
- cabinet-style door
- lever handle
- round knob
- push bar
- optional frame, threshold, and latch proxy

Acceptance:

- assets load through `sim.asset_loader.load_scene()`
- hinge joint is limited and numerically stable
- handle geoms are reachable and contact-enabled
- scene lints clean with `python scripts/validate_assets.py`

## 1.2.2 Contact and Latch Modeling

Start with a simple free hinge, then add latch proxies:

- free-hinge pull or push
- lever rotation before hinge release
- knob rotation before hinge release
- contact-slip detection at the handle
- torque and force thresholds for safety

The first latch model can be a hybrid proxy that gates hinge motion after handle rotation exceeds a threshold. Do not overfit the first version to exact hardware mechanics.

## 1.2.3 Alex Door Workspace

Active scene:

- `sim/assets/scenes/alex_door_workspace.xml`

Required cameras:

- overhead
- third_person
- handle_closeup
- later: wrist cameras when mounted and calibrated

Workspace acceptance:

- both end effectors can reach the handle region
- handle is visible in at least one external camera
- scene runs for 500+ simulation steps without NaNs

Validation:

```bash
python scripts/validate_door_scene.py
python scripts/validate_kinematics.py
vla-viewer sim/assets/scenes/alex_door_workspace.xml --show-contacts --show-joints
```

## 1.2.4 Episode Manager

Implement a door-specific episode manager that randomizes:

- initial door angle
- handle type
- latch enabled/disabled
- target opening angle
- robot start posture
- camera lighting and small object-pose perturbations

The manager should return metadata suitable for dataset generation:

- seed
- scene variant
- target angle
- initial angle
- handle type
- latch state
- success metrics

## 1.2.5 Scripted Baseline

Create a scripted controller before training:

1. move to pre-handle pose
2. approach handle
3. close or hook gripper
4. rotate or press handle if latched
5. pull or push along an opening arc
6. release and retract

Acceptance:

- opens the free-hinge door reliably
- produces 17-D actions through `SimRunner`
- records synchronized images, robot state, action, and door metrics

## 1.2.6 Validation Suite

Add focused validators:

- hinge range and stability
- handle contact
- latch release
- scripted open success
- camera visibility
- force/torque limits

Minimum first milestone:

```bash
python scripts/validate_door_scene.py
```

## Deliverables

- Door task config
- Door workspace scene
- Door validation script
- Scripted baseline plan
- Door metrics schema
- Dataset-generation readiness checklist
