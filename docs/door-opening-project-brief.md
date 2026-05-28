# Door-Opening Project Brief

## Purpose

This project is the independent door-opening manipulation successor to the VLA-LEGO baseline. The initial codebase keeps the EO-1-style VLA model, training loop, Hydra configuration, tracking, MuJoCo utilities, and robot-state abstractions, while the task layer should move from LEGO assembly to articulated-object manipulation.

## Target Task

The main behavior is opening doors through contact-rich manipulation:

- perceive the door, handle, hinge axis, and current opening angle
- choose a handle contact or grasp strategy
- rotate or press the handle if a latch is present
- pull or push the door to a target angle
- recover from handle slip, insufficient latch rotation, blocked motion, or poor approach pose

## Refactor Map

- Replace `sim/lego/` task logic with a door, handle, hinge, and latch simulation module
- Add MJCF assets for hinged doors, cabinet doors, lever handles, round knobs, and push bars
- Replace LEGO metrics with door metrics: handle grasp success, latch release, max door angle, final target-angle error, contact stability, force/torque limits, and recovery success
- Keep the VLA backbone, action head, training loop, tracking, cluster, and container infrastructure unless the door task exposes a specific mismatch
- Treat IHMC Alex transfer as a later validation stage after simulation metrics are stable

## Immediate Next Milestones

1. Define the first MuJoCo door asset and scripted controller baseline
2. Add validation scripts for hinge motion, handle contact, and latch release
3. Add a minimal dataset schema for visual observations, language instructions, robot state, and continuous door-opening actions
4. Run a smoke test that opens a simple hinged door in simulation
