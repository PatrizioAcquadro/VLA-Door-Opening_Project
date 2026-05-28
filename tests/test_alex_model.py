"""Tests for Alex V1 upper-body robot model (Phase 1.1.1 + 1.1.2 + 1.1.3).

Run:
    pytest tests/test_alex_model.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

# Canonical joint names in expected order (15 arm + 8 EZGripper)
EXPECTED_JOINTS = [
    "spine_z",
    "left_shoulder_y",
    "left_shoulder_x",
    "left_shoulder_z",
    "left_elbow_y",
    "left_wrist_z",
    "left_wrist_x",
    "left_gripper_z",
    "left_knuckle_palm_l1_1",
    "left_knuckle_l1_l2_1",
    "left_knuckle_palm_l1_2",
    "left_knuckle_l1_l2_2",
    "right_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_z",
    "right_elbow_y",
    "right_wrist_z",
    "right_wrist_x",
    "right_gripper_z",
    "right_knuckle_palm_l1_1",
    "right_knuckle_l1_l2_1",
    "right_knuckle_palm_l1_2",
    "right_knuckle_l1_l2_2",
]

# Expected joint ranges (radians) from URDF — (lower, upper) per joint
EXPECTED_RANGES = {
    "spine_z": (-0.5236, 0.5236),
    "left_shoulder_y": (-3.1416, 1.2217),
    "left_shoulder_x": (-0.3491, 2.7925),
    "left_shoulder_z": (-1.9199, 1.2217),
    "left_elbow_y": (-2.3562, 0.1745),
    "left_wrist_z": (-2.6180, 2.6180),
    "left_wrist_x": (-1.8326, 0.6109),
    "left_gripper_z": (-2.6180, 2.6180),
    "right_shoulder_y": (-3.1416, 1.2217),
    "right_shoulder_x": (-2.7925, 0.3491),
    "right_shoulder_z": (-1.2217, 1.9199),
    "right_elbow_y": (-2.3562, 0.1745),
    "right_wrist_z": (-2.6180, 2.6180),
    "right_wrist_x": (-0.6109, 1.8326),
    "right_gripper_z": (-2.6180, 2.6180),
    # EZGripper joints (Phase 1.1.3)
    "left_knuckle_palm_l1_1": (0.0, 1.94),
    "left_knuckle_l1_l2_1": (0.0, 1.94),
    "left_knuckle_palm_l1_2": (0.0, 1.94),
    "left_knuckle_l1_l2_2": (0.0, 1.94),
    "right_knuckle_palm_l1_1": (0.0, 1.94),
    "right_knuckle_l1_l2_1": (0.0, 1.94),
    "right_knuckle_palm_l1_2": (0.0, 1.94),
    "right_knuckle_l1_l2_2": (0.0, 1.94),
}


@pytest.fixture(scope="module")
def alex_model():
    """Load the Alex upper-body scene."""
    from sim.asset_loader import load_scene

    return load_scene("alex_upper_body")


@pytest.fixture(scope="module")
def alex_data(alex_model):
    """Create MjData for the Alex model."""
    return mujoco.MjData(alex_model)


# ---------------------------------------------------------------------------
# A. Loading tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexModelLoads:
    """Verify the Alex model loads correctly through the asset pipeline."""

    def test_load_alex_scene(self) -> None:
        from sim.asset_loader import load_scene

        model = load_scene("alex_upper_body")
        assert model.nq > 0

    def test_alex_robot_path_resolves(self) -> None:
        from sim.asset_loader import resolve_robot_path

        path = resolve_robot_path("alex")
        assert path.exists()
        assert path.name == "alex.xml"

    def test_alex_lint_clean(self) -> None:
        from sim.asset_linter import Severity, lint_mjcf
        from sim.asset_loader import resolve_robot_path

        path = resolve_robot_path("alex")
        issues = lint_mjcf(path)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Lint errors: {errors}"

    def test_alex_scene_lint_clean(self) -> None:
        from sim.asset_linter import Severity, lint_mjcf
        from sim.asset_loader import SCENES_DIR

        issues = lint_mjcf(SCENES_DIR / "alex_upper_body.xml")
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Lint errors: {errors}"


# ---------------------------------------------------------------------------
# B. Joint contract tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexJointContract:
    """Verify the Alex model's joint structure matches the contract."""

    def test_joint_count(self, alex_model) -> None:
        assert alex_model.njnt == 23  # 15 arm + 8 EZGripper

    def test_no_freejoint(self, alex_model) -> None:
        """No freejoint means nq == njnt (all hinge joints, 1 DoF each)."""
        assert alex_model.nq == 23
        assert alex_model.nv == 23

    def test_joint_names(self, alex_model) -> None:
        actual = []
        for i in range(alex_model.njnt):
            name = mujoco.mj_id2name(alex_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            actual.append(name)
        assert actual == EXPECTED_JOINTS

    def test_spine_z_is_first(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
        assert idx == 0

    def test_joint_ranges_match_spec(self, alex_model) -> None:
        for name, (lo, hi) in EXPECTED_RANGES.items():
            idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert idx >= 0, f"Joint {name} not found"
            actual_lo = alex_model.jnt_range[idx, 0]
            actual_hi = alex_model.jnt_range[idx, 1]
            assert abs(actual_lo - lo) < 0.01, f"{name} lower: expected {lo}, got {actual_lo}"
            assert abs(actual_hi - hi) < 0.01, f"{name} upper: expected {hi}, got {actual_hi}"

    def test_actuator_count(self, alex_model) -> None:
        assert alex_model.nu == 17  # 15 arm + 2 EZGripper


# ---------------------------------------------------------------------------
# C. Site and camera tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexSitesAndCameras:
    """Verify EE sites and cameras exist."""

    def test_left_ee_site_exists(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_SITE, "left_ee_site")
        assert idx >= 0

    def test_right_ee_site_exists(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")
        assert idx >= 0

    def test_robot_cam_exists(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam")
        assert idx >= 0

    def test_overhead_cam_exists(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        assert idx >= 0

    def test_third_person_cam_exists(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")
        assert idx >= 0


# ---------------------------------------------------------------------------
# D. Stability tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexStability:
    """Verify the Alex model is numerically stable."""

    def test_gravity_settle_no_explosion(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        for _ in range(5000):
            mujoco.mj_step(alex_model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.max(np.abs(data.qpos)) < 10.0

    def test_no_nans(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        for _ in range(5000):
            mujoco.mj_step(alex_model, data)
        assert not np.any(np.isnan(data.qpos))
        assert not np.any(np.isnan(data.qvel))

    def test_deterministic(self) -> None:
        from sim.asset_loader import resolve_scene_path
        from sim.mujoco_env import check_deterministic

        path = resolve_scene_path("alex_upper_body")
        assert check_deterministic(str(path))

    def test_home_pose_no_penetration(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        mujoco.mj_forward(alex_model, data)
        for i in range(data.ncon):
            dist = data.contact[i].dist
            assert dist > -0.005, f"Contact {i} penetration {dist:.4f}m exceeds 5mm"

    def test_energy_bounded(self, alex_model) -> None:
        data = mujoco.MjData(alex_model)
        for _ in range(2000):
            mujoco.mj_step(alex_model, data)
        total_energy = data.energy[0] + data.energy[1]  # potential + kinetic
        assert np.isfinite(total_energy), "Energy is NaN/Inf"
        assert total_energy < 1000.0, f"Energy exploded: {total_energy:.1f} J"


# ---------------------------------------------------------------------------
# E. Collision geometry tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexCollision:
    """Verify collision geometry setup."""

    def test_has_collision_geoms(self, alex_model) -> None:
        collision_count = sum(1 for i in range(alex_model.ngeom) if alex_model.geom_contype[i] > 0)
        assert collision_count > 0, "No collision-capable geoms found"

    def test_has_visual_geoms(self, alex_model) -> None:
        visual_count = sum(
            1
            for i in range(alex_model.ngeom)
            if alex_model.geom_contype[i] == 0 and alex_model.geom_conaffinity[i] == 0
        )
        assert visual_count > 0, "No visual-only geoms found"


# ---------------------------------------------------------------------------
# F. Dynamics contract tests (Phase 1.1.2)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexDynamicsContract:
    """Verify dynamics parameters match Phase 1.1.2 specification."""

    def test_spine_has_armature(self, alex_model) -> None:
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
        assert alex_model.dof_armature[idx] == pytest.approx(0.01, abs=1e-6)

    def test_all_joints_have_armature(self, alex_model) -> None:
        for i in range(alex_model.njnt):
            name = mujoco.mj_id2name(alex_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            assert alex_model.dof_armature[i] > 0, f"Joint {name} has zero armature"

    def test_solver_iterations(self, alex_model) -> None:
        assert alex_model.opt.iterations >= 50

    def test_integrator_implicitfast(self, alex_model) -> None:
        assert alex_model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)

    def test_energy_tracking_enabled(self, alex_model) -> None:
        assert alex_model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_ENERGY)

    def test_actuator_ctrlrange_within_joint_range(self, alex_model) -> None:
        for i in range(alex_model.nu):
            jnt_id = alex_model.actuator_trnid[i, 0]
            jnt_lo, jnt_hi = alex_model.jnt_range[jnt_id]
            ctrl_lo, ctrl_hi = alex_model.actuator_ctrlrange[i]
            assert ctrl_lo >= jnt_lo - 1e-6, f"Actuator {i} ctrl_lo below joint range"
            assert ctrl_hi <= jnt_hi + 1e-6, f"Actuator {i} ctrl_hi above joint range"

    def test_forcerange_matches_effort_limits(self, alex_model) -> None:
        from sim.control import JOINT_EFFORT_LIMITS

        for i in range(alex_model.nu):
            jnt_id = alex_model.actuator_trnid[i, 0]
            jnt_name = mujoco.mj_id2name(alex_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name in JOINT_EFFORT_LIMITS:
                expected = JOINT_EFFORT_LIMITS[jnt_name]
                actual_hi = alex_model.actuator_forcerange[i, 1]
                assert actual_hi == pytest.approx(
                    expected, rel=0.01
                ), f"Actuator for {jnt_name}: forcerange {actual_hi} != effort {expected}"


# ---------------------------------------------------------------------------
# G. Hold-pose tests (Phase 1.1.2)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexHoldPose:
    """Verify robot holds position under PD control without drift or jitter."""

    def test_hold_home_pose(self, alex_model) -> None:
        """Set ctrl=home (zeros), run 5s, check qpos stays near zero."""
        data = mujoco.MjData(alex_model)
        data.ctrl[:] = 0.0
        for _ in range(2500):  # 5s at 0.002s timestep
            mujoco.mj_step(alex_model, data)
        assert (
            np.max(np.abs(data.qpos)) < 0.1
        ), f"Drift from home: max|qpos| = {np.max(np.abs(data.qpos)):.4f}"

    def test_hold_home_no_jitter(self, alex_model) -> None:
        """After settling, velocity should be near zero."""
        data = mujoco.MjData(alex_model)
        data.ctrl[:] = 0.0
        # Settle for 4s
        for _ in range(2000):
            mujoco.mj_step(alex_model, data)
        # Record max velocity over the next 1s
        max_vel = 0.0
        for _ in range(500):
            mujoco.mj_step(alex_model, data)
            v = np.max(np.abs(data.qvel))
            if v > max_vel:
                max_vel = v
        assert max_vel < 0.05, f"Jitter detected: max|qvel| = {max_vel:.4f} rad/s"

    def test_hold_rest_pose(self, alex_model) -> None:
        """Hold the 'rest' keyframe (bent elbows) for 5s."""
        data = mujoco.MjData(alex_model)
        key_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_KEY, "rest")
        if key_id < 0:
            pytest.skip("No 'rest' keyframe defined")
        mujoco.mj_resetDataKeyframe(alex_model, data, key_id)
        target = data.qpos.copy()
        # ctrl has fewer entries than qpos (coupled EZGripper joints have no actuator)
        data.ctrl[:] = data.ctrl  # keep ctrl from keyframe
        for _ in range(2500):
            mujoco.mj_step(alex_model, data)
        drift = np.max(np.abs(data.qpos - target))
        assert drift < 0.15, f"Drift from rest pose: {drift:.4f} rad"


# ---------------------------------------------------------------------------
# H. Joint sweep tests (Phase 1.1.2)
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexJointSweep:
    """Sweep each joint through partial range; check stability."""

    def test_individual_joint_sweeps(self, alex_model) -> None:
        """Command each actuated joint to midrange, hold, check convergence."""
        # Build joint→actuator mapping (only sweep joints that have actuators)
        jnt_to_act: dict[int, int] = {}
        for a in range(alex_model.nu):
            jnt_id = alex_model.actuator_trnid[a, 0]
            jnt_to_act[int(jnt_id)] = a

        for j, act_idx in jnt_to_act.items():
            data = mujoco.MjData(alex_model)
            name = mujoco.mj_id2name(alex_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = alex_model.jnt_range[j]
            mid = (lo + hi) / 2.0

            data.ctrl[:] = 0.0
            data.ctrl[act_idx] = mid

            for _ in range(1500):  # 3s
                mujoco.mj_step(alex_model, data)

            assert np.all(np.isfinite(data.qpos)), f"NaN in qpos sweeping {name}"
            assert np.all(np.isfinite(data.qvel)), f"NaN in qvel sweeping {name}"
            assert abs(data.qpos[j] - mid) < 0.35, (
                f"Joint {name} did not reach target: "
                f"target={mid:.3f}, actual={data.qpos[j]:.3f}"
            )

    def test_simultaneous_sweep_no_explosion(self, alex_model) -> None:
        """Command all actuators to 30% range simultaneously."""
        data = mujoco.MjData(alex_model)
        for a in range(alex_model.nu):
            jnt_id = alex_model.actuator_trnid[a, 0]
            lo, hi = alex_model.jnt_range[jnt_id]
            data.ctrl[a] = lo + 0.3 * (hi - lo)

        for _ in range(2500):  # 5s
            mujoco.mj_step(alex_model, data)

        assert np.all(np.isfinite(data.qpos)), "NaN in simultaneous sweep"
        assert np.max(np.abs(data.qpos)) < 10.0, "Divergence in simultaneous sweep"

    def test_velocity_bounded_during_sweep(self, alex_model) -> None:
        """During a joint sweep, velocities should stay within bounds."""
        from sim.control import JOINT_VELOCITY_LIMITS

        data = mujoco.MjData(alex_model)
        data.ctrl[:] = 0.0
        idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_y")
        act_idx = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_y")
        lo, hi = alex_model.jnt_range[idx]
        data.ctrl[act_idx] = hi * 0.8

        max_vel = 0.0
        for _ in range(2500):
            mujoco.mj_step(alex_model, data)
            v = abs(data.qvel[idx])
            if v > max_vel:
                max_vel = v

        hw_limit = JOINT_VELOCITY_LIMITS["left_shoulder_y"]
        assert (
            max_vel < hw_limit * 1.5
        ), f"Velocity {max_vel:.2f} exceeds 1.5x hardware limit {hw_limit}"


# ---------------------------------------------------------------------------
# I. Kinematics validation tests (Phase 1.1.4)
# ---------------------------------------------------------------------------

# Left/right arm joint names for kinematics tests
_LEFT_ARM = [
    "left_shoulder_y",
    "left_shoulder_x",
    "left_shoulder_z",
    "left_elbow_y",
    "left_wrist_z",
    "left_wrist_x",
    "left_gripper_z",
]
_RIGHT_ARM = [
    "right_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_z",
    "right_elbow_y",
    "right_wrist_z",
    "right_wrist_x",
    "right_gripper_z",
]
# Joints negated in left→right mirror mapping (X-axis and Z-axis joints)
_MIRROR_NEGATE = {"shoulder_x", "shoulder_z", "wrist_x", "wrist_z", "gripper_z"}


def _site_pos(model, data, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()


def _site_quat(model, data, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return quat


@pytest.mark.mujoco
@pytest.mark.assets
class TestAlexKinematics:
    """Kinematics validation: symmetry, workspace, axis directions, continuity."""

    def test_fk_left_right_symmetry(self, alex_model) -> None:
        """Mirrored joint configs produce Y-mirrored EE positions.

        Only tests configs where the mirrored value fits within the right arm's
        range without clipping, for a clean symmetry comparison.
        """
        rng = np.random.default_rng(42)
        n_target = 50
        max_pos_err = 0.0
        tested = 0

        for _ in range(n_target * 3):
            if tested >= n_target:
                break
            data = mujoco.MjData(alex_model)
            clipped = False
            for lname, rname in zip(_LEFT_ARM, _RIGHT_ARM, strict=True):
                lid = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, lname)
                rid = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, rname)
                lo, hi = alex_model.jnt_range[lid]
                val = rng.uniform(lo, hi)
                data.qpos[lid] = val
                suffix = lname.split("_", 1)[1]
                rval = -val if suffix in _MIRROR_NEGATE else val
                rlo, rhi = alex_model.jnt_range[rid]
                if rval < rlo - 1e-6 or rval > rhi + 1e-6:
                    clipped = True
                    break
                data.qpos[rid] = rval
            if clipped:
                continue

            mujoco.mj_forward(alex_model, data)
            l_pos = _site_pos(alex_model, data, "left_ee_site")
            r_pos = _site_pos(alex_model, data, "right_ee_site")

            expected_r = np.array([l_pos[0], -l_pos[1], l_pos[2]])
            err = np.linalg.norm(r_pos - expected_r)
            if err > max_pos_err:
                max_pos_err = err
            tested += 1

        assert tested >= 20, f"Only {tested} clean mirror configs found"
        assert (
            max_pos_err < 0.005
        ), f"FK symmetry error: max={max_pos_err * 100:.3f} cm"  # 5mm tolerance

    def test_workspace_reaches_table(self, alex_model) -> None:
        """EE can reach typical LEGO workspace region."""
        rng = np.random.default_rng(123)
        n_samples = 200
        all_joints = _LEFT_ARM + _RIGHT_ARM
        jids = [mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in all_joints]

        positions = {"left": [], "right": []}
        for _ in range(n_samples):
            data = mujoco.MjData(alex_model)
            for jid in jids:
                lo, hi = alex_model.jnt_range[jid]
                data.qpos[jid] = rng.uniform(lo, hi)
            mujoco.mj_forward(alex_model, data)
            positions["left"].append(_site_pos(alex_model, data, "left_ee_site"))
            positions["right"].append(_site_pos(alex_model, data, "right_ee_site"))

        for side in ["left", "right"]:
            arr = np.array(positions[side])
            # Must be able to reach forward (X > 0.3m) and LEGO height zone
            assert arr[:, 0].max() > 0.3, f"{side} arm can't reach 0.3m forward"
            assert arr[:, 2].min() < 1.2, f"{side} arm can't reach below 1.2m"
            assert arr[:, 2].max() > 0.8, f"{side} arm can't reach above 0.8m"

    def test_joint_axis_directions(self, alex_model) -> None:
        """Perturbing each joint affects the EE (position or orientation).

        Uses a general base config (25% of range) to avoid degenerate cases.
        Checks both position and orientation since distal Z-axis joints may
        only rotate the EE without translating it.
        """
        delta = 0.1
        # Build general base config
        base_qpos = np.zeros(alex_model.nq)
        for jname in _LEFT_ARM + _RIGHT_ARM:
            jid = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            lo, hi = alex_model.jnt_range[jid]
            base_qpos[jid] = lo + 0.25 * (hi - lo)

        for joints, ee_site in [(_LEFT_ARM, "left_ee_site"), (_RIGHT_ARM, "right_ee_site")]:
            data_base = mujoco.MjData(alex_model)
            data_base.qpos[:] = base_qpos.copy()
            mujoco.mj_forward(alex_model, data_base)
            base_pos = _site_pos(alex_model, data_base, ee_site)
            base_quat = _site_quat(alex_model, data_base, ee_site)

            for jname in joints:
                jid = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                lo, hi = alex_model.jnt_range[jid]
                data = mujoco.MjData(alex_model)
                data.qpos[:] = base_qpos.copy()
                new_val = data.qpos[jid] + delta
                if new_val > hi:
                    new_val = data.qpos[jid] - delta
                data.qpos[jid] = np.clip(new_val, lo, hi)
                mujoco.mj_forward(alex_model, data)
                pert_pos = _site_pos(alex_model, data, ee_site)
                pert_quat = _site_quat(alex_model, data, ee_site)
                pos_dist = float(np.linalg.norm(pert_pos - base_pos))
                # Orientation change via quaternion dot product
                dot = float(np.clip(np.abs(np.dot(base_quat, pert_quat)), 0.0, 1.0))
                orient_diff = float(np.degrees(2.0 * np.arccos(dot)))
                has_effect = pos_dist > 1e-5 or orient_diff > 0.01
                assert has_effect, (
                    f"Joint {jname} perturbation had no effect on {ee_site} "
                    f"(pos_dist={pos_dist:.6f}m, orient_diff={orient_diff:.4f}°)"
                )

    def test_ee_position_continuity(self, alex_model) -> None:
        """Smooth joint trajectory produces smooth EE trajectory (no jumps)."""
        rng = np.random.default_rng(99)
        all_joints = _LEFT_ARM + _RIGHT_ARM
        jids = [mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in all_joints]

        q_start = np.zeros(alex_model.nq)
        q_end = np.zeros(alex_model.nq)
        for jid in jids:
            lo, hi = alex_model.jnt_range[jid]
            q_start[jid] = rng.uniform(lo, hi)
            q_end[jid] = rng.uniform(lo, hi)

        n_steps = 50
        prev: dict[str, np.ndarray] = {}
        max_jump = 0.0
        for step in range(n_steps + 1):
            alpha = step / n_steps
            data = mujoco.MjData(alex_model)
            data.qpos[:] = q_start * (1 - alpha) + q_end * alpha
            mujoco.mj_forward(alex_model, data)
            for site_name in ["left_ee_site", "right_ee_site"]:
                pos = _site_pos(alex_model, data, site_name)
                if site_name in prev:
                    jump = float(np.linalg.norm(pos - prev[site_name]))
                    max_jump = max(max_jump, jump)
                prev[site_name] = pos

        assert max_jump < 0.05, f"EE jump of {max_jump * 100:.2f} cm detected"

    def test_ee_orientation_valid(self, alex_model) -> None:
        """EE quaternions are unit quaternions at random configs."""
        rng = np.random.default_rng(77)
        all_joints = _LEFT_ARM + _RIGHT_ARM
        jids = [mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in all_joints]

        for _ in range(50):
            data = mujoco.MjData(alex_model)
            for jid in jids:
                lo, hi = alex_model.jnt_range[jid]
                data.qpos[jid] = rng.uniform(lo, hi)
            mujoco.mj_forward(alex_model, data)
            for site_name in ["left_ee_site", "right_ee_site"]:
                quat = _site_quat(alex_model, data, site_name)
                assert (
                    abs(np.linalg.norm(quat) - 1.0) < 1e-6
                ), f"{site_name} quaternion not unit: norm={np.linalg.norm(quat)}"

    def test_home_ee_positions_reasonable(self, alex_model) -> None:
        """At home pose, EE sites are symmetric and at expected positions."""
        data = mujoco.MjData(alex_model)
        mujoco.mj_forward(alex_model, data)

        l_pos = _site_pos(alex_model, data, "left_ee_site")
        r_pos = _site_pos(alex_model, data, "right_ee_site")

        # Y symmetry: left Y ≈ -right Y
        assert (
            abs(l_pos[1] + r_pos[1]) < 0.01
        ), f"Home EE not Y-symmetric: left_y={l_pos[1]:.4f}, right_y={r_pos[1]:.4f}"
        # X and Z should match
        assert abs(l_pos[0] - r_pos[0]) < 0.01, "Home EE X mismatch"
        assert abs(l_pos[2] - r_pos[2]) < 0.01, "Home EE Z mismatch"

        # Sanity: EE should be near robot (within 1m of base at z=1.0)
        for pos, name in [(l_pos, "left"), (r_pos, "right")]:
            dist_from_base = np.linalg.norm(pos - np.array([0, 0, 1.0]))
            assert dist_from_base < 1.5, f"{name} EE too far from base: {dist_from_base:.2f} m"
