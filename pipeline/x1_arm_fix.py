"""Production arm-retarget fix for the X1 (the phase-2 'split' winner).

Why (measured on 07_01/82_13/09_01/09_02/137_01, see tools/x1_arm_study.py and
docs/X1_PHASE2_REPORT.md):
  * GMR's arm targets are pelvis-relative scaled human data. The X1's torso is far
    shorter than 0.93x a human's, so the shoulder/elbow/wrist targets float
    0.25-0.45 m above where the robot's arm can be; the stock solve gives up on
    them (wrist FK-vs-target error p50 0.29-0.44 m) and parks the elbows on the
    +2.0 rad stop (library-wide 67.2% of frames had a joint within 0.02 rad of a
    limit, left elbow 57.8%).
  * X1 arm geometry: the elbow joint sits ~at the shoulder (|shoulder_yaw_link ->
    elbow_pitch_link| = 0.048 m) and the forearm is 0.260 m, so the wrist distance
    from the shoulder is confined to [0.244 (elbow 2.0), 0.308 (elbow 0)] m. Any
    wrist target outside that band is unreachable.

Fix: keep GMR's whole-body solve for base/legs/torso (with softened wrist/elbow
weights, baked into motions/x1_ik_config.json), then re-solve ONLY the arms in a
second pass:
  - freeze base/legs/torso with a strong posture cost toward GMR's solution;
  - single arm task: wrist position, anchored on the ROBOT's live shoulder FK,
    direction = human shoulder->wrist vector, distance = clip(0.62 * human
    |wrist-shoulder|, 0.268, 0.300) -- reachable by construction;
  - weak posture prior on the arm joints (elbow 0.5, shoulder roll -0.2) resolves
    the shoulder-yaw/elbow-yaw redundancy;
  - hard ConfigurationLimit with 0.15 rad margin (warm starts are clamped INTO the
    margin band: mink's limit only prevents entering, it never extracts a joint
    already parked on the stop) + arm VelocityLimit;
  - per-frame convergence loop with temporal warm start, then 5-tap binomial
    smoothing of the arm track.

Result on the study set: elbow-pitch at-limit 0% (stock 95-100%), any-arm at-limit
0% (stock 100%), wrist direction error 22-45 deg (stock 78-120), elbow p50
0.52-0.77 rad, jerk at/below stock after 50 Hz resampling.
"""
from __future__ import annotations

import numpy as np

ARM_JOINTS = (
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_elbow_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint',
)
RATIO = 0.62          # robot arm reach 0.293 / human walking |wrist-shoulder| ~0.47
D_MIN, D_MAX = 0.268, 0.300  # wrist distance band inside the reachable [0.244, 0.308]
MARGIN = 0.15         # rad, joint-limit avoidance band
POSTURE_COST_ARM = 0.30
FREEZE_COST = 20.0
WRIST_POS_COST = 8.0
VEL_LIMIT = 30.0      # rad/s, arm dofs
WARMUP_ITER = 300     # first frame converges from the stock pose
MAX_ITER = 60
DT = 0.01
TOL = 1e-4
SMOOTH_TAPS = 5


def make_retargeter(base_retargeter_cls):
    """Wrap GMR's class: robot-anchored wrist targets (both IK tables)."""
    import mink
    import mujoco

    class AnchoredRetarget(base_retargeter_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.shoulder_bid = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                      f'{s}_shoulder_yaw_link') for s in ('left', 'right')}

        def update_targets(self, human_data, offset_to_ground=False):
            super().update_targets(human_data, offset_to_ground)
            for side in ('left', 'right'):
                wr_h = np.asarray(human_data[f'{side}_wrist'][0], dtype=float)
                sh_h = np.asarray(human_data[f'{side}_shoulder'][0], dtype=float)
                sh_anchor = None
                if getattr(self, 'configuration', None) is not None:
                    mujoco.mj_kinematics(self.model, self.configuration.data)
                    sh_anchor = self.configuration.data.xpos[self.shoulder_bid[side]].copy()
                if sh_anchor is None:
                    sh_anchor = np.asarray(self.scaled_human_data[f'{side}_shoulder'][0], dtype=float)
                v = wr_h - sh_h
                n = np.linalg.norm(v)
                if n < 1e-6:
                    continue
                d = float(np.clip(RATIO * n, D_MIN, D_MAX))
                pos = sh_anchor + v * (d / n)
                quat = self.scaled_human_data[f'{side}_wrist'][1]  # wxyz, as GMR passes to mink
                target = mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos)
                for table in (self.human_body_to_task1, self.human_body_to_task2):
                    task = table.get(f'{side}_wrist')
                    if task is not None:
                        task.set_target(target)

    return AnchoredRetarget


def arm_refine(r, qpos_seq: np.ndarray, frames: list,
               margin: float = MARGIN, posture_cost_arm: float = POSTURE_COST_ARM,
               freeze_cost: float = FREEZE_COST, wrist_pos_cost: float = WRIST_POS_COST,
               vel_limit: float = VEL_LIMIT, warmup_iter: int = WARMUP_ITER,
               max_iter: int = MAX_ITER, dt: float = DT, tol: float = TOL,
               smooth_taps: int = SMOOTH_TAPS) -> np.ndarray:
    """Second pass over a GMR-solved clip: re-solve the arms only. (N, 36) -> (N, 36)."""
    import mink
    import mujoco

    m = r.model
    dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in range(m.nv)]
    arm_dofs = np.array([i for i, n in enumerate(dof_names) if n in ARM_JOINTS])
    arm_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS])
    natural = np.zeros(m.nq)
    for adr, nm in zip(arm_qadr, ARM_JOINTS):
        if 'elbow_pitch' in nm:
            natural[adr] = 0.5
        elif 'shoulder_roll' in nm:
            natural[adr] = -0.2

    cost = np.full(m.nv, freeze_cost)
    cost[arm_dofs] = posture_cost_arm
    cost[:6] = freeze_cost  # free joint
    tasks_by = {(s, 'wr'): mink.FrameTask(f'{s}_wrist_roll_link', 'body', wrist_pos_cost, 0)
                for s in ('left', 'right')}
    posture = mink.PostureTask(m, cost)
    limits = [mink.ConfigurationLimit(m, min_distance_from_limits=margin),
              mink.VelocityLimit(m, {n: vel_limit for n in ARM_JOINTS})]
    all_tasks = list(tasks_by.values()) + [posture]

    def set_targets(frame):
        r.update_targets(frame)  # sets GMR's own table tasks to the anchored targets
        import numpy as _np
        for side in ('left', 'right'):
            # EXPLICIT anchored intent (same law as AnchoredRetarget.update_targets):
            # sh_fk + unit(human wr-sh) * clip(RATIO*|wr-sh|, D_MIN, D_MAX). Reading
            # r.scaled_human_data instead would silently re-reference the STOCK
            # pelvis-relative target (measured: |FK-stock| 6-9 mm vs |FK-anchored|
            # 11-19 cm -- the refine used to converge to the stock target and only
            # looked right because the anchored base solve had moved the shoulder).
            sh_fk = cfg.data.xpos[r.shoulder_bid[side]]
            wr_h = _np.asarray(frame[f'{side}_wrist'][0], dtype=float)
            sh_h = _np.asarray(frame[f'{side}_shoulder'][0], dtype=float)
            v = wr_h - sh_h
            n = _np.linalg.norm(v)
            quat = r.scaled_human_data[f'{side}_wrist'][1]  # wxyz orientation target (unused, rot cost 0)
            if n < 1e-6:
                pos = sh_fk
            else:
                pos = sh_fk + v * (float(_np.clip(RATIO * n, D_MIN, D_MAX)) / n)
            tasks_by[(side, 'wr')].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))

    cfg = mink.Configuration(m)
    out = qpos_seq.copy()
    prev_arm = None
    # clamp warm starts INTO the margin band: mink's ConfigurationLimit only
    # prevents entering the band; a joint already parked on the stop stays there.
    band_lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] + margin
                        for n in ARM_JOINTS])
    band_hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] - margin
                        for n in ARM_JOINTS])
    for k, frame in enumerate(frames[: len(qpos_seq)]):
        q = out[k].copy()
        if prev_arm is not None:
            q[arm_qadr] = prev_arm
        q[arm_qadr] = np.clip(q[arm_qadr], band_lo, band_hi)
        cfg.update(q)
        posture_target = q.copy()
        posture_target[arm_qadr] = natural[arm_qadr]
        posture.set_target(posture_target)
        set_targets(frame)
        n_iter = warmup_iter if k == 0 else max_iter
        prev_err = np.inf
        for _ in range(n_iter):
            # NB: `limits` MUST be passed as a keyword argument -- the 6th positional
            # parameter of mink.solve_ik is `safety_break`, and passing the limits
            # list positionally silently disables every limit.
            v = mink.solve_ik(cfg, all_tasks, dt, r.solver, r.damping, limits=limits)
            cfg.integrate_inplace(v, dt)
            err = float(np.linalg.norm(np.concatenate(
                [t.compute_error(cfg) for t in tasks_by.values()])))
            if k > 0 and prev_err - err < tol:
                break
            prev_err = err
        out[k] = cfg.data.qpos.copy()
        prev_arm = out[k][arm_qadr].copy()
    if smooth_taps > 1:
        k = np.array([1, 4, 6, 4, 1], dtype=float)[: smooth_taps if smooth_taps >= 5 else smooth_taps + 2]
        k /= k.sum()
        pad = len(k) // 2
        for c in arm_qadr:
            padded = np.pad(out[:, c], pad, mode='edge')
            out[:, c] = np.convolve(padded, k, mode='valid')
    return out
