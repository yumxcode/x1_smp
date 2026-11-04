"""Foot-retarget fix for the X1 (second-round data unlock, exp_78e7c4ac step 2).

Root cause being fixed (fnd_95cd3382): GMR's whole-body solve drops the swing
foot on brisk human walks (cad >= 115 spm) -- one foot drags with swing 0.12-0.2
s / duty 0.85+, so no >=0.85 m/s healthy clip survives the gait audit. Same
failure shape (and same cure) as the phase-2 arm fix: the stock targets are
pelvis-relative scaled human data that the robot's short legs cannot reach, so
the solver gives up on the swing foot.

Fix (tools/x1_arm_fix.py pattern): second IK pass re-solving ONLY the legs,
with the ankle target anchored on the ROBOT's live hip FK:
  target = hip_fk + unit(human foot-hip) * clip(0.675 * |human foot-hip|, 0.30, 0.545)
X1 leg |hip->ankle| = 0.574 m at q=0; human walking |foot-hip| ~0.55-0.86 m, so
0.675 * human spans 0.37-0.58 -> clipped into the reachable [0.30, 0.545] band
(3 cm short of full extension to avoid the straight-leg singularity). The
stance foot is stable in this law by construction (the human stance foot is
fixed while the hip travels over it), and the swing foot recovers the human
swing trajectory -- exactly the signal GMR loses.
"""

from __future__ import annotations

import numpy as np

LEG_JOINTS = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
)
RATIO_LEG = 0.62         # X1 |hip->ankle| 0.574 / human walking |foot-hip| ~0.85
D_MIN, D_MAX = 0.30, 0.52  # inside reachable [~0.28, 0.574], 3 cm short of lockout
MARGIN = 0.10
POSTURE_COST_LEG = 0.30
FREEZE_COST = 20.0
ANKLE_POS_COST = 8.0
VEL_LIMIT = 25.0          # rad/s, leg dofs
WARMUP_ITER = 300
MAX_ITER = 60
DT = 0.01
TOL = 1e-4
SMOOTH_TAPS = 5


def foot_refine(r, qpos_seq: np.ndarray, frames: list,
                margin: float = MARGIN, posture_cost_leg: float = POSTURE_COST_LEG,
                freeze_cost: float = FREEZE_COST, ankle_pos_cost: float = ANKLE_POS_COST,
                vel_limit: float = VEL_LIMIT, warmup_iter: int = WARMUP_ITER,
                max_iter: int = MAX_ITER, dt: float = DT, tol: float = TOL,
                smooth_taps: int = SMOOTH_TAPS) -> np.ndarray:
    """Second pass over a GMR+arm-refined clip: re-solve the legs only. (N, 36) -> (N, 36)."""
    import mink
    import mujoco

    m = r.model
    dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[i])) for i in range(m.nv)]
    leg_dofs = np.array([i for i, n in enumerate(dof_names) if n in LEG_JOINTS])
    leg_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEG_JOINTS])
    hip_bid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{s}_hip_pitch_link')
               for s in ('left', 'right')}
    natural = np.zeros(m.nq)
    for adr, nm in zip(leg_qadr, LEG_JOINTS):
        if 'knee_pitch' in nm:
            natural[adr] = 0.4          # keep the knee clear of the straight-leg stop
        elif 'ankle_pitch' in nm:
            natural[adr] = -0.15

    cost = np.full(m.nv, freeze_cost)
    cost[leg_dofs] = posture_cost_leg
    cost[:6] = freeze_cost  # free joint: root trajectory is preserved as solved
    tasks_by = {(s, 'ank'): mink.FrameTask(f'{s}_ankle_roll_link', 'body', ankle_pos_cost, 0)
                for s in ('left', 'right')}
    posture = mink.PostureTask(m, cost)
    limits = [mink.ConfigurationLimit(m, min_distance_from_limits=margin),
              mink.VelocityLimit(m, {n: vel_limit for n in LEG_JOINTS})]
    all_tasks = list(tasks_by.values()) + [posture]

    def set_targets(cfg, frame, foot_z_rel):
        """V3: swing-selective -- only the foot the HUMAN is swinging gets the
        anchored target; the stance foot's task target is pinned to its own
        current FK (hold), so the GMR stance solution is not disturbed."""
        ank_bid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{s}_ankle_roll_link')
                   for s in ('left', 'right')}
        for side in ('left', 'right'):
            ank_fk = cfg.data.xpos[ank_bid[side]]
            if foot_z_rel is not None and foot_z_rel.get(side, 0.0) < 0.02:
                pos = ank_fk  # human stance -> hold the robot foot where it is
            else:
                hip_fk = cfg.data.xpos[hip_bid[side]]
                ft_h = np.asarray(frame[f'{side}_foot'][0], dtype=float)
                hp_h = np.asarray(frame[f'{side}_hip'][0], dtype=float)
                v = ft_h - hp_h
                n = np.linalg.norm(v)
                if n < 1e-6:
                    pos = hip_fk
                else:
                    pos = hip_fk + v * (float(np.clip(RATIO_LEG * n, D_MIN, D_MAX)) / n)
            quat = r.scaled_human_data.get(f'{side}_ankle', r.scaled_human_data[f'{side}_foot'])[1]
            tasks_by[(side, 'ank')].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))

    cfg = mink.Configuration(m)
    out = qpos_seq.copy()
    prev_leg = None
    band_lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] + margin
                        for n in LEG_JOINTS])
    band_hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] - margin
                        for n in LEG_JOINTS])
    # per-frame human swing state: foot z relative to the lower of the two feet
    ftz = {s: np.array([np.asarray(f[f'{s}_foot'][0], dtype=float)[2] for f in frames[: len(qpos_seq)]])
           for s in ('left', 'right')}
    lower = np.minimum(ftz['left'], ftz['right'])
    for k, frame in enumerate(frames[: len(qpos_seq)]):
        swing_state = {s: float(ftz[s][k] - lower[k]) for s in ('left', 'right')}
        q = out[k].copy()
        if prev_leg is not None:
            q[leg_qadr] = prev_leg
        q[leg_qadr] = np.clip(q[leg_qadr], band_lo, band_hi)
        cfg.update(q)
        posture_target = q.copy()
        posture_target[leg_qadr] = natural[leg_qadr]
        posture.set_target(posture_target)
        set_targets(cfg, frame, swing_state)
        n_iter = warmup_iter if k == 0 else max_iter
        prev_err = np.inf
        for _ in range(n_iter):
            v = mink.solve_ik(cfg, all_tasks, dt, r.solver, r.damping, limits=limits)
            cfg.integrate_inplace(v, dt)
            err = float(np.linalg.norm(np.concatenate(
                [t.compute_error(cfg) for t in tasks_by.values()])))
            if k > 0 and prev_err - err < tol:
                break
            prev_err = err
        out[k] = cfg.data.qpos.copy()
        prev_leg = out[k][leg_qadr].copy()
    if smooth_taps > 1:
        k = np.array([1, 4, 6, 4, 1], dtype=float)[: smooth_taps if smooth_taps >= 5 else smooth_taps + 2]
        k /= k.sum()
        pad = len(k) // 2
        for c in leg_qadr:
            padded = np.pad(out[:, c], pad, mode='edge')
            out[:, c] = np.convolve(padded, k, mode='valid')
    return out
