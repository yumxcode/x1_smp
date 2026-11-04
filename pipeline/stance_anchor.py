"""Stance-aware anchoring for the X1 SMP library rebuild (AUDIT_FAILURE §6-1/2).

Root cause being fixed (fnd_0b28742e, R-A): the R1-R3 library has NO planted
stance -- GMR's foot IK does not anchor the stance foot's world position, and
per-frame grounding presses the lowest (moving) foot onto the floor, so every
"contact" frame slides at 0.13-0.69 m/s and flight frames cannot exist.

This module provides the three building blocks of the rebuild:

  1. Human contact labels: Schmitt trigger on the SMPLX foot-joint height
     relative to its own per-clip stance baseline (the human source has real
     contact timing; the retarget preserves cadence, so human timing transfers).
  2. anchor_ik: second IK pass over a GMR+arm-refined clip (src fps) that PINS
     the stance foot's full SE3 at a FIXED world pose for the whole stance phase
     -- the anchor is the foot's own touchdown pose, with z adjusted so the sole
     sits on a common per-clip ground plane. Root / arms / torso are frozen
     (root trajectory -- hence speed, cadence and step length -- is preserved);
     the swing leg follows the GMR solution via a posture task. This is what
     R2's V2/V2b/V3 never tried: V3 "held" the stance foot at its own CURRENT
     FK each frame (which slides with the root); here the target is constant in
     the world, forcing the real hip-over-foot rotation.
  3. root_warp: fallback anchoring with NO IK -- rigid per-frame XY translation
     of the whole pose by minus the stance foot's slide (geometrically exact:
     translating the root translates the foot). Joints are untouched; the cost
     is a speed reduction by the slip amount (needs faster human windows).

Stance-aware grounding: with anchor_ik the stance soles are on the common plane
z=0 by construction, double support has both feet on it, and flight frames are
left alone; only a final uniform per-clip shift (min sole -> 0) is applied.
"""
from __future__ import annotations

import numpy as np

LEG_JOINTS = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
)
SIDES = ('left', 'right')

# ---- IK-pin parameters (x1_foot_fix.py lineage, re-tuned for stance pinning) ----
PIN_COST = 8.0            # stance-foot frame task (pos+ori) cost while pinned
PIN_COST_OFF = 1e-6       # swing foot: effectively free
POSTURE_COST_LEG = 0.30   # pull legs toward the GMR solution (swing tracks GMR)
FREEZE_COST = 20.0        # root (6) + arms + torso stay at the GMR solution
MARGIN = 0.10             # joint-limit margin (rad)
VEL_LIMIT = 25.0          # rad/s on leg dofs
WARMUP_ITER = 300
MAX_ITER = 60
DT = 0.01
TOL = 1e-4
SMOOTH_TAPS = 5

# ---- human contact labeling (heights in METERS, human scale) ----
H_GROUND = 0.030          # below: contact candidate (rel. to per-clip p5 baseline)
H_AIR = 0.080             # above: airborne
H_MIN_PHASE_S = 0.08      # flicker suppression


def schmitt_contact(z_rel: np.ndarray, fps: float,
                    ground: float = H_GROUND, air: float = H_AIR,
                    min_phase_s: float = H_MIN_PHASE_S) -> np.ndarray:
    """Schmitt trigger on a (already baseline-subtracted) height series.

    state=contact initially iff z_rel[0] < ground; leaves contact at z>air,
    re-enters at z<ground; phases shorter than min_phase frames are merged
    away. Same semantics as audit/fk_gait_stats.segment_phases."""
    n = len(z_rel)
    min_ph = max(2, int(round(min_phase_s * fps)))
    contact = np.zeros(n, dtype=bool)
    state = z_rel[0] < ground
    start = 0
    phases: list[tuple[bool, int, int]] = []
    for i in range(1, n):
        if state and z_rel[i] > air:
            phases.append((True, start, i)); state, start = False, i
        elif not state and z_rel[i] < ground:
            phases.append((False, start, i)); state, start = True, i
    phases.append((state, start, n))
    out = []
    for kind, i0, i1 in phases:
        if out and (i1 - i0) < min_ph:
            k0, j0, j1 = out[-1]
            out[-1] = (k0, j0, i1)
        else:
            out.append((kind, i0, i1))
    for kind, i0, i1 in out:
        if kind:
            contact[i0:i1] = True
    return contact


def human_contact(frames: list, n: int, fps: float) -> tuple[dict, dict]:
    """Per-foot contact labels from SMPLX foot-joint heights.

    Baseline per foot = 5th percentile of the joint height over the clip (its
    stance height); z_rel = h - baseline. Returns (contact{left,right} bool
    arrays, stats dict with duty / double-stance fraction / n phases)."""
    contact, stats = {}, {}
    z = {}
    for side in SIDES:
        h = np.array([float(np.asarray(f[f'{side}_foot'][0])[2]) for f in frames[:n]])
        base = float(np.percentile(h, 5))
        z[side] = h
        contact[side] = schmitt_contact(h - base, fps)
    both = contact['left'] & contact['right']
    for side in SIDES:
        c = contact[side]
        runs = runs_of(c)
        stats[side] = dict(
            duty=float(c.mean()), n_phases=len(runs),
            stance_s_med=float(np.median([i1 - i0 for i0, i1 in runs]) / fps) if runs else float('nan'),
        )
    stats['double_stance_frac'] = float(both.mean())
    stats['flight_frac'] = float((~contact['left'] & ~contact['right']).mean())
    return contact, stats


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [i0, i1) runs of True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def resample_series(x: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Linear resample on the exact time grid of prepare_motions.resample
    (phase over (n-1), endpoint-inclusive) so labels align with the 50 Hz qpos."""
    n = len(x)
    duration = (n - 1) / fps_in
    times = np.arange(0, duration, 1.0 / fps_out)
    pos = times / duration * (n - 1)
    return np.interp(pos, np.arange(n), x)


def resample_contact(contact: np.ndarray, fps_in: float, fps_out: float,
                     thresh: float = 0.5) -> np.ndarray:
    return resample_series(contact.astype(np.float64), fps_in, fps_out) >= thresh


def _foot_geom_ids(model, data, side: str) -> tuple[int, list[int]]:
    """Sole corner-sphere geom ids under the ankle body. The vendored X1 XML
    names them '*_collision'; the GMR retarget model leaves them unnamed, so
    fall back to the geometric signature (4 spheres of radius 2 mm at the
    sole corners, distinct from the 20 mm visual spheres)."""
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_ankle_roll_link')
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid
            and 'collision' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
    if len(gids) != 4:
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                and model.geom_size[g][0] < 0.005]
    assert len(gids) == 4, f'expected 4 sole corners on {side}, got {len(gids)}'
    return bid, gids


def sole_corners(model, data, qpos: np.ndarray) -> dict:
    """(T,4,3) world corners per foot via MuJoCo FK on `model`."""
    import mujoco
    T = len(qpos)
    out = {}
    gmap = {s: _foot_geom_ids(model, data, s)[1] for s in SIDES}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        for side in SIDES:
            out.setdefault(side, np.empty((T, 4, 3)))[t] = [data.geom_xpos[g] for g in gmap[side]]
    return out


def anchor_ik(r, qpos_seq: np.ndarray, contact: dict,
              pin_cost: float = PIN_COST, posture_cost_leg: float = POSTURE_COST_LEG,
              freeze_cost: float = FREEZE_COST, margin: float = MARGIN,
              vel_limit: float = VEL_LIMIT, warmup_iter: int = WARMUP_ITER,
              max_iter: int = MAX_ITER, dt: float = DT, tol: float = TOL,
              smooth_taps: int = SMOOTH_TAPS) -> tuple[np.ndarray, dict]:
    """Second IK pass pinning each stance foot at a FIXED world SE3.

    Runs at the clip's source fps over the GMR+arm-refined solution. The root
    (free joint) and all non-leg dofs are frozen at the GMR solution, so the
    root trajectory -- speed, cadence, step lengths -- is preserved. Returns
    (anchored qpos_seq, diagnostics)."""
    import mink
    import mujoco

    m = r.model
    dof_joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.dof_jntid[i]))
                       for i in range(m.nv)]
    leg_dofs = np.array([i for i, n in enumerate(dof_joint_names) if n in LEG_JOINTS])
    leg_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                         for n in LEG_JOINTS])

    data = mujoco.MjData(m)
    corners = sole_corners(m, data, qpos_seq)          # (per side) (T,4,3)
    sole_z = {s: corners[s][:, :, 2].min(axis=1) for s in SIDES}

    # anchors: one constructed HEEL-TOE ROLL arc per stance run.
    #
    # Measured geometry (43_01, src fps): during a labelled stance run the GMR
    # hip already sweeps 0.49-0.53 m past the ankle (hip-ankle horizontal
    # 0.01 -> 0.27 m -- feasible), and the ankle itself advances 0.14-0.22 m
    # ~= the scaled human heel-toe roll, but WITHOUT pitching -- the whole
    # foot translates near the ground, so every sole corner moves ~0.29 m/s
    # (the audit's slip). Pinning the ankle to a FIXED point is infeasible
    # (needs +-0.11 m of extra sweep on top of the already-marginal 0.27).
    # The feasible-and-honest fix is a real roll:
    #   phase A (u<0.30): pivot about the HEEL corner pair, pitch -> 0;
    #   phase B (0.30<=u<0.60): FLAT, ankle pinned constant;
    #   phase C (u>=0.60): pivot about the TOE corner pair, pitch -> up.
    # The pivot corner is stationary at every instant, which is exactly what
    # the conservative min-corner-slip metric measures as "planted".
    ank_bid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{s}_ankle_roll_link')
               for s in SIDES}
    hip_bid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{s}_hip_pitch_link')
               for s in SIDES}
    runs = {s: runs_of(contact[s]) for s in SIDES}

    def rot_axis(axis: np.ndarray, th: float) -> np.ndarray:
        a = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    def quat_mul_wxyz(q1, q2):
        w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                         w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2,
                         w1*z2 + x1*y2 - y1*x2 + z1*w2])

    def mat_to_quat_wxyz(Rm):
        q = np.empty(4)
        t = Rm.trace()
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[0] = 0.25 * s; q[1] = (Rm[2,1]-Rm[1,2])/s; q[2] = (Rm[0,2]-Rm[2,0])/s; q[3] = (Rm[1,0]-Rm[0,1])/s
        else:
            i = int(np.argmax(np.diag(Rm)))
            if i == 0:
                s = np.sqrt(1+Rm[0,0]-Rm[1,1]-Rm[2,2])*2
                q = np.array([(Rm[2,1]-Rm[1,2])/s, 0.25*s, (Rm[0,1]+Rm[1,0])/s, (Rm[0,2]+Rm[2,0])/s])
            elif i == 1:
                s = np.sqrt(1+Rm[1,1]-Rm[0,0]-Rm[2,2])*2
                q = np.array([(Rm[0,2]-Rm[2,0])/s, (Rm[0,1]+Rm[1,0])/s, 0.25*s, (Rm[1,2]+Rm[2,1])/s])
            else:
                s = np.sqrt(1+Rm[2,2]-Rm[0,0]-Rm[1,1])*2
                q = np.array([(Rm[1,0]-Rm[0,1])/s, (Rm[0,2]+Rm[2,0])/s, (Rm[1,2]+Rm[2,1])/s, 0.25*s])
        return q / np.linalg.norm(q)

    roll_stats = []
    anchors: dict[tuple[str, int], list] = {}   # (side, i0) -> [(pos, quat)] per frame
    for side in SIDES:
        for i0, i1 in runs[side]:
            L = i1 - i0
            # hip travel direction over the run
            hips = np.empty((L, 3))
            for kk, t in enumerate(range(i0, i1)):
                data.qpos[:] = qpos_seq[t]; mujoco.mj_kinematics(m, data)
                hips[kk] = data.xpos[hip_bid[side]]
            dv = hips[-1] - hips[0]
            dv[2] = 0.0
            dirv = dv / (np.linalg.norm(dv) + 1e-9)
            axis = np.array([-dirv[1], dirv[0], 0.0])   # left of travel
            # flat reference = the run's most-flat, lowest-sole GMR frame
            sole_run = corners[side][i0:i1, :, 2].min(axis=1)
            tf = i0 + int(np.argmin(sole_run))
            data.qpos[:] = qpos_seq[tf]; mujoco.mj_kinematics(m, data)
            quat0 = data.xquat[ank_bid[side]].copy().astype(np.float64)
            flat_c = corners[side][tf].copy()                     # (4,3) world
            proj = flat_c[:, :2] @ dirv
            toe_i, heel_i = int(np.argmax(proj)), int(np.argmin(proj))
            hf = 0.5 * float(proj.max() - proj.min())             # foot half length
            ank_z_flat = float(data.xpos[ank_bid[side]][2]) - float(sole_run.min())
            posB = np.median(corners[side][i0:i1, :, :2].mean(axis=1), axis=0)
            posB = np.array([posB[0], posB[1], ank_z_flat])       # flat ankle pose
            pv_h = np.array([*(flat_c[heel_i, :2]), 0.0])          # heel pivot (on ground)
            pv_t = np.array([*(flat_c[toe_i, :2]), 0.0])           # toe pivot
            # roll amplitudes from the GMR ankle-z arc at the run ends
            z_start = float(np.median(corners[side][i0:i0+3, :, 2].min(axis=1)))
            z_end = float(np.median(corners[side][i1-3:i1, :, 2].min(axis=1)))
            thA0 = float(np.clip(np.arcsin(np.clip((z_start - sole_run.min()) / max(hf, 1e-3), -1, 1)), 0.05, 0.45))
            thC0 = float(np.clip(np.arcsin(np.clip((z_end - sole_run.min()) / max(hf, 1e-3), -1, 1)), 0.05, 0.45))
            arc = []
            for kk in range(L):
                u = kk / max(L - 1, 1)
                if u < 0.30:
                    th = thA0 * (1 - u / 0.30)
                    Rm = rot_axis(axis, -th)          # toes up (heel strike roll-in)
                    pv = pv_h
                elif u < 0.60:
                    th = 0.0; Rm = np.eye(3); pv = posB
                else:
                    th = thC0 * ((u - 0.60) / 0.40)
                    Rm = rot_axis(axis, th)           # heel up (toe-off roll-out)
                    pv = pv_t
                pos = posB if th == 0.0 else pv + Rm @ (posB - pv)
                quat = quat0 if th == 0.0 else quat_mul_wxyz(mat_to_quat_wxyz(Rm), quat0)
                arc.append((pos, quat))
            anchors[(side, i0)] = arc
            roll_stats.append(dict(side=side[0], frames=L, hf=round(hf, 3),
                                   thA=round(thA0, 3), thC=round(thC0, 3),
                                   sweep=round(float(np.linalg.norm(hips[-1, :2] - hips[0, :2])), 3)))

    cost = np.full(m.nv, freeze_cost)
    cost[leg_dofs] = posture_cost_leg
    cost[:6] = freeze_cost                              # free joint frozen ...
    cost[2] = 10.0                                      # ... except root z: the
    # pelvis may dip a few cm where planted-foot reach demands it (a 0.44 m
    # step needs the hip ~3 cm lower at the stance extremes than mid-stance)
    # position cost [x, y, z]: the constructed roll arc provides z as well
    # (softer: the arc's z is approximate); orientation pinned to the arc.
    tasks = {s: mink.FrameTask(f'{s}_ankle_roll_link', 'body',
                               np.array([pin_cost, pin_cost, 4.0]), pin_cost)
             for s in SIDES}
    posture = mink.PostureTask(m, cost)
    limits = [mink.ConfigurationLimit(m, min_distance_from_limits=margin),
              mink.VelocityLimit(m, {n: vel_limit for n in LEG_JOINTS})]
    # task list is rebuilt per frame: an unpinned foot contributes nothing
    # (mink FrameTask raises TargetNotSet if listed without a target)

    n = len(qpos_seq)
    # per-frame pinned flag + anchor index per side
    pinned = {s: np.array([any(i0 <= k < i1 for i0, i1 in runs[s]) for k in range(n)]) for s in SIDES}
    def anchor_of(side: str, k: int):
        for i0, i1 in runs[side]:
            if i0 <= k < i1:
                return anchors[(side, i0)][k - i0]
        return None

    cfg = mink.Configuration(m)
    out = qpos_seq.copy()
    prev_leg = None
    band_lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)][0] + margin
                        for nm in LEG_JOINTS])
    band_hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)][1] - margin
                        for nm in LEG_JOINTS])
    resid = []
    for k in range(n):
        q = qpos_seq[k].copy()
        if prev_leg is not None:
            q[leg_qadr] = prev_leg
        q[leg_qadr] = np.clip(q[leg_qadr], band_lo, band_hi)
        cfg.update(q)
        posture.set_target(qpos_seq[k])
        err = 0.0
        active = [posture]
        for side in SIDES:
            if pinned[side][k]:
                t = tasks[side]
                t.position_cost = np.array([pin_cost, pin_cost, 4.0])
                t.orientation_cost = pin_cost
                pos, quat = anchor_of(side, k)
                t.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3(quat), pos))                   # both wxyz
                active.append(t)
        n_iter = warmup_iter if k == 0 else max_iter
        prev_err = np.inf
        # HARD freeze: only leg dofs may move -- the root trajectory (speed,
        # cadence, step length) and arms/torso stay exactly at the GMR solution
        # (a posture-soft root was measured to drift: 43_01 speed 0.84 -> 0.41,
        #  wrist-err gate failing because arm FK follows the drifting root).
        freeze_mask = np.ones(m.nv, dtype=bool)
        freeze_mask[leg_dofs] = False
        freeze_mask[2] = False                          # root z may move (see cost[2])
        for _ in range(n_iter):
            v = mink.solve_ik(cfg, active, dt, r.solver, r.damping, limits=limits)
            v[freeze_mask] = 0.0
            cfg.integrate_inplace(v, dt)
            err = float(np.linalg.norm(np.concatenate(
                [t.compute_error(cfg) for t in active if t is not posture]))) if len(active) > 1 else 0.0
            if k > 0 and prev_err - err < tol:
                break
            prev_err = err
        resid.append(err)
        out[k] = cfg.data.qpos.copy()
        prev_leg = out[k][leg_qadr].copy()
    if smooth_taps > 1:
        k = np.array([1, 4, 6, 4, 1], dtype=float)
        k /= k.sum()
        pad = len(k) // 2
        for c in leg_qadr:
            padded = np.pad(out[:, c], pad, mode='edge')
            out[:, c] = np.convolve(padded, k, mode='valid')
    # clamp smoothed leg joints inside their limits (2 cm margin kept small so
    # the clamp moves feet by at most ~mm; gate R hard-fails any over-limit)
    lo_j = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)][0] + 0.02
                     for nm in LEG_JOINTS])
    hi_j = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)][1] - 0.02
                     for nm in LEG_JOINTS])
    out[:, leg_qadr] = np.clip(out[:, leg_qadr], lo_j, hi_j)
    diag = dict(pin_resid_p50=float(np.median(resid)), pin_resid_max=float(np.max(resid)),
                n_runs={s: len(runs[s]) for s in SIDES}, rolls=roll_stats)
    return out, diag


def root_warp(qp: np.ndarray, contact: dict, model, data, fps: float = 50.0) -> tuple[np.ndarray, dict]:
    """Trajectory post-processing anchoring WITHOUT IK (preregistered round-2
    approach; stance detection by the preregistered "foot-velocity-lowest
    phase" route): rigid per-frame XY translation of the whole pose by minus
    the anchored foot's motion. Geometrically exact for the tracked point
    (corner centroid): the root translation moves every foot point
    identically, so an anchored foot's centroid is pinned and its corners
    keep only their small rotation-about-centroid motion. Joints, arms, torso
    and root orientation are untouched.

    Steps: (1) uniform per-clip grounding (min sole -> 0) so absolute z
    thresholds mean something (the raw retarget z is arbitrarily offset);
    (2) anchor runs detected from the retargeted foot itself: zmin < 35 mm,
    centroid speed < 1.0 m/s (swing feet are already excluded by the z
    window; this only removes fast through-passes), runs >= 8 frames that
    touch (zmin < 20 mm inside); (3) XY correction chains continuously
    across runs (double-anchor frames blend); (4) z: anchored feet levelled
    onto 0, linear interpolation elsewhere, any residual penetration lifted
    into the 8 mm hysteresis band (NEVER clamped onto 0, which would
    manufacture strict-contact sliding frames)."""
    import numpy as _np
    n = len(qp)
    # (1) uniform grounding
    c0 = sole_corners(model, data, qp)
    floor0 = float(min(c0[s][:, :, 2].min() for s in SIDES))
    qp = qp.copy()
    qp[:, 2] -= floor0
    corners = sole_corners(model, data, qp)
    ctr = {s: corners[s][:, :, :2].mean(axis=1) for s in SIDES}   # (n,2)
    zmin = {s: corners[s][:, :, 2].min(axis=1) for s in SIDES}
    spd = {s: _np.concatenate([_np.zeros(1), _np.linalg.norm(_np.diff(ctr[s], axis=0), axis=1)]) * fps
           for s in SIDES}
    # anchor set = UNION of the two preregistered detections:
    #   (a) human label AND foot low AND foot slow (labels have ~0.15 s
    #       mismatch on some clips; the low+slow refinement removes the
    #       whips -- the earlier failure here was a m/frame vs m/s unit bug);
    #   (b) foot-velocity-lowest phases independent of labels (catches the
    #       opposite mismatch direction: label says swing, foot is planted).
    ANCHOR_Z, ANCHOR_V, MIN_RUN = 0.050, 0.6, 6
    anchor = {s: _np.zeros(n, dtype=bool) for s in SIDES}
    for s in SIDES:
        low_slow = (zmin[s] < ANCHOR_Z) & (spd[s] < ANCHOR_V)
        for i0, i1 in runs_of(low_slow & (contact[s] if s in contact else True)):
            if i1 - i0 >= MIN_RUN:
                anchor[s][i0:i1] = True
        for i0, i1 in runs_of(low_slow):
            if i1 - i0 >= 2 * MIN_RUN:      # label-free detection: stricter
                anchor[s][i0:i1] = True
    # XY correction: carry-chained per anchor run (world origin drifts forward
    # by the removed slide; continuity across runs)
    corr = _np.zeros((n, 2))
    state: dict[str, dict | None] = {'left': None, 'right': None}
    for t in range(n):
        for s in SIDES:
            if anchor[s][t] and state[s] is None:
                prev = corr[t - 1] if t > 0 else _np.zeros(2)
                state[s] = dict(ref=ctr[s][t].copy(), carry=prev.copy())
            elif not anchor[s][t] and state[s] is not None:
                state[s] = None
        act = [s for s in SIDES if state[s] is not None]
        if act:
            corr[t] = _np.mean([state[s]['carry'] + ctr[s][t] - state[s]['ref'] for s in act], axis=0)
        else:
            corr[t] = corr[t - 1] if t > 0 else _np.zeros(2)
    # z correction: ABSOLUTE -- active anchor soles level onto 0 (mean if
    # double), linear interpolation elsewhere
    idx = _np.arange(n)
    any_anchor = anchor['left'] | anchor['right']
    shift = _np.zeros(n)
    if any_anchor.any():
        zacc = _np.zeros(n)
        zcnt = _np.zeros(n)
        for s in SIDES:
            zacc[anchor[s]] += zmin[s][anchor[s]]
            zcnt[anchor[s]] += 1
        shift[any_anchor] = zacc[any_anchor] / zcnt[any_anchor]
        if (~any_anchor).any():
            shift = _np.interp(idx, idx[any_anchor], shift[any_anchor])
    out = qp.copy()
    out[:, :2] -= corr
    out[:, 2] -= shift
    # anti-penetration: lift below-zero soles into the 8 mm hysteresis band
    c2 = sole_corners(model, data, out)
    pen = _np.minimum(0.0, _np.minimum(c2['left'][:, :, 2].min(axis=1), c2['right'][:, :, 2].min(axis=1)))
    out[:, 2] += -pen + _np.where(pen < 0, 0.008, 0.0)
    dc = _np.linalg.norm(_np.diff(corr, axis=0), axis=1) if n > 1 else _np.zeros(1)
    diag = dict(max_corr_step_m=float(dc.max()), mean_corr_rate_mps=float(dc.mean() * fps),
                warp_xy_total_m=float(_np.linalg.norm(corr[-1] - corr[0])),
                pen_max_m=float(-pen.min()), anchor_frac={s: float(anchor[s].mean()) for s in SIDES},
                anchor_runs={s: len(runs_of(anchor[s])) for s in SIDES})
    return out, {**diag, 'anchor': {s: anchor[s] for s in SIDES}}


def stance_ground(model, data, qp: np.ndarray, contact: dict) -> tuple[np.ndarray, dict]:
    """Stance-aware grounding: per-frame root-z shift that puts the STANCE
    foot's sole on z=0 (double stance: mean of the two soles; flight: hold).
    Absorbs IK and smoothing residuals in z; flight frames are preserved."""
    n = len(qp)
    corners = sole_corners(model, data, qp)
    zmin = {s: corners[s][:, :, 2].min(axis=1) for s in SIDES}
    shift = np.zeros(n)
    any_c = np.zeros(n, dtype=bool)
    for s in SIDES:
        c = contact[s]
        shift[c] += zmin[s][c]
        any_c |= c
    nz = (contact['left'].astype(int) + contact['right'].astype(int))
    shift[any_c] /= nz[any_c]
    if (~any_c).any():
        idx = np.arange(n)
        shift = np.interp(idx, idx[any_c], shift[any_c])
    out = qp.copy()
    out[:, 2] -= shift
    c2 = sole_corners(model, data, out)
    floor = float(min(c2[s][:, :, 2].min() for s in SIDES))
    out[:, 2] -= floor
    diag = dict(per_frame_shift_max=float(np.abs(np.diff(shift)).max()) if n > 1 else 0.0,
                uniform_floor=float(floor))
    return out, diag


def uniform_ground(model, data, qp: np.ndarray) -> tuple[np.ndarray, float]:
    """Single constant per-clip z shift so the clip's lowest sole point is 0."""
    corners = sole_corners(model, data, qp)
    floor = float(min(corners[s][:, :, 2].min() for s in SIDES))
    out = qp.copy()
    out[:, 2] -= floor
    return out, floor


def conservative_slip(qp: np.ndarray, model, data, fps: float,
                      strict_z: float = 0.003, boundary: int = 3):
    """audit/slip_conservative.py semantics, computed inline for gating:
    per foot, over STRICT contact frames (sole z < strict_z on consecutive
    frames AND interior of its Schmitt stance phase), the horizontal speed of
    the slowest of the 4 sole corners. Returns per-foot dict + flight count."""
    corners = sole_corners(model, data, qp)
    zmin = {s: corners[s][:, :, 2].min(axis=1) for s in SIDES}
    dt = 1.0 / fps
    out = {'feet': {}, 'flight_frames_both_above_25mm':
           int(((zmin['left'] > 0.025) & (zmin['right'] > 0.025)).sum())}
    for s in SIDES:
        z = zmin[s]
        v = np.linalg.norm(np.diff(corners[s][:, :, :2], axis=0), axis=2).min(axis=1) / dt
        mask = (z[:-1] < strict_z) & (z[1:] < strict_z)
        # interior of Schmitt stance phases (AIR 25mm / GROUND 8mm, min 4 frames)
        interior = np.zeros(len(z), dtype=bool)
        for kind, i0, i1 in _segment(z):
            if kind == 'stance':
                lo, hi = i0 + boundary, max(i1 - boundary, i0 + boundary)
                interior[lo:hi] = True
        m = mask & interior[:-1]
        vs = v[m]
        out['feet'][s] = dict(
            n_strict=int(m.sum()),
            slip_p50=float(np.median(vs)) if len(vs) else None,
            slip_p90=float(np.percentile(vs, 90)) if len(vs) else None,
            slide_frac_gt015=float((vs > 0.15).mean()) if len(vs) else None)
    return out


def _segment(z: np.ndarray, air: float = 0.025, ground: float = 0.008, min_frames: int = 4):
    """Schmitt phases, audit/fk_gait_stats.segment_phases semantics."""
    n = len(z)
    phases = []
    state = 'stance' if z[0] < ground else ('swing' if z[0] > air else 'band')
    start = 0
    for i in range(1, n):
        if state == 'stance' and z[i] > air:
            phases.append(('stance', start, i)); state, start = 'swing', i
        elif state == 'swing' and z[i] < ground:
            phases.append(('swing', start, i)); state, start = 'stance', i
        elif state == 'band':
            if z[i] > air: state, start = 'swing', i
            elif z[i] < ground: state, start = 'stance', i
    phases.append((state, start, n))
    out = []
    for kind, i0, i1 in phases:
        if out and (i1 - i0) < (min_frames if kind == 'swing' else min_frames):
            k0, j0, j1 = out[-1]
            out[-1] = (k0, j0, i1)
        else:
            out.append((kind, i0, i1))
    return out
