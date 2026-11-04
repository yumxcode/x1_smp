#!/usr/bin/env python3
"""AUDIT (independent): gait statistics of the retargeted X1 library, computed
directly from qpos foot trajectories with an implementation that shares NO code
with scripts/x1_gait.py / scripts/x1_data_lib.gait_metrics.

Differences from the library tools (deliberate):
  * contact is decided from ABSOLUTE sole height (data is grounded per frame,
    so the stance foot sits at z≈0), not from the RELATIVE height between feet;
  * swing/stance segmentation uses a Schmitt trigger on absolute height
    (air > 2.5 cm, ground < 0.8 cm, min phase 4 frames @50Hz=0.08s);
  * step length uses the FOOT's own xy displacement between successive
    touchdowns (not root displacement);
  * stance slip: while a foot is classified in contact, the horizontal speed of
    its sole center is measured -- if the retargeted data has the "stance" foot
    sliding, this is the number that shows it.

Run:  .venv/bin/python audit/fk_gait_stats.py [--npz data/x1_smp_train.npz]
Output: audit/gait_stats_independent.json (+ stdout table)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / 'assets' / 'x1_29dof' / 'x1_29dof.xml'

AIR, GROUND = 0.025, 0.008          # m, absolute sole height
MIN_AIR_FRAMES, MIN_CONTACT_FRAMES = 4, 4


def foot_geom_data(model):
    """per foot: the 4 sole collision geom ids and their local offsets."""
    feet = {}
    for fb in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb)
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid
                and 'collision' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
        feet[fb] = (bid, gids)
    return feet


def segment_phases(z: np.ndarray):
    """Schmitt trigger on absolute height -> list of (kind, i0, i1) half-open."""
    n = len(z)
    contact = z < GROUND
    # a frame is airborne if z > AIR; hysteresis resolves the band in between
    phases = []
    state = 'stance' if contact[0] else ('swing' if z[0] > AIR else 'band')
    start = 0
    for i in range(1, n):
        if state == 'stance' and z[i] > AIR:
            phases.append(('stance', start, i)); state, start = 'swing', i
        elif state == 'swing' and z[i] < GROUND:
            phases.append(('swing', start, i)); state, start = 'stance', i
        elif state == 'band':
            if z[i] > AIR:
                state, start = 'swing', i
            elif z[i] < GROUND:
                state, start = 'stance', i
    phases.append((state, start, n))
    # drop flicker: phases shorter than the minimum length get merged away
    out = []
    for kind, i0, i1 in phases:
        if out and (i1 - i0) < (MIN_AIR_FRAMES if kind == 'swing' else MIN_CONTACT_FRAMES):
            # merge tiny phase into previous (keep previous kind, extend)
            k0, j0, j1 = out[-1]
            out[-1] = (k0, j0, i1)
        else:
            out.append((kind, i0, i1))
    return out


def analyse_clip(model, data, feet, qpos: np.ndarray, fps: float):
    dt = 1.0 / fps
    T = len(qpos)
    names = ('L', 'R')
    per_foot = {}
    zs, xys = {}, {}
    for k, fb in enumerate(('left_ankle_roll_link', 'right_ankle_roll_link')):
        bid, gids = feet[fb]
        zmin = np.empty(T); zctr = np.empty(T); xy = np.empty((T, 2))
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            gp = np.array([data.geom_xpos[g] for g in gids])  # (4,3)
            zmin[t] = gp[:, 2].min()
            zctr[t] = gp[:, 2].mean()
            xy[t] = gp[:, :2].mean(axis=0)
        zs[names[k]] = zmin; xys[names[k]] = xy
        # root
    root = qpos[:, :3]
    vxy = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) / dt
    lo, hi = int(0.1 * T), max(int(0.9 * T), int(0.1 * T) + 1)
    speed = float(np.median(vxy[lo:hi]))

    res = {'speed_mps': speed, 'root_z': [float(root[:, 2].min()), float(root[:, 2].max())],
           'feet': {}}
    td_all = {}
    for k in names:
        z = zs[k]; xy = xys[k]
        phases = segment_phases(z)
        swings = [(i1 - i0) * dt for kind, i0, i1 in phases if kind == 'swing' and (i1 - i0) * dt >= 0.08]
        stances = [(i1 - i0) * dt for kind, i0, i1 in phases if kind == 'stance' and (i1 - i0) * dt >= 0.08]
        # touchdowns = phase-start indices of stance segments (i0>0: mid-clip)
        td = [i0 for kind, i0, i1 in phases if kind == 'stance' and i0 > 0]
        td_all[k] = td
        cadence = float(1.0 / np.median(np.diff(td)) * fps) if len(td) >= 2 else float('nan')
        stride_s = float(np.median(np.diff(td)) * dt) if len(td) >= 2 else float('nan')
        # step length from the foot's own displacement at successive touchdowns
        if len(td) >= 2:
            pts = xy[td]
            sl = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            sl = sl[sl > 0.05]
            step_len = float(np.median(sl)) if len(sl) else float('nan')  # full stride per foot cycle
        else:
            step_len = float('nan')
        contact_frac = float((z < GROUND + 0.004).mean())
        lift_max = float(z.max()); lift_p90 = float(np.percentile(z, 90))
        # stance slip: horizontal speed of sole center on contact frames
        mask = z < GROUND + 0.004
        vfoot = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        if mask.sum() > 3:
            vf_c = vfoot[np.where(mask[:-1] & mask[1:])[0]]
            slip = dict(n=int(mask.sum()),
                        mean=float(vf_c.mean()), p50=float(np.median(vf_c)),
                        p90=float(np.percentile(vf_c, 90)), max=float(vf_c.max()))
            # total slip distance while in contact
            steps_xy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            slip_dist = float(steps_xy[np.where(mask[:-1] & mask[1:])[0]].sum())
        else:
            slip = None; slip_dist = float('nan')
        res['feet'][k] = dict(
            n_swing=len(swings), swing_s_med=float(np.median(swings)) if swings else float('nan'),
            swing_s_max=float(np.max(swings)) if swings else float('nan'),
            n_stance=len(stances), stance_s_med=float(np.median(stances)) if stances else float('nan'),
            cadence_hz=cadence, stride_s=stride_s, step_len_m=step_len,
            duty_abs=contact_frac, lift_max_m=lift_max, lift_p90_m=lift_p90,
            stance_slip_mps=slip, stance_slip_dist_m=slip_dist,
            sole_z_p50=float(np.percentile(z, 50)), sole_z_max=float(z.max()),
            touchdowns=len(td))
    # grounded-ness: the per-frame minimum sole across both feet
    bothmin = np.minimum(zs['L'], zs['R'])
    res['min_sole_both_p50'] = float(np.percentile(bothmin, 50))
    res['min_sole_both_max'] = float(bothmin.max())
    # travel
    res['xy_travel_m'] = float(np.hypot(*(root[-1, :2] - root[0, :2])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=str(ROOT / 'data' / 'x1_smp_train.npz'))
    ap.add_argument('--out', default=str(ROOT / 'audit' / 'gait_stats_independent.json'))
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    feet = foot_geom_data(model)

    d = np.load(args.npz, allow_pickle=True)
    names = [str(n) for n in d['names']]
    starts, lengths, fps = d['starts'], d['lengths'], float(d['fps'])
    report = {}
    for n, s, L in zip(names, starts, lengths):
        q = np.asarray(d['qpos'][s:s + L], dtype=np.float64)
        r = analyse_clip(model, data, feet, q, fps)
        report[n] = r
        fL, fR = r['feet']['L'], r['feet']['R']

        def fmt(x):
            return 'nan' if x != x else f'{x:.3f}'
        print(f"{n:15s} v={r['speed_mps']:.2f} | L: cad {fmt(fL['cadence_hz'])} swing {fmt(fL['swing_s_med'])} "
              f"duty {fL['duty_abs']:.2f} lift {1000*fL['lift_max_m']:.0f}mm step {fmt(fL['step_len_m'])} "
              f"slip(p50) {fmt(fL['stance_slip_mps']['p50']) if fL['stance_slip_mps'] else 'na'} "
              f"p90 {fmt(fL['stance_slip_mps']['p90']) if fL['stance_slip_mps'] else 'na'} | "
              f"R: cad {fmt(fR['cadence_hz'])} swing {fmt(fR['swing_s_med'])} duty {fR['duty_abs']:.2f} "
              f"lift {1000*fR['lift_max_m']:.0f}mm step {fmt(fR['step_len_m'])} "
              f"slip(p50) {fmt(fR['stance_slip_mps']['p50']) if fR['stance_slip_mps'] else 'na'} "
              f"p90 {fmt(fR['stance_slip_mps']['p90']) if fR['stance_slip_mps'] else 'na'}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
