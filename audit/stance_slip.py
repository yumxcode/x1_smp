#!/usr/bin/env python3
"""AUDIT (independent, part 2): stance slip with a contact-point (lowest-corner)
velocity measure, boundary-frame exclusion, and a strict-contact / near-ground
drag split. Also dumps per-frame trajectories for selected clips (the
"frame-by-frame manual check" requested by the audit brief).

Metrics per foot:
  strict contact  sole z < 3 mm   -> sliding if the lowest corner moves
  drag band       3 mm <= z < 25 mm while the foot is not airborne (no clean
                  swing phase) -> drag-stepping
  airborne        z > 25 mm

  pivot_speed(t): horizontal speed of the corner that is lowest at t
                  (the would-be contact point). Planted foot -> ~0.
  For each clip we also report "no-support frames": frames where BOTH feet are
  below 25 mm (neither in a swing) yet both pivot speeds > 0.2 m/s -- the
  whole robot is sliding / dragging on both feet.

Run: .venv/bin/python audit/stance_slip.py [--clips 43_01_w0 136_21_w0 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / 'assets' / 'x1_29dof' / 'x1_29dof.xml'
AIR = 0.025
STRICT = 0.003
GUARD = 3          # frames excluded at stance-phase boundaries
SLIDE_T = 0.15     # m/s pivot speed above which a contact frame counts as sliding


def foot_geom_ids(model):
    feet = {}
    for fb in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb)
        gids = [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid
                and 'collision' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
        feet[fb] = gids
    return feet


def fk_soles(model, data, gids, qpos):
    T = len(qpos)
    corners = np.empty((T, 4, 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        corners[t] = [data.geom_xpos[g] for g in gids]
    return corners


def stance_runs(zmin):
    """boolean airborne (z>AIR) -> stance runs as (i0,i1) inside non-airborne spans."""
    air = zmin > AIR
    runs, i = [], 0
    n = len(zmin)
    while i < n:
        if not air[i]:
            j = i
            while j < n and not air[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def analyse_clip(model, data, feet, qpos, fps, dump=None):
    dt = 1.0 / fps
    T = len(qpos)
    out = {'feet': {}}
    corners = {k: fk_soles(model, data, feet[f'{k}_ankle_roll_link'], qpos)
               for k in ('left', 'right')}
    zmin = {k: corners[k][:, :, 2].min(axis=1) for k in corners}
    # pivot point per frame: the lowest corner
    pivot = {}
    pv = {}
    for k in corners:
        idx = corners[k][:, :, 2].argmin(axis=1)
        pivot[k] = corners[k][np.arange(T), idx, :2]
        pv[k] = np.linalg.norm(np.diff(pivot[k], axis=0), axis=1) / dt  # (T-1,)

    rows = []
    for k, name in (('left', 'L'), ('right', 'R')):
        z = zmin[k]
        v = pv[k]
        runs = stance_runs(z)
        # interior frames of each run, boundary-guarded
        inner = np.zeros(T, dtype=bool)
        for i0, i1 in runs:
            inner[max(i0 + GUARD, 0):max(i1 - GUARD, 0)] = True
        strict = (z < STRICT) & inner
        dragband = (z >= STRICT) & (z < AIR) & inner
        n_strict, n_drag = int(strict.sum()), int(dragband.sum())
        # frame-to-frame speeds on those frames (use v[t] where mask t and t+1 both in same class)
        def speeds(mask):
            m = mask[:-1] & mask[1:]
            return v[m] if m.any() else np.array([])

        s_strict, s_drag = speeds(strict), speeds(dragband)
        # sliding fraction among strict-contact frames
        sliding_frac = float((s_strict > SLIDE_T).mean()) if len(s_strict) else float('nan')
        # per-run median pivot speed (one number per stance phase, robust)
        run_med = []
        for i0, i1 in runs:
            seg = slice(max(i0 + GUARD, 0), max(i1 - GUARD, i0 + GUARD + 1))
            zz = z[seg]
            mm = zz < STRICT
            if mm.sum() >= 5:
                vv = pv[k][seg][:-1][mm[:-1] & mm[1:]]
                if len(vv):
                    run_med.append((float(np.median(vv)), int(mm.sum())))
        out['feet'][name] = dict(
            frames=T,
            airborne_frac=float((z > AIR).mean()),
            strict_contact_frames=n_strict,
            dragband_frames=n_drag,
            strict_slip_mps=dict(
                p50=float(np.median(s_strict)) if len(s_strict) else None,
                mean=float(s_strict.mean()) if len(s_strict) else None,
                p90=float(np.percentile(s_strict, 90)) if len(s_strict) else None,
                sliding_frac=sliding_frac),
            drag_speed_mps=dict(
                p50=float(np.median(s_drag)) if len(s_drag) else None,
                p90=float(np.percentile(s_drag, 90)) if len(s_drag) else None),
            stance_runs=len(runs),
            per_run_median_strict_slip=[round(m, 3) for m, _ in run_med],
        )

    # no-support frames: both feet below AIR and both pivots fast
    both_grounded = (zmin['left'] < AIR) & (zmin['right'] < AIR)
    fastL = np.zeros(T, bool); fastR = np.zeros(T, bool)
    fastL[:-1] = pv['left'] > SLIDE_T; fastR[:-1] = pv['right'] > SLIDE_T
    nosupp = both_grounded & fastL & fastR
    out['no_support_frames'] = int(nosupp.sum())
    out['no_support_frac'] = float(nosupp.mean())
    out['both_grounded_frac'] = float(both_grounded.mean())

    if dump is not None:
        with open(dump, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'sole_z_L', 'sole_z_R', 'pivot_vx_L', 'pivot_v_L',
                        'pivot_vx_R', 'pivot_v_R', 'both_grounded', 'no_support'])
            for t in range(T):
                w.writerow([t, f'{zmin["left"][t]:.4f}', f'{zmin["right"][t]:.4f}',
                            f'{pv["left"][t] if t < T-1 else float("nan"):.3f}',
                            f'{(zmin["left"][t] < AIR)}',
                            f'{pv["right"][t] if t < T-1 else float("nan"):.3f}',
                            f'{(zmin["right"][t] < AIR)}',
                            int(both_grounded[t]), int(nosupp[t] if t < T else 0)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=str(ROOT / 'data' / 'x1_smp_train.npz'))
    ap.add_argument('--out', default=str(ROOT / 'audit' / 'stance_slip.json'))
    ap.add_argument('--dump', default='43_01_w0,136_21_w0,108_09_w0,p2_132_34_walk,105_19_w0',
                    help='clips to dump per-frame csv for')
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    feet = foot_geom_ids(model)
    d = np.load(args.npz, allow_pickle=True)
    names = [str(n) for n in d['names']]
    dump_set = set(args.dump.split(','))
    report = {}
    dumpdir = ROOT / 'audit' / 'frames'
    dumpdir.mkdir(exist_ok=True)
    for n, s, L in zip(names, d['starts'], d['lengths']):
        q = np.asarray(d['qpos'][s:s + L], dtype=np.float64)
        dump = dumpdir / f'{n}.csv' if n in dump_set else None
        r = analyse_clip(model, data, feet, q, float(d['fps']), dump)
        report[n] = r
        fL, fR = r['feet']['L'], r['feet']['R']
        ssL, ssR = fL['strict_slip_mps'], fR['strict_slip_mps']

        def g(ss):
            return 'na' if ss['p50'] is None else f"{ss['p50']:.2f}/{ss['p90']:.2f}/sl{ss['sliding_frac']:.2f}"
        print(f"{n:15s} air% L {fL['airborne_frac']:.2f} R {fR['airborne_frac']:.2f} | "
              f"strict n L{fL['strict_contact_frames']:4d} R{fR['strict_contact_frames']:4d} | "
              f"strict slip p50/p90/slide% L {g(ssL)} R {g(ssR)} | "
              f"drag n L{fL['dragband_frames']:4d} R{fR['dragband_frames']:4d} | "
              f"both_gnd {r['both_grounded_frac']:.2f} nosupp {r['no_support_frac']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f'-> {args.out}; per-frame csvs in audit/frames/')


if __name__ == '__main__':
    main()
