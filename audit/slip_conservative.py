#!/usr/bin/env python3
"""AUDIT (independent, part 3): conservative slip measure + grounding check.

  slip_min_corner(t) = min over the foot's 4 sole corners of |d/dt corner_xy|.
  This is a LOWER bound on "is any foot point planted": if even the slowest
  corner moves at v, no point of the foot is static. Corner-switching and
  rollover artifacts cannot inflate it.

  grounding check: the library was built with per-frame root-z shift so that
  min(sole_L, sole_R) == 0 every frame; we verify that signature exactly
  (p50 of per-frame min sole ~ 0 and >99% frames within +-2mm), and count
  "flight frames" (both feet > 25 mm), which healthy walks have but a
  per-frame-grounded library cannot (except frames where the minimum still
  lands exactly at 0 by chance of the shift being 0).

Run: .venv/bin/python audit/slip_conservative.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / 'assets' / 'x1_29dof' / 'x1_29dof.xml'
AIR = 0.025
STRICT = 0.003
SLIDE_T = 0.15


def main():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    feet = {}
    for fb in ('left_ankle_roll_link', 'right_ankle_roll_link'):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb)
        feet[fb] = [g for g in range(model.ngeom)
                    if model.geom_bodyid[g] == bid
                    and 'collision' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]
    d = np.load(ROOT / 'data' / 'x1_smp_train.npz', allow_pickle=True)
    report = {}
    for n, s, L in zip(d['names'], d['starts'], d['lengths']):
        q = np.asarray(d['qpos'][s:s + L], dtype=np.float64)
        T = len(q)
        C = {}
        for k, fb in (('L', 'left_ankle_roll_link'), ('R', 'right_ankle_roll_link')):
            corners = np.empty((T, 4, 3))
            for t in range(T):
                data.qpos[:] = q[t]
                mujoco.mj_forward(model, data)
                corners[t] = [data.geom_xpos[g] for g in feet[fb]]
            C[k] = corners
        zmin = {k: C[k][:, :, 2].min(axis=1) for k in C}
        vmin = {}
        for k in C:
            v = np.linalg.norm(np.diff(C[k][:, :, :2], axis=0), axis=2) / 0.02  # (T-1,4)
            vmin[k] = v.min(axis=1)
        row = {}
        both_min = np.minimum(zmin['L'], zmin['R'])
        row['grounding'] = dict(
            min_sole_p50=float(np.median(both_min)),
            frac_within_2mm=float((np.abs(both_min) < 0.002).mean()),
            flight_frames_both_above_25mm=int(((zmin['L'] > AIR) & (zmin['R'] > AIR)).sum()))
        for k in ('L', 'R'):
            z, v = zmin[k], vmin[k]
            m = (z[:-1] < STRICT) & (z[1:] < STRICT)
            vs = v[m]
            row[k] = dict(
                n_strict=int(m.sum()),
                slip_min_corner_p50=float(np.median(vs)) if len(vs) else None,
                p90=float(np.percentile(vs, 90)) if len(vs) else None,
                slide_frac=float((vs > SLIDE_T).mean()) if len(vs) else None)
        report[str(n)] = row
        a, b = row['L'], row['R']
        print(f"{str(n):15s} gnd: p50min {row['grounding']['min_sole_p50']:+.4f} "
              f"|2mm| {row['grounding']['frac_within_2mm']:.2f} flight {row['grounding']['flight_frames_both_above_25mm']:3d} | "
              f"L n{a['n_strict']:4d} minslip p50 {a['slip_min_corner_p50']:.2f} slide% {a['slide_frac']:.2f} | "
              f"R n{b['n_strict']:4d} minslip p50 {b['slip_min_corner_p50']:.2f} slide% {b['slide_frac']:.2f}", flush=True)
    Path(ROOT / 'audit' / 'slip_conservative.json').write_text(json.dumps(report, indent=1))
    print('-> audit/slip_conservative.json')


if __name__ == '__main__':
    main()
