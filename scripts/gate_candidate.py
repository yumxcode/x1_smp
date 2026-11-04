#!/usr/bin/env python3
"""G1' (amendment rev2): gate candidate X1 walking libraries against the FULL
G1 hard-gate suite before they may be used for SMP prior training.

Candidates:
  * X1_29_AMP policy MuJoCo rollout npz (physics-solved contact): keys
    t/q/dq/base_pos/base_quat/sole_xyz/contact_dist, logged at 50 Hz, MuJoCo
    wxyz quat, joint order == X1_JOINT_ORDER (verified against meta).
  * 36-col qpos library npz (qpos/qvel/starts/lengths/names/fps), e.g. the
    GMR libraries kept for contrast.

Gates (preregistered, unchanged from G1):
  S  slip_conservative stance slip p50 < 0.05 m/s per foot
     (strict contact z<3mm consecutive + Schmitt-interior, min-corner speed);
     flight frames (both soles > 25 mm) > 0 per clip
  B  gait bands v3 (cadence 0.55-1.2 Hz/foot, speed 0.4-1.3, step ratio
     0.40-0.95 & <=0.65, swing >=0.15, duty <=0.87, lift 0.025-0.25) + symmetry
  A  retarget quality (over-limit <=0.35, sole pen > -0.02, root_z >=0.45,
     tilt <=0.35, qd <=25)
  R  full retarget-acceptance gates via tools.x1_retarget_gates.score_clip;
     sub-gates needing a HUMAN reference (wrist_err p50/p90, chest_yaw_err)
     are N/A for rollouts and reported as such -- never silently passed.

Run:
  .venv/bin/python scripts/gate_candidate.py <npz...> [--head-s 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'pipeline'))
sys.path.insert(0, str(Path.home() / 'code' / 'super_mimic_F1'))

from x1_data_lib import X1FK, gait_metrics, audit_pass  # noqa: E402
from stance_anchor import conservative_slip, sole_corners, SIDES  # noqa: E402
from tools.x1_retarget_gates import GateModel, score_clip  # noqa: E402

ROOT = HERE.parent
HUMAN_REF_GATES = {'wrist_err_p50_m', 'wrist_err_p90_m', 'chest_yaw_err_p90_deg'}


def load_rollout(p: Path, head_s: float):
    d = np.load(p, allow_pickle=True)
    t = np.asarray(d['t'], dtype=float)
    dt = float(np.median(np.diff(t)))
    qpos = np.concatenate([d['base_pos'], d['base_quat'], d['q']], axis=1).astype(np.float64)
    head = int(round(head_s / dt))
    qpos = qpos[head:]
    ref_soles = d['sole_xyz'][head:] if 'sole_xyz' in d.files else None
    return qpos, 1.0 / dt, ref_soles


def gate_one(qpos: np.ndarray, fps: float, fk: X1FK, gate_model: GateModel,
             ref_soles=None) -> dict:
    T = len(qpos)
    # 35-col qvel in the library contract: root lin (3), root ang (3, unused by
    # the gates), joint vel (29). Finite difference, same as lib builds.
    grad = np.gradient(qpos, axis=0) * fps
    qv = np.concatenate([grad[:, :3], np.zeros((T, 3)), grad[:, 7:]], axis=1)
    qd_max = float(np.abs(grad[:, 7:]).max())

    # model consistency: our FK sole corners vs the rollout's logged sole_xyz
    fk_check = None
    if ref_soles is not None:
        ours = sole_corners(fk.model, fk.data, qpos)
        diffs = [np.abs(ours[s][:len(ref_soles)] - ref_soles[:, i]).max()
                 for i, s in enumerate(SIDES)]
        fk_check = float(max(diffs))

    # ---- gate A ----
    q = qpos[:, 7:]
    lo, hi = fk.model.jnt_range[1:, 0], fk.model.jnt_range[1:, 1]
    over = np.maximum(q - hi, lo - q)
    tilt = 2 * (qpos[:, 4] ** 2 + qpos[:, 5] ** 2)
    k = fk.fk_clip(qpos, qv)
    a_pass = bool(over.max() <= 0.35 and k['sole_h'].min() > -0.02
                  and qpos[:, 2].mean() >= 0.45 and tilt.max() <= 0.35
                  and qd_max <= 25.0)

    # ---- gate B ----
    gm = gait_metrics(k['sole_h'], k['root_pos'], fps)
    b_pass, vals = audit_pass(gm)
    f0, f1 = gm['feet']
    sym = (not np.isnan(f0['duty']) and not np.isnan(f1['duty'])
           and abs(f0['duty'] - f1['duty']) <= 0.25
           and abs(f0['cadence_hz'] - f1['cadence_hz']) <= 0.35)
    b_pass = bool(b_pass and sym)

    # ---- gate S ----
    slip = conservative_slip(qpos, fk.model, fk.data, fps)
    s_pass = True
    s_stat = {}
    for s in SIDES:
        st = slip['feet'][s]
        s_stat[s] = st
        s_pass &= st['slip_p50'] is not None and st['slip_p50'] < 0.05
    flight = slip['flight_frames_both_above_25mm']
    s_pass &= flight > 0
    # diagnostics (not gated for external rollouts: physics contact IS the
    # reference; strict-contact fraction reported for transparency)
    for s in SIDES:
        z = k['sole_h'][SIDES.index(s)]
        s_stat[s]['contact_frac_z3mm'] = float((z < 0.003).mean())

    # ---- gate R (applicable subset) ----
    sc = score_clip(qpos, fps, gate_model, tag='walk', human_net_yaw=0.0)
    r_gates = {g: (ok if g not in HUMAN_REF_GATES else None)
               for g, ok in sc['gates'].items()}
    r_pass = all(ok for ok in r_gates.values() if ok is not None)

    return dict(frames=T, a_pass=a_pass, b_pass=b_pass, s_pass=bool(s_pass),
                r_pass=bool(r_pass),
                fk_sole_maxdiff_m=fk_check,
                gait=dict(speed=gm['speed_mps'],
                          cadence=[f0['cadence_hz'], f1['cadence_hz']],
                          step=[f0['step_len_m'], f1['step_len_m']],
                          swing=[f0['swing_s_med'], f1['swing_s_med']],
                          duty=[f0['duty'], f1['duty']],
                          lift=[f0['lift_m'], f1['lift_m']]),
                bands=vals,
                slip=s_stat, flight=flight,
                r_gates=r_gates, r_metrics=sc['metrics'],
                root_z=[float(qpos[:, 2].min()), float(qpos[:, 2].max())])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='+', type=Path)
    ap.add_argument('--head-s', type=float, default=0.5)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'candidate_gates.json')
    args = ap.parse_args()

    fk = X1FK()
    gate_model = GateModel()
    report = {}
    for p in args.npz:
        d = np.load(p, allow_pickle=True)
        if 'base_pos' in d.files:
            clips = [load_rollout(p, args.head_s)]
            names = [p.stem]
        else:
            clips = [(np.asarray(d['qpos'][s:s + int(L)], dtype=np.float64),
                      float(d['fps']), None)
                     for s, L in zip(d['starts'], d['lengths'])]
            names = [f'{p.stem}:{n}' for n in d['names']]
        for (qpos, fps, ref), name in zip(clips, names):
            if len(qpos) < 100:
                report[name] = {'error': 'too short'}
                continue
            r = gate_one(qpos, fps, fk, gate_model, ref)
            report[name] = r
            ok = r['a_pass'] and r['b_pass'] and r['s_pass'] and r['r_pass']
            sl, sr = r['slip']['left']['slip_p50'], r['slip']['right']['slip_p50']
            print(f"{name:34s} v={r['gait']['speed']:.2f} "
                  f"cad {r['gait']['cadence'][0]:.2f}/{r['gait']['cadence'][1]:.2f} "
                  f"slip {sl if sl is None else round(sl, 4)}/{sr if sr is None else round(sr, 4)} "
                  f"flt {r['flight']:3d} fkΔ {r['fk_sole_maxdiff_m'] if r['fk_sole_maxdiff_m'] is None else round(r['fk_sole_maxdiff_m'] * 1000, 1)}mm "
                  f"| A{int(r['a_pass'])}B{int(r['b_pass'])}S{int(r['s_pass'])}R{int(r['r_pass'])} "
                  f"{'OK' if ok else ''} "
                  f"Rbad={[g for g, v in r['r_gates'].items() if v is False]}", flush=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f'-> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
