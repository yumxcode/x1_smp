#!/usr/bin/env python3
"""AUDIT (gates replay): reproduce the meta scorecard with the project's own
tooling, then re-evaluate every kept clip under each gate revision (v1 / v2 / v3).

Checks:
 1. meta['windows'] gait numbers == re-running scripts/x1_data_lib.gait_metrics
    on data/x1_smp_train.npz clips (scorecard honesty);
 2. band-revision replay: per kept clip, pass/fail under
      v1: cad .55-1.2, speed .4-1.3, step .35-.65, swing>=.24, duty<=.80, lift>=.025
      v2: swing>=.15, duty<=.87 (rest as v1)
      v3: v2 + step<=.65 and step/v in [.40,.95], lift<=.25
    (the build-time bands, symmetric feet extra check as in build_smp_library);
 3. which kept clips are phase2 merges and their full-gate status from
    data/x1_phase2.meta.json.

Run: .venv/bin/python audit/check_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from x1_data_lib import X1FK, gait_metrics  # noqa: E402

V1 = dict(cad=(0.55, 1.20), speed=(0.40, 1.30), step=(0.35, 0.65), swing=(0.24, None),
          duty=(None, 0.80), lift=(0.025, None))
V2 = dict(V1, swing=(0.15, None), duty=(None, 0.87))
V3 = dict(V2, step=None, step_max=0.65, step_ratio=(0.40, 0.95), lift=(0.025, 0.25))


def in_band(x, b):
    lo, hi = b
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def eval_bands(gm, bands):
    f0, f1 = gm['feet']
    v = gm['speed_mps']
    if np.isnan(f0['cadence_hz']) or np.isnan(f1['cadence_hz']):
        return False
    ok = in_band(v, bands['speed'])
    for f in (f0, f1):
        ok &= in_band(f['cadence_hz'], bands['cad'])
        if not np.isnan(f['swing_s_med']):
            ok &= in_band(f['swing_s_med'], bands['swing'])
        if not np.isnan(f['duty']):
            ok &= in_band(f['duty'], bands['duty'])
        if not np.isnan(f['lift_m']):
            ok &= in_band(f['lift_m'], bands['lift'])
        if not np.isnan(f['step_len_m']):
            if bands['step'] is not None:
                ok &= in_band(f['step_len_m'], bands['step'])
            else:
                ok &= f['step_len_m'] <= bands['step_max'] and in_band(
                    f['step_len_m'] / max(v, 1e-6), bands['step_ratio'])
    ok &= (abs(f0['duty'] - f1['duty']) <= 0.25) and (abs(f0['cadence_hz'] - f1['cadence_hz']) <= 0.35)
    return bool(ok)


def main():
    meta = json.load(open(ROOT / 'data' / 'x1_smp_train.meta.json'))
    stored = {w['clip']: w for w in meta['windows'] if 'gait' in w}
    d = np.load(ROOT / 'data' / 'x1_smp_train.npz', allow_pickle=True)
    fk = X1FK()
    rows = []
    for n, s, L in zip([str(x) for x in d['names']], d['starts'], d['lengths']):
        qp = np.asarray(d['qpos'][s:s + L])
        qv = np.asarray(d['qvel'][s:s + L])
        k = fk.fk_clip(qp, qv)
        gm = gait_metrics(k['sole_h'], k['root_pos'], float(d['fps']))
        f0, f1 = gm['feet']
        st = stored.get(n)
        match = ''
        if st:
            ds = abs(st['gait']['speed'] - gm['speed_mps'])
            dc = max(abs(st['gait']['cadence'][i] - (f0, f1)[i]['cadence_hz']) for i in (0, 1)
                     if not np.isnan((f0, f1)[i]['cadence_hz']))
            match = f'meta_match(dspd {ds:.3f}, dcad {dc:.3f})'
        v1p = eval_bands(gm, V1)
        v2p = eval_bands(gm, V2)
        v3p = eval_bands(gm, V3)
        rows.append((n, gm['speed_mps'], v1p, v2p, v3p, match))
        print(f'{n:15s} v={gm["speed_mps"]:.2f} v1pass={v1p} v2={v2p} v3={v3p} {match}', flush=True)
    n1 = sum(r[2] for r in rows); n2 = sum(r[3] for r in rows); n3 = sum(r[4] for r in rows)
    print(f'\nkept-18 pass counts: v1={n1}  v2={n2}  v3(build)={n3}')
    out = dict(rows=[dict(clip=r[0], speed=r[1], v1=r[2], v2=r[3], v3=r[4], note=r[5]) for r in rows],
               counts=dict(v1=n1, v2=n2, v3=n3))
    # phase2 gate status of the p2 merges
    try:
        p2 = json.load(open(ROOT / 'data' / 'x1_phase2.meta.json'))
        kept_p2 = [r[0] for r in rows if r[0].startswith('p2_')]
        status = {}
        for c in p2.get('clips', []):
            name = c.get('name') or c.get('clip')
            if name and any(name.endswith(k.split('p2_')[1]) or k.endswith(name) for k in kept_p2):
                status[name] = c.get('gates', c.get('pass', c.get('retarget_gates', '?')))
        out['phase2_kept_gate_status'] = status
        print('phase2 kept clips gate fields:', json.dumps(status)[:600])
    except Exception as e:  # noqa: BLE001
        print('phase2 meta parse:', e)
    Path(ROOT / 'audit' / 'gates_replay.json').write_text(json.dumps(out, indent=1))
    print('-> audit/gates_replay.json')


if __name__ == '__main__':
    main()
