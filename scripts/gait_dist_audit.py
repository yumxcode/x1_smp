#!/usr/bin/env python3
"""Gait-distribution audit of the retargeted X1 walk library (preregistered).

For every *_walk clip in motions/x1_phase2_{train,heldout}.npz (50 Hz qpos,
already retargeted + arm-fixed by the super_mimic_F1 pipeline), run FK and
measure rhythm_gates-compatible metrics, then apply the preregistered data
bands (docs/SMP_PLAN.md §4):

  cadence/foot 0.55-1.20 Hz | speed 0.40-1.30 m/s | step(stride/2) 0.35-0.65 m
  swing >= 0.24 s (data side) | duty <= 0.80 | lift >= 25 mm

Output: data/gait_audit.json (per-clip metrics + pass flags + distribution
summary). Clips that fail are excluded from prior training; this script only
reports — build_x1_windows.py consumes the pass list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x1_data_lib import X1FK, gait_metrics, audit_pass  # noqa: E402

DEFAULT_TRAIN = str(Path.home() / 'code/super_mimic_F1/motions/x1_phase2_train.npz')
DEFAULT_HELDOUT = str(Path.home() / 'code/super_mimic_F1/motions/x1_phase2_heldout.npz')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default=DEFAULT_TRAIN)
    ap.add_argument('--heldout', default=DEFAULT_HELDOUT)
    ap.add_argument('--out', default='data/gait_audit.json')
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N clips')
    args = ap.parse_args()

    fk = X1FK()
    report = {'clips': {}, 'summary': {}}

    for split, path in (('train', args.train), ('heldout', args.heldout)):
        d = np.load(path, allow_pickle=True)
        names = [str(n) for n in d['names']]
        starts, lengths, qpos, qvel, fps = d['starts'], d['lengths'], d['qpos'], d['qvel'], float(d['fps'])
        walk = [(i, n) for i, n in enumerate(names) if n.endswith('_walk')]
        if args.limit:
            walk = walk[:args.limit]
        n_pass = 0
        for i, name in walk:
            s, L = int(starts[i]), int(lengths[i])
            clip = dict(
                qpos=qpos[s:s + L], qvel=qvel[s:s + L])
            k = fk.fk_clip(clip['qpos'], clip['qvel'])
            m = gait_metrics(k['sole_h'], k['root_pos'], fps)
            ok, vals = audit_pass(m)
            m['pass'] = ok
            m['split'] = split
            report['clips'][name] = m
            n_pass += int(ok)
            f0, f1 = m['feet']
            print(f"{split:8s} {name:18s} {'PASS' if ok else 'FAIL'} "
                  f"cad {f0['cadence_hz']:.2f}/{f1['cadence_hz']:.2f} Hz "
                  f"v {m['speed_mps']:.2f} m/s "
                  f"step {f0['step_len_m']:.2f}/{f1['step_len_m']:.2f} m "
                  f"swing {f0['swing_s_med']:.2f}/{f1['swing_s_med']:.2f} s "
                  f"duty {f0['duty']:.2f}/{f1['duty']:.2f} "
                  f"lift {100 * max(f0['lift_m'], f1['lift_m']):.0f} cm")
        report['summary'][split] = dict(n_walk=len(walk), n_pass=n_pass)

    # distribution summary over passing train clips
    passing = [c for c in report['clips'].values() if c['pass'] and c['split'] == 'train']
    if passing:
        def stat(key):
            xs = []
            for c in passing:
                for f in c['feet']:
                    v = f[key]
                    if not np.isnan(v):
                        xs.append(v)
            if not xs:
                return None
            a = np.array(xs)
            return dict(n=len(a), p5=float(np.percentile(a, 5)), p50=float(np.median(a)),
                        p95=float(np.percentile(a, 95)))
        report['summary']['train_pass_distribution'] = {
            'cadence_hz': stat('cadence_hz'), 'speed_mps': None, 'step_len_m': stat('step_len_m'),
            'swing_s': stat('swing_s_med'), 'duty': stat('duty'), 'lift_m': stat('lift_m')}
        sp = [c['speed_mps'] for c in passing]
        report['summary']['train_pass_distribution']['speed_mps'] = dict(
            n=len(sp), p5=float(np.percentile(sp, 5)), p50=float(np.median(sp)),
            p95=float(np.percentile(sp, 95)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
