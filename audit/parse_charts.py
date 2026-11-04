#!/usr/bin/env python3
"""AUDIT: parse the gradmotion chart JSONs pulled into audit/logs/ and verify
the training-curve claims in docs/SMP_REPORT.md (S3/S10.2) plus the
termination-structure numbers (time_out vs base_too_low).

Run: python3 audit/parse_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / 'audit' / 'logs'


def load(task_prefix, key):
    f = L / f'{task_prefix}_{key.replace("/", "_")}.json'
    d = json.load(open(f))
    v = d[key]
    return np.array(v['steps']), np.array(v['values'], dtype=float)


def at(steps, vals, s):
    i = int(np.argmin(np.abs(steps - s)))
    return float(vals[i]), int(steps[i])


def span(tag, steps, vals, name):
    print(f'{tag} {name}: n={len(steps)} steps [{steps[0]}..{steps[-1]}] '
          f'v0={vals[0]:.4f} vend={vals[-1]:.4f}')


def main():
    for task, label in (('037', 'R1 v6 multiplicative (TASK_20261104_037)'),
                        ('208', 'R3 additive (TASK_20261104_208)')):
        print(f'===== {label} =====')
        s_rw, v_rw = load(task, 'Train/mean_reward')
        s_el, v_el = load(task, 'Train/mean_episode_length')
        s_ve, v_ve = load(task, 'Metrics/steering/error_vel_xy')
        s_bl, v_bl = load(task, 'Episode_Termination/base_too_low')
        s_to, v_to = load(task, 'Episode_Termination/time_out')
        span('  ', s_rw, v_rw, 'reward')
        span('  ', s_el, v_el, 'eplen')
        span('  ', s_ve, v_ve, 'vel_err')
        for it in (100, 471, 1000, 1500, 2000, 2900, 3000, 3500, 3900, 4000, 4100, 4500, 5000, 5141):
            if s_el[0] <= it <= s_el[-1]:
                rw, _ = at(s_rw, v_rw, it)
                el, si = at(s_el, v_el, it)
                ve, _ = at(s_ve, v_ve, it)
                bl, _ = at(s_bl, v_bl, it)
                to, _ = at(s_to, v_to, it)
                print(f'  iter {si:5d}: reward {rw:6.2f} eplen {el:7.1f} vel_err {ve:.3f} '
                      f'term_low {bl:.3f} term_timeout {to:.3f}')
        # last-1000-iters quartiles of vel_err (the "plateau" check)
        m = s_ve >= s_ve[-1] - 1000
        q = np.percentile(v_ve[m], [25, 50, 75, 100])
        print(f'  vel_err last-1000-iters quartiles: {q.round(3)} (n={m.sum()})')
        # max eplen & when first above 900
        i900 = np.argmax(v_el > 900) if (v_el > 900).any() else -1
        print(f'  first eplen>900 at iter {s_el[i900] if i900>=0 else "never"}; max eplen {v_el.max():.0f}')
        # termination balance in the last 500 iters
        m2 = s_to >= s_to[-1] - 500
        print(f'  last-500 time_out {np.median(v_to[m2]):.3f} vs base_too_low {np.median(v_bl[s_bl>=s_bl[-1]-500]):.3f}')


if __name__ == '__main__':
    main()
