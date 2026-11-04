#!/usr/bin/env python3
"""AUDIT: evidence plots for docs/AUDIT_FAILURE.md.

  fig1: per-clip stance-slip (conservative min-corner metric) -- box/dots per
        clip, both feet, with the 0.15 m/s "planted" threshold
  fig2: 43_01_w0 frame-by-frame: sole z of both feet + pivot speed of the
        grounded foot (the frame-by-frame manual check, quantitative)
  fig3: prior slide-discrimination: intact vs slide-signature reward per clip
        (63-dim prior), plus the 59-dim comparison bars

Run: MPLCONFIGDIR=/tmp/mpl .venv/bin/python audit/make_plots.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / 'audit'


def fig_slip():
    d = json.load(open(A / 'slip_conservative.json'))
    names = list(d.keys())
    L = [d[n]['L']['slip_min_corner_p50'] for n in names]
    R = [d[n]['R']['slip_min_corner_p50'] for n in names]
    sl = [d[n]['L']['slide_frac'] for n in names]
    sr = [d[n]['R']['slide_frac'] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.18, L, 0.36, label='left foot', color='#7f9cf5')
    ax.bar(x + 0.18, R, 0.36, label='right foot', color='#f5a97f')
    ax.axhline(0.15, color='crimson', ls='--', lw=1, label='planted threshold 0.15 m/s')
    for i, (a, b) in enumerate(zip(sl, sr)):
        ax.text(i, max(L[i], R[i]) + 0.02, f'{a:.1f}/{b:.1f}', ha='center', fontsize=7, color='dimgray')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_walk', '').replace('_w0', '') for n in names], rotation=60, fontsize=7)
    ax.set_ylabel('contact-frame slip, slowest corner (m/s, p50)')
    ax.set_title('X1_SMP library: every clip slides during "contact" (labels: slide fraction L/R)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(A / 'fig1_stance_slip.png', dpi=130)


def fig_43_01():
    rows = list(csv.DictReader(open(A / 'frames' / '43_01_w0.csv')))
    zL = np.array([float(r['sole_z_L']) for r in rows])
    zR = np.array([float(r['sole_z_R']) for r in rows])
    vL = np.array([float(r['pivot_vx_L']) if r['pivot_vx_L'] else np.nan for r in rows])
    vR = np.array([float(r['pivot_vx_R']) if r['pivot_vx_R'] else np.nan for r in rows])
    t = np.arange(len(rows)) * 0.02
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    a1.plot(t, 1000 * zL, label='left sole z', color='#7f9cf5')
    a1.plot(t, 1000 * zR, label='right sole z', color='#f5a97f')
    a1.axhline(2.0, color='gray', ls=':', lw=1)
    a1.set_ylabel('sole height (mm)')
    a1.legend(fontsize=8)
    a1.set_title('43_01_w0 (the library\'s "only healthy >=0.7 m/s" clip): one foot glides at the grounding '
                 'constant (2 mm) while the other swings')
    a2.plot(t, vL, color='#7f9cf5', alpha=.7, label='left lowest-corner speed')
    a2.plot(t, vR, color='#f5a97f', alpha=.7, label='right lowest-corner speed')
    a2.axhline(0.15, color='crimson', ls='--', lw=1, label='0.15 m/s')
    a2.set_ylabel('|v| of lowest corner (m/s)')
    a2.set_xlabel('time (s)')
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(A / 'fig2_43_01_frames.png', dpi=130)


def fig_prior():
    d = json.load(open(A / 'prior_slide_test.json'))
    r3 = d['r3_63d']['per_clip']
    names = sorted(r3.keys())
    intact = [r3[n]['intact'] for n in names]
    slide = [r3[n].get('t3_slide', np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(x - 0.2, intact, 0.4, label='intact windows', color='#7f9cf5')
    ok = [s for s in slide if s == s]
    ax.bar(x[:len(ok)] + 0.2, ok, 0.4, label='slide-signature windows', color='#ed8796')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_walk', '').replace('_w0', '') for n in names], rotation=60, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('prior SDS reward exp(-6*err)')
    ax.set_title('63-dim contact prior: reward for the policy\'s slide signature vs real windows '
                 f'(library mean ratio {d["r3_63d"]["summary"]["ratio_t3"]:.2f})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(A / 'fig3_prior_slide.png', dpi=130)


if __name__ == '__main__':
    fig_slip()
    fig_43_01()
    fig_prior()
    print('-> audit/fig1_stance_slip.png fig2_43_01_frames.png fig3_prior_slide.png')
