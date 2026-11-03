#!/usr/bin/env python3
"""Slice the X1 motion library into 59-dim SMP windows + normalization stats.

Input : data/x1_smp_train.npz (or any library npz with qpos/qvel/starts/
        lengths/names/fps, 36-col qpos = free7 + 29 joints, 50 Hz)
Output: data/windows/<name>.npz  each with `windows` (N, 10, 59) float32
        (layout of smp.rl.utils.MotionFeatureBuffer.compute_features, built by
        scripts/x1_data_lib.compute_windows via MuJoCo FK)
        data/prior/norm_stats.npz  q_low/q_high per feature dim (q01/q99)

Storage red line: only the windowed npz set + norm stats + a small meta json
are kept; nothing else is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from x1_data_lib import X1FK, compute_windows, FEATURE_DIM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', type=Path, default=HERE.parent / 'data' / 'x1_smp_train.npz')
    ap.add_argument('--out-dir', type=Path, default=HERE.parent / 'data' / 'windows')
    ap.add_argument('--stats', type=Path, default=HERE.parent / 'data' / 'prior' / 'norm_stats.npz')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--q-low', type=float, default=0.01)
    ap.add_argument('--q-high', type=float, default=0.99)
    ap.add_argument('--heldout-every', type=int, default=8,
                    help='every N-th clip is excluded from the stats (prior sanity eval)')
    args = ap.parse_args()

    fk = X1FK()
    d = np.load(args.lib, allow_pickle=True)
    names = [str(n) for n in d['names']]
    starts, lengths, qpos, qvel, fps = d['starts'], d['lengths'], d['qpos'], d['qvel'], float(d['fps'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    all_frames, n_windows, kept = [], 0, []
    for i, name in enumerate(names):
        s, L = int(starts[i]), int(lengths[i])
        k = fk.fk_clip(qpos[s:s + L], qvel[s:s + L])
        w = compute_windows(k, window_size=args.window, stride=args.stride)
        if w is None or len(w) == 0:
            print(f'{name}: too short, skipped')
            continue
        np.savez(args.out_dir / f'{name}.npz', windows=w)
        n_windows += len(w)
        kept.append(name)
        if i % args.heldout_every != 0:  # stats exclude held-out clips
            all_frames.append(w.reshape(-1, FEATURE_DIM))
    frames = np.concatenate(all_frames, axis=0)
    q_low = np.percentile(frames, args.q_low * 100, axis=0)
    q_high = np.percentile(frames, args.q_high * 100, axis=0)
    np.savez(args.stats, q_low=q_low.astype(np.float32), q_high=q_high.astype(np.float32))
    meta = dict(library=str(args.lib), clips=len(kept), windows=n_windows,
                window_size=args.window, stride=args.stride, feature_dim=FEATURE_DIM,
                fps=fps, q_low=args.q_low, q_high=args.q_high,
                stats_frames=int(len(frames)), kept=kept,
                heldout=[n for j, n in enumerate(names) if j % args.heldout_every == 0 and n in kept])
    (args.out_dir / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(kept)} clips -> {n_windows} windows of ({args.window},{FEATURE_DIM}); '
          f'stats from {len(frames)} frames -> {args.stats}')


if __name__ == '__main__':
    main()
