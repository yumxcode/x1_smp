#!/usr/bin/env python3
"""CMU SMPL-H walking-clip selection + stageii-schema conversion (v72 data route).

Input : data/amass/CMU_raw/*_poses.npz   (AMASS SMPL-H flavor from the
        realdream-ai/AMASS per-file mirror: poses(N,156), trans, betas,
        dmpls, gender, mocap_framerate)
Output: data/amass/CMU/<name>_stageii.npz  (stageii SMPL-X schema that
        GMR's load_smplx_file consumes: root_orient(N,3), pose_body(N,63),
        trans(N,3), betas(10,), gender, mocap_frame_rate) — written ONLY
        for the selected steady-straight-walking window of qualifying clips.

Selection metrics (human scale, from trans + root yaw + pelvis sway FFT):
  speed     = median horizontal speed in heading frame
  straight  = net / travel
  cadence   = steps/min (both feet) = 60 * vertical-oscillation Hz
              (pelvis bobs twice per gait cycle = once per step)
  step_len  = speed / step_freq
Targets (Froude-scaled from X1 1.0 m/s neighborhood, audit 2026-10-22):
  human cadence 95-140 spm | speed 0.7-1.9 m/s | step 0.50-0.85 m
  window >= 6 s straight overground walking (no start/stop)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "amass" / "CMU_raw"
OUT = ROOT / "data" / "amass" / "CMU"

TGT = dict(cad_lo=95, cad_hi=140, v_lo=0.7, v_hi=1.9, s_lo=0.50, s_hi=0.85,
           min_win=6.0)


def yaw_from_aa(aa):
    """axis-angle (N,3) -> unwrapped yaw."""
    ang = np.linalg.norm(aa, axis=1)
    axis = np.where(ang[:, None] > 1e-8, aa / np.maximum(ang[:, None], 1e-8), 0.0)
    axis = np.where(ang[:, None] > 1e-8, axis, np.array([0.0, 0.0, 1.0]))
    k = axis
    K = np.stack([np.zeros_like(k[:, 0]), -k[:, 2], k[:, 1],
                  k[:, 2], np.zeros_like(k[:, 0]), -k[:, 0],
                  -k[:, 1], k[:, 0], np.zeros_like(k[:, 0])], axis=1).reshape(-1, 3, 3)
    I = np.eye(3)[None]
    A = np.zeros_like(K)
    A[:, 0, 1], A[:, 0, 2], A[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    A[:, 1, 0], A[:, 2, 0], A[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    sin, cos = np.sin(ang), 1 - np.cos(ang)
    K2 = np.einsum("nij,njk->nik", K, K)
    R = I + sin[:, None, None] * K + cos[:, None, None] * K2
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.unwrap(yaw)


def dominant_freq(x, fps, f_lo, f_hi):
    """FFT dominant frequency of a detrended signal within [f_lo, f_hi] Hz."""
    x = x - x.mean()
    n = len(x)
    if n < 32:
        return 0.0, 0.0
    w = np.hanning(n)
    F = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        return 0.0, 0.0
    idx = np.where(band)[0]
    j = idx[np.argmax(F[idx])]
    # parabolic interpolation for sub-bin accuracy
    if 0 < j < len(F) - 1:
        a, b, c = F[j - 1], F[j], F[j + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        f = freqs[j] + delta * (freqs[1] - freqs[0])
    else:
        f = freqs[j]
    power = F[j] ** 2 / max(np.sum(F ** 2), 1e-12)
    return float(f), float(power)


def velocity_heading(trans, fps, smooth_s=1.0):
    """Convention-free heading: smoothed displacement direction (unwrapped).
    Robust to the SMPL root_orient body-facing offset (CMU fits ~90 deg off
    the travel direction)."""
    k = max(1, int(smooth_s * fps / 2))
    n = len(trans)
    dx = np.zeros(n)
    dy = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        dx[i] = trans[b - 1, 0] - trans[a, 0]
        dy[i] = trans[b - 1, 1] - trans[a, 1]
    h = np.arctan2(dy, dx)
    return np.unwrap(h)


def window_metrics(wx, wy, hy, z, fps, i, j):
    """Metrics for frame window [i, j). World coords for distances,
    heading-frame y for sway/cadence FFT."""
    xx, yy = wx[i:j], wy[i:j]
    d = np.hypot(np.diff(xx), np.diff(yy))
    hy = hy[i:j]
    travel, net = d.sum(), math.hypot(xx[-1] - xx[0], yy[-1] - yy[0])
    if travel < 0.5:
        return None
    v = float(np.median(d) * fps)
    # vertical bob = step frequency; lateral sway = step freq / 2
    fz, pz = dominant_freq(z[i:j], fps, 0.5, 3.5)
    fy, py = dominant_freq(hy, fps, 0.25, 1.75)
    # consistency: fz should be ~2*fy
    step_hz = fz if pz > 0.05 else 2 * fy
    cad = 60 * step_hz
    sl = v / step_hz if step_hz > 0.3 else 0.0
    return dict(v=v, straight=net / travel, cad=cad, step=sl, step_hz=step_hz,
                fz=fz, fy=fy, pz=pz, py=py, dur=(j - i) / fps,
                sway_amp=float(np.ptp(hy - np.linspace(hy[0], hy[-1], len(hy))))
                if len(hy) else 0.0)


def find_best_window(wx, wy, y, z, fps, want=None):
    """Slide 8-s windows, score by target closeness; return best (i, j, m)."""
    N = len(wx)
    w = int(min(8.0 * fps, N))
    best = None
    for i in range(0, N - w + 1, max(1, int(0.5 * fps))):
        j = i + w
        m = window_metrics(wx, wy, y, z, fps, i, j)
        if m is None:
            continue
        if want:
            ok = (TGT["cad_lo"] <= m["cad"] <= TGT["cad_hi"]
                  and TGT["v_lo"] <= m["v"] <= TGT["v_hi"]
                  and TGT["s_lo"] <= m["step"] <= TGT["s_hi"]
                  and m["straight"] >= 0.85)
            if not ok:
                continue
        score = (abs(m["v"] - 1.3) + 2 * abs(m["cad"] - 118) / 118
                 + abs(m["step"] - 0.65) / 0.65 + (1 - m["straight"]))
        if best is None or score < best[0]:
            best = (score, i, j, m)
    return best


def convert(npz_path, i, j, out_path):
    d = np.load(npz_path, allow_pickle=True)
    poses = np.asarray(d["poses"], dtype=np.float32)[i:j]
    trans = np.asarray(d["trans"], dtype=np.float32)[i:j].copy()
    trans[:, :2] -= trans[0, :2]
    betas = np.asarray(d["betas"], dtype=np.float32).reshape(-1)[:10]
    gender = str(np.asarray(d["gender"]).item())
    fps = float(np.asarray(d["mocap_framerate"]).item())
    root_orient = poses[:, :3]
    pose_body = poses[:, 3:66]
    np.savez(out_path,
             root_orient=root_orient.astype(np.float32),
             pose_body=pose_body.astype(np.float32),
             trans=trans.astype(np.float32),
             betas=betas.astype(np.float32),
             gender=np.array(gender),
             mocap_frame_rate=np.array(fps),
             source_frames=np.array([i, j]))
    return fps, j - i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true",
                    help="write stageii npz for qualifying clips")
    ap.add_argument("--all", action="store_true", help="report all clips")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RAW.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        try:
            poses = np.asarray(d["poses"], dtype=np.float32)
            trans = np.asarray(d["trans"], dtype=np.float32)
            fps = float(np.asarray(d["mocap_framerate"]).item())
        except Exception as e:
            print(f"{p.stem}: SKIP {e}")
            continue
        if poses.ndim != 2 or poses.shape[1] < 66 or len(poses) < 120:
            continue
        h = velocity_heading(trans, fps)
        y = -np.sin(h) * trans[:, 0] + np.cos(h) * trans[:, 1]
        z = trans[:, 2]
        bw = find_best_window(trans[:, 0], trans[:, 1], y, z, fps, want=True)
        any_w = bw if bw else find_best_window(trans[:, 0], trans[:, 1], y, z,
                                               fps, want=False)
        row = dict(name=p.stem.replace("_poses", ""),
                   fps=fps, dur=len(poses) / fps)
        if any_w:
            m = any_w[3]
            row.update({k: round(v, 3) for k, v in m.items()},
                       i=any_w[1], j=any_w[2], qualify=bool(bw))
        rows.append(row)

    print(f"{'clip':10s} {'dur':>5s} {'v':>5s} {'spm':>5s} {'step':>5s} "
          f"{'str':>5s} {'swy':>5s} {'qual':>4s}")
    for r in sorted(rows, key=lambda r: (not r.get("qualify", False), r.get("cad", 0))):
        if not args.all and not r.get("qualify"):
            continue
        print(f"{r['name']:10s} {r['dur']:5.1f} {r.get('v', 0):5.2f} "
              f"{r.get('cad', 0):5.0f} {r.get('step', 0):5.2f} "
              f"{r.get('straight', 0):5.2f} {r.get('sway_amp', 0):5.2f} "
              f"{'Y' if r.get('qualify') else '-':>4s}")

    if args.convert:
        n = 0
        for r in rows:
            if r.get("qualify"):
                src = RAW / f"{r['name']}.npz"
                dst = OUT / f"{r['name']}_stageii.npz"
                fps, nf = convert(src, r["i"], r["j"], dst)
                print(f"[CONVERT] {r['name']}: frames {r['i']}:{r['j']} "
                      f"({nf} @ {fps:.0f}fps) -> {dst.name}")
                n += 1
        print(f"[DONE] converted {n} clips")
    (ROOT / "data" / "amass" / "cmu_scan_report.json").write_text(
        json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
