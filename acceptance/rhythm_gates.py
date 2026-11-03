#!/usr/bin/env python3
"""RHYTHM GATES (P8) — gait cadence/swing/duty/step-length vs HUMAN reference.

User directive (contract rev2): "增加节律门，步频、摆腿时长、占空比、步长
基于人类参考步态设置合适的阈值，并落入文档中，100hz 下训练的模型，摆腿
频率过大，导致步幅偏小。"

Human reference (measured from the demo dataset itself — the same clips the
discriminator imitates; acceptance/evidence/rhythm_reference.json, 2026-09-21,
walk family, jog clips excluded, 16 foot-series):

    cadence/foot   median 0.839 Hz   band [0.661, 1.064]   (lit. 0.85-1.0)
    swing duration median 0.429 s    band [0.223, 0.554]   (lit. 0.38-0.42)
    duty factor    median 0.611      band [0.390, 0.718]   (lit. 0.55-0.65)
    step length    0.71-0.92 m at 1.0-1.4 m/s overground clips
                   (36_01 0.92/0.71 @1.01 m/s, 103_07 0.79/0.79 @1.38 m/s)

Gates (thresholds = the human band relaxed ~15% for retarget/sim physics;
NOT relaxed to accommodate the policy family — the 100 Hz micro-cadence
defect is exactly what these gates exist to expose):

    R1 cadence/foot   0.55 - 1.20 Hz     (walk scenarios)
    R2 swing duration 0.28 - 0.60 s
    R3 duty factor    0.45 - 0.75
    R4 step length    0.45 - 1.05 m      (walk10 @ 1.0 m/s command)

Detection axis (both sides): the foot is SWINGING when its tracked point is
lifted > 25 mm above its own per-run minimum height, with stance gaps
< 60 ms glued and swings < 100 ms discarded.
  - policy side: rollout npz sole_xyz foot CENTER (mean of 4 corners) z;
  - reference side: demo key_body ankle-roll origin z (the datasets carry no
    sole corners; the lift-above-min convention removes the fixed geometric
    offset between the two points).

Usage:
    python rhythm_gates.py <rollout.npz> [--cmd-vel 1.0] [--json out.json]

Exit 0 = all rhythm gates pass; 1 = any fail. Scenario applicability:
R4 only meaningful on straight-walk logs with a forward command (default
1.0 m/s); stand/back05/turn logs should pass --cmd-vel 0 to skip R4.
"""
import argparse
import json
import sys

import numpy as np

LIFT_M = 0.025
GLUE_S = 0.06
MIN_SWING_S = 0.10

GATES = {
    "R1_cadence_hz": (0.55, 1.20),
    "R2_swing_s": (0.28, 0.60),
    "R3_duty": (0.45, 0.75),
    "R4_step_len_m": (0.45, 1.05),
}

# v71 L-gate (user directive 2026-10-17: 新增抬腿步幅要求; spec
# acceptance/X1_GAIT_SPEC_V71.md §5): per-swing APEX (max lift-above-min of
# the foot-center z within each detected swing). p95 >= 25 mm is structurally
# implied by the swing detector (lift > LIFT_M = 25 mm); the binding check
# is the MEDIAN apex >= 35 mm on BOTH feet (drag family reads 12-17 mm,
# healthy refs 22-63 mm).
L_GATE = {"L1_apex_med_m": (0.035, None), "L1_min_swings": (4, None)}


def swing_segments(z, dt):
    lift = z - z.min()
    on = lift > LIFT_M
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    glued = []
    for a, b in runs:
        if glued and a - glued[-1][1] < GLUE_S / dt:
            glued[-1] = (glued[-1][0], b)
        else:
            glued.append((a, b))
    return [g for g in glued if (g[1] - g[0]) * dt >= MIN_SWING_S]


def measure_foot(z, dt, xy=None):
    """z: tracked-point height series; xy: forward position (step length)."""
    segs = swing_segments(z, dt)
    if len(segs) < 3:
        return None
    sw = np.array([(b - a) * dt for a, b in segs])
    tds = np.array([a for a, b in segs])
    cyc = np.diff(tds) * dt
    lift = z - z.min()
    apex = np.array([float(lift[a:b].max()) for a, b in segs])
    out = {
        "n_swings": int(len(segs)),
        "swing_s_med": float(np.median(sw)),
        "stride_s_med": float(np.median(cyc)),
        "cadence_hz": float(1.0 / np.median(cyc)),
        "duty": float(1.0 - sw.sum() / (len(z) * dt)),
        "apex_med_m": float(np.median(apex)),
        "apex_p95_m": float(np.percentile(apex, 95)),
    }
    if xy is not None:
        sl = np.linalg.norm(xy[tds[1:]] - xy[tds[:-1]], axis=1)
        out["step_len_m_med"] = float(np.median(sl) / 2.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--cmd-vel", type=float, default=1.0,
                    help="forward command m/s; 0 disables R4 (step length)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    dt = float(meta["control_dt"])
    start = int(meta["settle_steps"]) + 2
    sole = z["sole_xyz"].astype(np.float64)          # (n,2,4,3)
    center = sole.mean(axis=2)                       # (n,2,3) foot centers

    results, fails = {}, []
    agg = {}
    for f, foot in enumerate(("L", "R")):
        m = measure_foot(center[start:, f, 2], dt,
                         xy=center[start:, f, :2] if args.cmd_vel > 0 else None)
        results[foot] = m
        if m is None:
            fails.append(f"{foot}:insufficient_swings")
            continue
        for k in agg_keys(m):
            agg.setdefault(k, []).append(m[k])

    checks = []
    med = {k: float(np.median(v)) for k, v in agg.items()}
    for gate, (lo, hi) in GATES.items():
        key = {"R1_cadence_hz": "cadence_hz", "R2_swing_s": "swing_s_med",
               "R3_duty": "duty", "R4_step_len_m": "step_len_m_med"}[gate]
        if key not in med:
            if gate == "R4_step_len_m" and args.cmd_vel <= 0:
                checks.append((gate, None, None, True, "skipped (cmd-vel 0)"))
                continue
            checks.append((gate, None, None, False, "no data"))
            fails.append(gate)
            continue
        val = med[key]
        ok = lo <= val <= hi
        if not ok:
            fails.append(gate)
        checks.append((gate, val, (lo, hi), ok, ""))
    # v71 L-gate (per-foot, both must pass; median aggregated per foot first)
    for gate, (lo, hi) in L_GATE.items():
        vals = [results[f] for f in ("L", "R") if results[f] is not None]
        if gate == "L1_min_swings":
            ok = all(v["n_swings"] >= lo for v in vals) and len(vals) == 2
            val = min((v["n_swings"] for v in vals), default=None)
            checks.append((gate, val, (lo, None), ok, "per-foot min"))
        else:
            ok = all(v["apex_med_m"] >= lo for v in vals) and len(vals) == 2
            val = min((v["apex_med_m"] for v in vals), default=None)
            checks.append((gate, val, (lo, None), ok, "per-foot min"))
        if not ok:
            fails.append(gate)

    print("=" * 72)
    print(f"RHYTHM GATES (P8)  {args.npz.split('/')[-1]}  dt={dt:.3f} "
          f"cmd_vel={args.cmd_vel:g}")
    print("=" * 72)
    for f, m in results.items():
        if m is None:
            print(f"  {f}: insufficient swings")
            continue
        extra = f" step {m.get('step_len_m_med', float('nan')):.2f}m" \
            if "step_len_m_med" in m else ""
        print(f"  {f}: cadence {m['cadence_hz']:.2f} Hz  swing "
              f"{m['swing_s_med']:.2f} s  duty {m['duty']:.2f}  "
              f"apex med/p95 {m['apex_med_m']*1000:.0f}/"
              f"{m['apex_p95_m']*1000:.0f} mm{extra}")
    for gate, val, band, ok, note in checks:
        v = f"{val:.3f}" if val is not None else "—"
        b = (f"[{band[0]:.2f}, {band[1]:.2f}]" if band[1] is not None
             else f">= {band[0]:.3f}") if band else ""
        print(f"  {'ok  ' if ok else 'FAIL'} {gate:16s} {v:>8s} in {b} {note}")
    verdict = "PASS" if not fails else "FAIL"
    print(f"VERDICT: {verdict}  fails={fails}")
    if args.json:
        json.dump({"file": args.npz, "cmd_vel": args.cmd_vel,
                   "feet": results, "median": med,
                   "checks": [{"gate": g, "value": v, "band": b, "pass": ok}
                              for g, v, b, ok, _ in checks],
                   "verdict": verdict, "fails": fails},
                  open(args.json, "w"), indent=1)
        print(f"[JSON] {args.json}")
    return 0 if not fails else 1


def agg_keys(m):
    return [k for k in ("cadence_hz", "swing_s_med", "duty", "step_len_m_med")
            if k in m]


if __name__ == "__main__":
    sys.exit(main())
