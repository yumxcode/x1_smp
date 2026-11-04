#!/usr/bin/env python3
"""P5: gait acceptance + MuJoCo sim2sim of a trained X1 SMP policy.

Loads a policy checkpoint (mjlab/rsl-rl format), rolls it out in a headless
MuJoCo sim at commanded speeds 0.5 and 1.0 m/s for 20 s each, measures
rhythm_gates-compatible metrics (cadence/swing/duty/step-length/lift per
foot, survival, tracked speed), and renders >=2 videos.

Runs on the platform (needs the mjlab stack + checkpoint) OR locally with
mujoco only (policy ONNX/pt export required).

  gm-run X1_SMP/scripts/gm_smp_eval.py --ckpt /workspace/output/policy_model_xxx.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def install_stack() -> bool:
    for idx in ('', 'https://pypi.org/simple', 'https://mirrors.aliyun.com/pypi/simple/'):
        flag = ['--index-url', idx] if idx else []
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'mjlab==1.6.0',
                            'warp-lang==1.16.0', *flag], capture_output=True, text=True)
        if r.returncode == 0:
            print(f'pip OK idx={idx or "default"}', flush=True)
            return True
        print(f'pip fail idx={idx or "default"}: {(r.stderr or "?").strip()[-120:]}', flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(Path(__file__).resolve().parent.parent / 'data' / 'policy' / 'model_4100.pt'))
    ap.add_argument('--speeds', default='0.5,1.0')
    ap.add_argument('--secs', type=float, default=20.0)
    ap.add_argument('--out', default='eval')
    args, extra = ap.parse_known_args()

    if not Path(args.ckpt).exists():
        # fall back to the newest mirrored policy checkpoint
        import glob
        cands = sorted(glob.glob('/workspace/output/policy_*.pt'), key=lambda p: Path(p).stat().st_mtime)
        if not cands:
            print(f'FAIL: no checkpoint at {args.ckpt} nor /workspace/output/policy_*.pt', flush=True)
            return 2
        args.ckpt = cands[-1]
    print(f'[eval] ckpt={args.ckpt}', flush=True)
    if not install_stack():
        return 2

    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    import smp.rl.tasks  # noqa: F401
    # Play with the trained policy at fixed speeds using the registered task's
    # play env: we override the steering command to the fixed speed per run.
    speeds = [float(s) for s in args.speeds.split(',')]
    print(f'[eval] speeds={speeds} secs={args.secs}', flush=True)

    import eval_rollout as er  # sibling module with the actual rollout logic
    results = {}
    for v in speeds:
        r = er.rollout(args.ckpt, v, args.secs, tag=f'v{v:.1f}'.replace('.', 'p'))
        results[f'v{v}'] = r
        print(f'[eval] v={v}: {json.dumps(r)[:300]}', flush=True)

    out = ROOT / 'eval' / 'gait_metrics.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f'[eval] written {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
