#!/usr/bin/env python3
"""gm-run entry for P2: pretrain the TinyMDM motion prior on X1 walk windows.

The platform image ships CUDA torch; only light deps are pip-installed
(tyro). Runs scripts/pretrain_prior.py with the X1 configuration, streams
progress, and mirrors the final + best checkpoints to output/ (the channel
`gm task model list` scans).

  gm-run X1_SMP/scripts/gm_smp_prior.py            # full run
  python scripts/gm_smp_prior.py --dry-run         # print the plan only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--num-epochs', type=int, default=int(os.environ.get('SMP_PRIOR_EPOCHS', '20000')))
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--resume-from', default='', help='prior ckpt to resume from')
    args, extra = ap.parse_known_args()

    windows = ROOT / 'data' / 'windows'
    stats = ROOT / 'data' / 'prior' / 'norm_stats.npz'
    missing = [p for p in (windows, stats) if not p.exists()]
    if missing:
        print(f'FAIL: missing {missing}; build windows first (scripts/build_x1_windows.py)', flush=True)
        return 2
    n_npz = len(list(windows.glob('*.npz')))
    print(f'[prior] windows dir: {windows} ({n_npz} npz), stats: {stats}', flush=True)
    print(f'[prior] d_model={args.d_model} epochs={args.num_epochs} batch={args.batch_size} lr={args.lr}', flush=True)
    if args.dry_run:
        return 0

    print('[prior] pip install tyro (image torch is kept)', flush=True)
    installed = False
    for idx in ('', 'https://pypi.org/simple', 'https://mirrors.aliyun.com/pypi/simple/'):
        flag = ['--index-url', idx] if idx else []
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tyro', *flag],
                           capture_output=True, text=True)
        installed = r.returncode == 0
        print(f'pip tyro idx={idx or "default"}: {"OK" if installed else (r.stderr or "").strip().splitlines()[-1][:120]}', flush=True)
        if installed:
            break
    import torch  # noqa: F401  (after any installs)

    import threading
    def _watchdog():
        import subprocess as sp, time as _t
        while True:
            _t.sleep(60)
            try:
                gpu = sp.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
                ram = sp.run(['free', '-m'], capture_output=True, text=True).stdout.splitlines()[1]
                print(f'[watchdog] GPU {gpu} | RAM {ram}', flush=True)
            except Exception:
                pass
    threading.Thread(target=_watchdog, daemon=True).start()

    sys.path.insert(0, str(ROOT / 'src'))
    cmd = [sys.executable, str(ROOT / 'scripts' / 'pretrain_prior.py'),
           f'--data-dir={windows}', f'--norm-stats-file={stats}',
           f'--d-model={args.d_model}', f'--num-epochs={args.num_epochs}',
           f'--batch-size={args.batch_size}', f'--lr={args.lr}',
           '--no-use-ema', '--no-use-wandb', '--save-interval=250',
           '--train-split=0.95', '--name=x1_walk_prior'] + ([f'--resume-from={args.resume_from}'] if args.resume_from else []) + extra
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src') + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    print('[prior] exec:', ' '.join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    dt = (time.time() - t0) / 60
    print(f'[prior] pretrain exited {proc.returncode} after {dt:.1f} min', flush=True)

    # mirror checkpoints to output/ (outside the git tree, what the model channel scans)
    out_dir = Path('/workspace/output') if Path('/workspace').exists() else ROOT / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    saved = ROOT / 'logs' / 'pretrain' / 'x1_walk_prior'
    n_mirrored = 0
    if saved.exists():
        for ck in sorted(saved.glob('*.pt')):
            shutil.copy2(ck, out_dir / f'prior_{ck.name}')
            n_mirrored += 1
    print(f'[prior] mirrored {n_mirrored} checkpoints to {out_dir}', flush=True)
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
