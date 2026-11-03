#!/usr/bin/env python3
"""Rate-limited downloader for AMASS mirror files with size validation."""
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / 'data' / 'raw'


def fetch(url: str) -> tuple[str, bool]:
    import subprocess
    name = url.split('/')[-1].replace('_poses.npz', '.npz')
    out = RAW / name
    for attempt in range(4):
        try:
            sz = out.stat().st_size if out.exists() else 0
            if sz > 100_000:
                return name, True
            subprocess.run(['curl', '-sL', '-m', '120', '-o', str(out), url], check=True)
            if out.stat().st_size > 100_000:
                return name, True
        except Exception as e:  # noqa: BLE001
            print(f'{name}: attempt {attempt+1} failed: {e}', file=sys.stderr)
        time.sleep(0.5 + attempt)
    return name, False


if __name__ == '__main__':
    urls = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    RAW.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, good in ex.map(fetch, urls):
            ok += good
            if not good:
                print(f'FAILED {name}', file=sys.stderr)
    print(f'downloaded OK: {ok}/{len(urls)}')
