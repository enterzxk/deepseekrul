"""Wait for the 3-seed raw table, then render the seed report.

The training/eval monitor (`tmp/monitor_s23.ps1`) writes
`paper_b/results/seed_robustness_raw.csv` by calling
`paper_b.experiments.seed_robustness`. This watcher only fires once that CSV
holds the full 3 cells x 4 subsets x 3 seeds grid, so it never renders a
half-downloaded state as if it were the result.

Exit codes: 0 = report written, 2 = timed out waiting.
"""
import subprocess
import sys
import time
from pathlib import Path

RAW = Path('paper_b/results/seed_robustness_raw.csv')
CELLS = {'F_single_proj_mean_hi0', 'F_single_proj_basic_hi1', 'F_full_basic_hi1'}
TARGET = 36
DEADLINE = time.time() + 100 * 60


def ready():
    if not RAW.exists():
        return False, 'no raw csv yet'
    try:
        import pandas as pd
    except ImportError:
        return False, 'pandas missing'
    try:
        df = pd.read_csv(RAW)
    except Exception as exc:  # partially written file
        return False, f'unreadable: {exc}'
    n = len(df)
    seeds = sorted(df.seed.unique().tolist()) if n else []
    tags = set(df.tag.unique().tolist()) if n else set()
    ok = n >= TARGET and tags == CELLS and seeds == [101, 202, 303]
    return ok, f'rows={n} seeds={seeds} cells={len(tags)}'


def main():
    last = None
    while time.time() < DEADLINE:
        ok, msg = ready()
        if msg != last:
            print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
            last = msg
        if ok:
            print('grid complete; rendering report', flush=True)
            rc = subprocess.call([sys.executable, '-m', 'paper_b.experiments.seed_report'])
            return 0 if rc == 0 else 1
        time.sleep(60)
    print('timeout waiting for the full 3-seed grid', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
