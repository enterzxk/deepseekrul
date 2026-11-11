"""Run a grid of Paper B experiments, idempotently and shardable.

Usage:
  python -m paper_b.experiments.run_grid --grid paper_b/experiments/grids/ablation8.json \
      --shard 0 --nshards 2 --device cuda:0 --data-dir ~/rulb/data/CMAPSS
"""
import argparse
import json
import traceback
from pathlib import Path

from ..training.train import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--out-root', default='paper_b/results/raw')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    runs = json.loads(Path(a.grid).read_text())
    runs = [r for i, r in enumerate(runs) if i % a.nshards == a.shard]
    runs.sort(key=lambda r: (r['subset'], r.get('span', 70), r['tag'], r['seed']))
    if a.limit:
        runs = runs[:a.limit]
    print(f'[grid] {a.grid} shard {a.shard}/{a.nshards}: {len(runs)} runs on {a.device}', flush=True)
    ok, skip, fail = 0, 0, 0
    for r in runs:
        out = Path(a.out_root) / r['tag'] / r['subset'] / f"s{r['seed']}" / 'metrics.json'
        if out.exists():
            skip += 1
            continue
        try:
            train_one({**r, 'data_dir': a.data_dir}, device=a.device, out_root=a.out_root)
            ok += 1
        except Exception as exc:
            fail += 1
            err = Path(a.out_root) / r['tag'] / r['subset'] / f"s{r['seed']}"
            err.mkdir(parents=True, exist_ok=True)
            (err / 'error.txt').write_text(traceback.format_exc())
            print(f'FAIL {r["tag"]} {r["subset"]} s{r["seed"]}: {exc}', flush=True)
    print(f'[grid] done ok={ok} skip={skip} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
