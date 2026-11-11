"""Evaluate every run listed in FREEZE.json on the official test set."""
import argparse
import json
from pathlib import Path

from .evaluate_test import evaluate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='paper_b/results/FREEZE.json')
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--raw-root', default='paper_b/results/raw')
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    if not man.get('frozen'):
        raise SystemExit('manifest is not frozen')
    n = 0
    for r in man['runs']:
        d = Path(a.raw_root) / r['tag'] / r['subset'] / f"s{int(r['seed'])}"
        if not (d / 'model.pt').exists():
            print('MISSING', d, flush=True)
            continue
        if (d / 'test_metrics.json').exists():
            print('SKIP', d, flush=True)
            continue
        evaluate_run(d, a.data_dir, a.device)
        n += 1
    print('evaluated', n)


if __name__ == '__main__':
    main()
