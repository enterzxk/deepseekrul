"""Evaluate MSFB runs on the official test under the Paper B protocol."""
import argparse
import glob
import json
from pathlib import Path

# Importing train_msfb patches paper_b.training.train._build to build MSFB.
import paper_b.experimental_msfb.train_msfb  # noqa: F401
from paper_b.evaluation.evaluate_test import evaluate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--freeze', default='')
    a = ap.parse_args()
    runs = sorted(glob.glob(str(Path(a.root) / 'msfb' / '*' / 's*')))
    if a.freeze:
        manifest = dict(root=a.root, runs=runs, note='MSFB test freeze')
        Path(a.freeze).write_text(json.dumps(manifest, indent=1))
        print('froze', len(runs), 'runs to', a.freeze)
    rows = []
    for r in runs:
        if not (Path(r) / 'model.pt').exists():
            print('skip (no model.pt)', r)
            continue
        m = evaluate_run(r, data_dir=a.data_dir, device=a.device)
        rows.append(dict(run=r, subset=m['subset'], seed=m['seed'],
                         rmse=m['rmse'], nasa=m['nasa_sum'], n=m['n']))
    out = Path(a.root) / 'test_summary.json'
    out.write_text(json.dumps(rows, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
