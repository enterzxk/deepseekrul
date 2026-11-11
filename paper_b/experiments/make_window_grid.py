"""Generate the window-search grid for the frozen final configuration.

Stage 1 (screening): L in [30,50,70,100,130], 4 subsets, seed 101.
Stage 2 (confirmation): the selected L, 4 subsets, seeds 101/202/303.
Selection uses validation RMSE only.
"""
import argparse
import json
from pathlib import Path

SPANS = [30, 50, 70, 100, 130]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-json', required=True, help='config.json of a completed final-model run')
    ap.add_argument('--tag-prefix', required=True)
    ap.add_argument('--out', default='paper_b/experiments/grids/window.json')
    ap.add_argument('--spans', default=','.join(map(str, SPANS)))
    ap.add_argument('--seeds', default='101')
    a = ap.parse_args()
    base = json.loads(Path(a.config_json).read_text())
    runs = []
    for span in [int(x) for x in a.spans.split(',')]:
        for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
            for seed in [int(x) for x in a.seeds.split(',')]:
                c = dict(base)
                c.update(span=span, subset=subset, seed=seed,
                         tag=f'{a.tag_prefix}_L{span}', epochs=40, patience=12)
                runs.append(c)
    Path(a.out).write_text(json.dumps(runs, indent=1))
    print('wrote', a.out, len(runs), 'runs')


if __name__ == '__main__':
    main()
