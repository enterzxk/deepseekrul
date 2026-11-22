"""Generate run_queue.csv from grid.json.

The queue is the list of cells the runner is allowed to launch; runs.csv is the
append-only ledger filled at launch/close. Regenerate after editing grid.json:

    python docs/paper_c/registry/gen_queue.py

Cell count = len(variant_tags) * len(methods) * len(subsets) * len(seeds).
With the default single variant tag that is 5 x 2 x 3 = 30 cells.
"""
import csv
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['run_id', 'wave', 'variant_tag', 'base_model', 'subset', 'seed', 'span',
        'split_protocol', 'split_seed', 'baseline_version', 'baseline_ref',
        'cell_key', 'launch_allowed', 'note']


def main():
    g = json.load(io.open(os.path.join(HERE, 'grid.json'), encoding='utf-8'))
    rows = []
    for tag in g['variant_tags']:
        for model in g['methods']:
            for subset in g['subsets']:
                for seed in g['seeds']:
                    run_id = g['run_id_pattern'] \
                        .replace('<variant_tag>', tag) \
                        .replace('<base_model>', model) \
                        .replace('<subset>', subset) \
                        .replace('<seed>', str(seed))
                    rows.append(dict(
                        run_id=run_id,
                        wave=g['wave'],
                        variant_tag=tag,
                        base_model=model,
                        subset=subset,
                        seed=seed,
                        span=g['span'],
                        split_protocol=g['split_protocol'],
                        split_seed=g['split_seed'],
                        baseline_version=g['baseline_version'],
                        baseline_ref='%s/%s/%s/s%s' % (g['baseline_source_condition'], model, subset, seed),
                        cell_key='%s|%s|%s|%s' % (tag, model, subset, seed),
                        launch_allowed='false',
                        note='launch gated on P2_CROSS_ARCH_PREREG.md being committed',
                    ))
    out = os.path.join(HERE, 'run_queue.csv')
    with io.open(out, 'w', encoding='utf-8', newline='\n') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote %s: %d rows (%d variants x %d methods x %d subsets x %d seeds)'
          % (out, len(rows), len(g['variant_tags']), len(g['methods']),
             len(g['subsets']), len(g['seeds'])))
    if len({r['run_id'] for r in rows}) != len(rows):
        raise SystemExit('duplicate run_id generated')


if __name__ == '__main__':
    main()
