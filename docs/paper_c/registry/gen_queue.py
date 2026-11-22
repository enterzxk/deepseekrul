"""Generate run_queue.csv from grid.json.

    python docs/paper_c/registry/gen_queue.py

Cell count = len(arms) * len(variant_tags) * len(methods) * len(subsets) * len(seeds).
With two arms (v0, cadf) and one variant tag that is 2 * 5 * 2 * 3 = 60 cells:
segment A = 30 V0 reproductions, segment B = 30 +CADF.

`launch_allowed` is computed, never hand-written:
  segment A opens when the prereg and the frozen tree are both committed
  segment B additionally needs a_segment_reconciled=true, which is a data change
  in grid.json (auditable in git), not an edit of the queue.

If runs.csv has no data rows this script also (re)writes its header from
schema_cols.COLS. A populated ledger is never touched.
"""
import csv
import io
import json
import math
import os

import schema_cols

HERE = os.path.dirname(os.path.abspath(__file__))

QUEUE_COLS = ['run_id', 'segment', 'arm', 'wave', 'variant_tag', 'base_model',
              'subset', 'seed', 'span', 'k0_mode', 'k0_resolved',
              'split_protocol', 'split_seed', 'epochs', 'patience',
              'baseline_version', 'baseline_source', 'baseline_ref',
              'cell_key', 'launch_allowed', 'launch_gate', 'note']


def k0_for(grid, variant, window):
    if variant not in grid['k0_mode_by_variant']:
        return 'none', ''          # the v0 arm has no CADF at all
    mode = grid['k0_mode_by_variant'][variant]
    if mode == 'prop':
        return 'prop', str(int(math.ceil(grid['k0_prop_fraction'] * window)))
    if mode == 'abs':
        return 'abs', str(grid['k0_abs_value'])
    return 'none', ''


def main():
    g = json.load(io.open(os.path.join(HERE, 'grid.json'), encoding='utf-8'))
    arms = {a['arm']: a for a in g['arms']}
    v0 = arms['v0']
    gates = []
    if not g['prereg'].get('commit'):
        gates.append('prereg_commit')
    if not g.get('freeze_commit'):
        gates.append('freeze_commit')

    rows = []
    for arm in g['arms']:
        variants = g['variant_tags'] if arm['arm'] == 'cadf' else [arm['cadf_variant']]
        for variant in variants:
            for model in g['methods']:
                for subset in g['subsets']:
                    window = g['windows'][model][subset]
                    k0_mode, k0_resolved = k0_for(g, variant, window)
                    for seed in g['seeds']:
                        run_id = arm['run_id_pattern'] \
                            .replace('<variant_tag>', variant) \
                            .replace('<base_model>', model) \
                            .replace('<subset>', subset) \
                            .replace('<seed>', str(seed))
                        if arm['baseline_source'] == 'frozen_ledger':
                            ref = '%s/%s/%s/s%s' % (g['baseline_source_condition'], model, subset, seed)
                            note = 'segment A: reproduce frozen %s cell %s' % (g['baseline_source_condition'], ref)
                        else:
                            ref = v0['run_id_pattern'] \
                                .replace('<base_model>', model) \
                                .replace('<subset>', subset) \
                                .replace('<seed>', str(seed))
                            note = 'segment B: pairs against this batch\'s segment-A cell %s' % ref
                        seg_gates = list(gates)
                        if arm['segment'] == 'B' and not g.get('a_segment_reconciled'):
                            seg_gates.append('a_segment_reconciled')
                        rows.append(dict(
                            run_id=run_id, segment=arm['segment'], arm=arm['arm'],
                            wave=g['wave'], variant_tag=variant, base_model=model,
                            subset=subset, seed=seed, span=window,
                            k0_mode=k0_mode, k0_resolved=k0_resolved,
                            split_protocol=g['split_protocol'], split_seed=g['split_seed'],
                            epochs=g['epochs'], patience=g['patience'],
                            baseline_version=g['baseline_version'],
                            baseline_source=arm['baseline_source'], baseline_ref=ref,
                            cell_key='%s|%s|%s|%s|%s' % (arm['arm'], variant, model, subset, seed),
                            launch_allowed='false' if seg_gates else 'true',
                            launch_gate=','.join(seg_gates) if seg_gates else 'open',
                            note=note,
                        ))

    out = os.path.join(HERE, 'run_queue.csv')
    with io.open(out, 'w', encoding='utf-8', newline='\n') as fh:
        w = csv.DictWriter(fh, fieldnames=QUEUE_COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    runs_path = os.path.join(HERE, 'runs.csv')
    n_data = 0
    if os.path.exists(runs_path):
        with io.open(runs_path, encoding='utf-8', newline='') as fh:
            n_data = sum(1 for _ in csv.reader(fh)) - 1
    if n_data <= 0:
        with io.open(runs_path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(','.join(schema_cols.COLS) + '\n')
        header_note = 'header rewritten (%d cols)' % len(schema_cols.COLS)
    else:
        header_note = 'runs.csv kept as-is (%d data rows)' % n_data

    per_seg = {}
    for r in rows:
        per_seg[r['segment']] = per_seg.get(r['segment'], 0) + 1
    print('wrote %s: %d rows (%d arms x %d variants x %d methods x %d subsets x %d seeds)'
          % (out, len(rows), len(g['arms']), len(g['variant_tags']), len(g['methods']),
             len(g['subsets']), len(g['seeds'])))
    print('  segments: %s' % ', '.join('%s=%d' % (k, per_seg[k]) for k in sorted(per_seg)))
    print('  launch gates still closed: %s' % (', '.join(gates) or 'none'))
    print('  %s' % header_note)
    if len({r['run_id'] for r in rows}) != len(rows):
        raise SystemExit('duplicate run_id generated')
    if len(rows) != len(g['arms']) * len(g['variant_tags']) * len(g['methods']) \
            * len(g['subsets']) * len(g['seeds']):
        raise SystemExit('row count does not match the declared grid')


if __name__ == '__main__':
    main()
