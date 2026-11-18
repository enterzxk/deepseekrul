"""Split-43 diagnostic: does the mid-life late bias survive without selection?

Reads the frozen truncation table (unit -> target RUL / truncation cycle) and,
for every split-43 cell, builds the diagnostic windows at those exact cycles,
then scores the SAME run twice:

  * `best_state` -- the checkpoint early stopping would have kept (selection ON)
  * `model`      -- the weights at the last epoch (selection OFF, because
                    patience >= epochs means early stopping never fired)

The difference between the two is the direct measurement of how much
checkpoint selection suppresses the tail bias.

Caliber, fixed by DIAG_SPLIT_SPEC.md:
  * target `y = min(max(L - c, 0), 125)` -- the same rule make_val_windows uses
  * error is per engine, averaged over seeds, then aggregated
  * main criterion is the MERGED 75-100 bucket (single-subset tail n is 4-10,
    so a single engine can flip a subset; single-subset numbers are reported
    for the direction count only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from _cells import resolve                                    # noqa: E402
from paper_b.training.data import (_prepare_split, _window,    # noqa: E402
                                   _baseline_ref, CYCLE_SCALE)
from paper_b.training.train import _build, predict             # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)
SPLIT = 43
TAIL = (75.0, 100.0)


def build_diag_windows(s, span, table, max_rul=125.0):
    """Same field layout as make_val_windows, but at explicit cycles per unit."""
    frame, values, clusters, pop_ref = s['va'], s['va_s'], s['lv'], s['pop']
    xs, ys, ms, bs, cs, cds, us, cts, css = [], [], [], [], [], [], [], [], []
    for uid, g in frame.groupby('unit', sort=True):
        uid = int(uid)
        if uid not in table:
            raise SystemExit(f'unit {uid} missing from the frozen truncation table')
        c = int(table[uid])
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], np.float32)
        cl = np.asarray(clusters[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        n = len(seq)
        if not (1 <= c <= n):
            raise SystemExit(f'unit {uid}: truncation cycle {c} outside 1..{n}')
        t = c - 1
        x, mask = _window(seq, t + 1, span)
        cw = np.zeros(span, np.int64)
        mm = min(t + 1, span)
        cw[span - mm:] = cl[t + 1 - mm:t + 1]
        xs.append(x); ms.append(mask); css.append(cw)
        ys.append(float(min(max(n - c, 0), max_rul)))
        bs.append(_baseline_ref(seq, cl, cyc, t, pop_ref))
        cs.append(min(cyc[t] / CYCLE_SCALE, 1.0)); cds.append(int(cl[t]))
        us.append(uid); cts.append(int(c))
    return dict(x=np.stack(xs), y=np.asarray(ys, np.float32), mask=np.stack(ms),
                baseline=np.stack(bs), cycle=np.asarray(cs, np.float32),
                cond=np.asarray(cds, np.int64), cond_seq=np.stack(css),
                unit=np.asarray(us), cut=np.asarray(cts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--table', required=True,
                    help='frozen truncation table: {subset: {unit: truncation_cycle}}')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root, data_dir = Path(a.root), Path(a.data_dir)
    frozen = json.loads(Path(a.table).read_text())['subsets']

    per = {}
    trace = []
    for subset in SUBSETS:
        rows = frozen[subset]['rows']
        table = {int(r['unit']): int(r['truncation_cycle']) for r in rows}
        if len(table) != len(rows):
            raise SystemExit(f'{subset}: duplicate units in the frozen table')
        for r in rows:
            if int(r['life']) - int(r['truncation_cycle']) != int(r['target_rul']):
                raise SystemExit(
                    f'{subset} unit {r["unit"]}: life - cycle != target_rul '
                    f'({r["life"]} - {r["truncation_cycle"]} != {r["target_rul"]})')
        print(f'{subset}: {len(table)} diagnostic units, '
              f'buckets {frozen[subset].get("bucket_counts")}, '
              f'fallback {sum(1 for r in rows if r.get("fallback"))}')
        err = {'best_state': [], 'model': []}
        units = None
        for seed in SEEDS:
            cell = Path(resolve(root, 'E0', subset, seed, SPLIT))
            cfg = json.loads((cell / 'config.json').read_text())
            ck = torch.load(cell / 'last.pt', map_location='cpu', weights_only=False)
            run_ep = json.loads((cell / 'metrics.json').read_text()).get('epochs_run')
            # The whole diagnosis rests on `model` being an UNSELECTED checkpoint, and
            # that only holds if early stopping never fired. Assert it here rather than
            # letting the reader trust a number this script merely collected and dropped.
            want_ep = int(cfg['epochs'])
            if run_ep != want_ep:
                raise SystemExit(
                    f'{cell}: epochs_run={run_ep} != epochs={want_ep}; early stopping fired, '
                    f'so `model` is NOT selection-free and this diagnosis is void')
            trace.append(dict(subset=subset, seed=seed, best_ep=ck.get('best_ep'),
                              epochs_run=run_ep))
            s = _prepare_split(subset, data_dir, SPLIT, cfg['fold'], cfg['val_frac'],
                               cfg['n_cond'])
            d = build_diag_windows(s, cfg['span'], table)
            if units is None:
                units = d['unit']
            elif not np.array_equal(units, d['unit']):
                raise SystemExit(f'{subset}: seeds disagree on the diagnostic units')
            net = _build(cfg)
            for key in ('best_state', 'model'):
                if key not in ck:
                    raise SystemExit(f'{cell}: last.pt has no {key!r}')
                net.load_state_dict(ck[key])
                net.eval()
                p = predict(net, d, 'cpu', cfg['mode'], fp32=True)
                err[key].append(p - d['y'])
        m = {k: np.mean(np.stack(v), axis=0) for k, v in err.items()}
        y = d['y']
        tail = (y >= TAIL[0]) & (y < TAIL[1])
        per[subset] = dict(
            n=int(len(y)), n_tail=int(tail.sum()),
            bias_best=float(m['best_state'][tail].mean()) if tail.any() else None,
            bias_last=float(m['model'][tail].mean()) if tail.any() else None,
            mae_best=float(np.abs(m['best_state'][tail]).mean()) if tail.any() else None,
            mae_last=float(np.abs(m['model'][tail]).mean()) if tail.any() else None,
        )
        print(f'{subset} n={per[subset]["n"]} tail={per[subset]["n_tail"]} '
              f'bias best {per[subset]["bias_best"]:+.2f} / last {per[subset]["bias_last"]:+.2f}',
              flush=True)

    def merged(key):
        vals, weights = [], []
        for s in SUBSETS:
            v = per[s][key]
            if v is None:
                continue
            vals.append(v); weights.append(per[s]['n_tail'])
        return float(np.average(vals, weights=weights)), len(vals)

    b_best, k = merged('bias_best')
    b_last, _ = merged('bias_last')
    pos_best = sum(1 for s in SUBSETS if per[s]['bias_best'] is not None and per[s]['bias_best'] > 0)
    pos_last = sum(1 for s in SUBSETS if per[s]['bias_last'] is not None and per[s]['bias_last'] > 0)
    print(f'\nMERGED tail bias  best_state {b_best:+.2f} ({pos_best}/4 positive)   '
          f'model {b_last:+.2f} ({pos_last}/4 positive)')
    print(f'selection suppression (best - last) = {b_best - b_last:+.2f}')
    verdict = (b_last >= 2.0 and pos_last >= 3)
    print(f'criterion (model bias >= +2.0 and >=3/4 positive): '
          f'{"PASS" if verdict else "FAIL"}')
    eps = sorted({t['epochs_run'] for t in trace})
    print(f'epochs_run values: {eps}  (must be a single value == epochs; '
          f'asserted per cell)')
    print(f'tail n per subset: ' +
          ', '.join(f'{s}={per[s]["n_tail"]}' for s in SUBSETS) +
          '  -- single-subset tail n is small; the merged number is the criterion')

    bad = [t for t in trace if t['epochs_run'] != 40]
    differ = sum(1 for t in trace if (t['best_ep'] or 0) < 40)
    print(f'epochs_run: {len(trace)} cells, all 40 = {not bad}; '
          f'best_ep < 40 in {differ}/{len(trace)} cells -> the last-epoch model is '
          f'NOT best_state in those cells')

    if a.json:
        Path(a.json).write_text(json.dumps(dict(
            split=SPLIT, per_subset=per, merged_bias_best=b_best,
            merged_bias_last=b_last, positive_best=pos_best, positive_last=pos_last,
            suppression=b_best - b_last, verdict='PASS' if verdict else 'FAIL',
            epochs_run_check='all == epochs (asserted)',
            trace=trace,
        ), indent=1), encoding='utf-8')
        print(f'wrote {a.json}')


if __name__ == '__main__':
    raise SystemExit(main())
