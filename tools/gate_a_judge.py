#!/usr/bin/env python3
"""Gate A judge — computes G1/G2/G2'/G3/G4/G5 directly from run artifacts.

Reads only ``close.json`` / ``metrics.json`` / ``preds.csv`` under the run root.
It never reads the registry ledger, so a concurrent edit of the ledger cannot
change a verdict.

Sign convention (pre-registered, GATE_A_WAVE1_PLAN.md §9):

    delta(A -> B) = (A - B) / A        positive  =>  B is BETTER than A

so G1 asks ``mean delta(A1 -> A2) >= +1.0%`` and G3 asks
``delta(A1 -> A0) <= -10.0%`` (A0 clearly worse).

Usage:
    python gate_a_judge.py [--root DIR] [--json]

Exit code 0 when every gate that can be evaluated passes, 1 otherwise.
Gates that lack data are reported as PENDING, never as pass.
"""
import argparse
import json
import os
import re
import sys

VARIANTS = ['A0', 'A1', 'A2', 'A3a', 'A4', 'A5']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
REFERENCE = 'A3p'
SUFFIXES = ['', 'r2', 'r3', 'r4', 'r5', 'r6']
EXPECTED_ROWS = {'FD001': 101, 'FD002': 261, 'FD003': 101, 'FD004': 251}

# Admission regex, pre-registered in RESULTS_AUDIT section 12.11 / GATE_A_WAVE1_PLAN.
# Anything that does not match is not a Gate A cell: timestamped ids such as
# -20261112T0030Z and probe runs such as -NOISE1 are excluded structurally, and
# the assertion below makes that exclusion explicit rather than incidental.
ADMIT_RE = re.compile(r'^HC-C-(A0|A1|A2|A3a|A4|A5|A3p)-FD00[1-4]-s101-W1(r[2-6])?$')

DEFAULT_ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc'


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _preds_rows(path):
    try:
        with open(path) as fh:
            return sum(1 for _ in fh)
    except Exception:
        return None


def pick_cell(root, variant, subset):
    """First ``ok`` run for this cell, in suffix order; else (None, None, None)."""
    base = 'HC-C-%s-%s-s101-W1' % (variant, subset)
    for suf in SUFFIXES:
        rid = base + suf
        d = os.path.join(root, rid)
        close = _read_json(os.path.join(d, 'close.json'))
        if close and close.get('status') == 'ok':
            return rid, d, close
    return None, None, None


def collect(root):
    cells = {}
    for variant in VARIANTS + [REFERENCE]:
        for subset in SUBSETS:
            rid, d, close = pick_cell(root, variant, subset)
            if not rid:
                cells[(variant, subset)] = None
                continue
            metrics = _read_json(os.path.join(d, 'metrics.json')) or {}
            manifest = _read_json(os.path.join(d, 'manifest.json')) or {}
            history = _read_json(os.path.join(d, 'history.json')) or []
            rows = _preds_rows(os.path.join(d, 'preds.csv'))
            series = [e.get('val_engine_mean_rmse') for e in history
                      if e.get('val_engine_mean_rmse') is not None]
            tail = series[-10:]
            plateau = sum(tail) / len(tail) if tail else None
            cells[(variant, subset)] = dict(
                run_id=rid,
                value=metrics.get('val_engine_mean_rmse'),
                plateau=plateau,
                n_epochs=len(series),
                pooled=metrics.get('val_pooled_rmse'),
                epochs_run=metrics.get('epochs_run'),
                best_ep=metrics.get('best_ep'),
                intervention=metrics.get('cycle_intervention'),
                baseline_version=manifest.get('baseline_version'),
                code_lock_git_sha=manifest.get('code_lock_git_sha'),
                worktree_dirty=manifest.get('worktree_dirty'),
                written_by=close.get('written_by') or manifest.get('written_by'),
                preds_rows=rows,
                preds_ok=(rows == EXPECTED_ROWS[subset]),
            )
    return cells


def value(cells, variant, subset):
    cell = cells.get((variant, subset))
    if not cell or cell.get('value') is None:
        return None
    return float(cell['value'])


def delta(cells, a, b, subset):
    """(A - B) / A; positive means B is better. None when data is missing."""
    va, vb = value(cells, a, subset), value(cells, b, subset)
    if va is None or vb is None or va == 0:
        return None
    return (va - vb) / va


def plateau(cells, variant, subset):
    cell = cells.get((variant, subset))
    if not cell:
        return None
    return cell.get('plateau')


def delta_plateau(cells, a, b, subset):
    """Same sign convention, but on the mean of the last 10 epochs."""
    va, vb = plateau(cells, a, subset), plateau(cells, b, subset)
    if va is None or vb is None or va == 0:
        return None
    return (va - vb) / va


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return 'n/a' if x is None else '%+.2f%%' % (x * 100.0)


def evaluate(cells):
    out = {}

    # ---- integrity -------------------------------------------------------
    problems = []
    for (variant, subset), cell in sorted(cells.items()):
        if not cell:
            continue
        rid = cell['run_id']
        if not ADMIT_RE.match(rid):
            # Explicit assertion: a probe run or a timestamped id must never be
            # admitted as a Gate A cell. If this fires, the cell-selection logic
            # is wrong -- do not silently drop it.
            problems.append('%s: run_id does not match the Gate A admission regex' % rid)
        if cell['baseline_version'] != 'v9':
            problems.append('%s: baseline_version=%r' % (rid, cell['baseline_version']))
        if cell['written_by'] != 'run_hc':
            problems.append('%s: written_by=%r' % (rid, cell['written_by']))
        if not cell['preds_ok']:
            problems.append('%s: preds rows=%r expected %d'
                            % (rid, cell['preds_rows'], EXPECTED_ROWS[subset]))
        if cell['pooled'] is not None and cell['value'] is not None \
                and cell['value'] > float(cell['pooled']) + 1e-9:
            problems.append('%s: Jensen violated (%r > %r)'
                            % (rid, cell['value'], cell['pooled']))
    out['integrity_problems'] = problems

    # ---- G1: cycle helps? A2 vs A1 ---------------------------------------
    g1 = {s: delta(cells, 'A1', 'A2', s) for s in SUBSETS}
    g1_mean = mean(g1.values())
    g1_done = all(v is not None for v in g1.values())
    g1_pass = None
    if g1_done:
        g1_pass = (g1_mean >= 0.010) and all(v >= -0.020 for v in g1.values())
    out['G1'] = dict(deltas=g1, mean=g1_mean, complete=g1_done, passed=g1_pass,
                     rule='mean d(A1->A2) >= +1.0% and no subset < -2.0%')

    # ---- G2: Fourier vs linear at the same injection point ---------------
    g2 = {s: delta(cells, 'A2', 'A3a', s) for s in SUBSETS}
    g2_mean = mean(g2.values())
    g2_done = all(v is not None for v in g2.values())
    g2_pass = None
    if g2_done:
        positives = sum(1 for v in g2.values() if v > 0)
        g2_pass = (g2_mean >= 0.010) and (positives >= 3)
    out['G2'] = dict(deltas=g2, mean=g2_mean, complete=g2_done, passed=g2_pass,
                     rule='mean d(A2->A3a) >= +1.0% and >=3/4 subsets same sign')

    # ---- G2' : injection point (reference only, no threshold) ------------
    g2p = {s: delta(cells, 'A3a', 'A3p', s) for s in SUBSETS}
    out['G2p'] = dict(deltas=g2p, mean=mean(g2p.values()),
                      rule='report only; no threshold; not in the main table')

    # ---- G3: age shortcut check ------------------------------------------
    g3 = {s: delta(cells, 'A1', 'A0', s) for s in SUBSETS}
    g3_done = all(v is not None for v in g3.values())
    g3_pass = None
    if g3_done:
        g3_pass = all(v <= -0.100 for v in g3.values())
    out['G3'] = dict(deltas=g3, mean=mean(g3.values()), complete=g3_done,
                     passed=g3_pass,
                     rule='d(A1->A0) <= -10.0% on every subset')

    # ---- G4: cycle path actually used ------------------------------------
    g4 = {}
    for variant in ('A2', 'A3a', 'A4', 'A5', 'A1'):
        vals = []
        for subset in SUBSETS:
            cell = cells.get((variant, subset))
            iv = (cell or {}).get('intervention') or {}
            vals.append((iv.get('mean_abs_delta'), iv.get('frac_negative')))
        g4[variant] = vals
    out['G4'] = dict(per_variant=g4,
                     rule='cycle arms: mean_abs_delta >= 0.5 and frac_negative >= 0.5; '
                          'A1 must have mean_abs_delta == 0')

    # ---- G5: sign stability ----------------------------------------------
    sign = {}
    for name, table in (('G1', g1), ('G2', g2)):
        signs = [v for v in table.values() if v is not None]
        if len(signs) == 4:
            pos = sum(1 for v in signs if v > 0)
            sign[name] = dict(positive=pos, consistent=(pos >= 3 or pos <= 1))
    out['G5'] = dict(sign=sign, rule='G1/G2 sign consistent on >=3/4 subsets')

    # ---- robustness: same comparisons on the plateau mean (last 10 epochs)
    robust = {}
    for name, a, b in (('G1', 'A1', 'A2'), ('G2', 'A2', 'A3a'), ('G3', 'A1', 'A0')):
        table = {s: delta_plateau(cells, a, b, s) for s in SUBSETS}
        robust[name] = dict(deltas=table, mean=mean(table.values()))
    out['robustness'] = robust
    return out


def render(cells, res):
    lines = []
    lines.append('%-6s %11s %11s %11s %11s' %
                 ('cell', 'FD001', 'FD002', 'FD003', 'FD004'))
    for variant in VARIANTS + [REFERENCE]:
        vals = []
        for subset in SUBSETS:
            v = value(cells, variant, subset)
            vals.append('         --' if v is None else '%11.4f' % v)
        lines.append('%-6s %s' % (variant, ' '.join(vals)))

    lines.append('')
    for key in ('G1', 'G2', 'G2p', 'G3'):
        block = res[key]
        cells_txt = ' '.join(pct(block['deltas'][s]) for s in SUBSETS)
        verdict = block.get('passed')
        tag = 'PENDING' if verdict is None else ('PASS' if verdict else 'FAIL')
        lines.append('%s  mean=%-9s  [%s]  %s' %
                     (key.ljust(4), pct(block['mean']), tag, cells_txt))
        lines.append('      rule: %s' % block['rule'])

    lines.append('G5   %s' % res['G5']['sign'])
    lines.append('')
    lines.append('ROBUSTNESS (same deltas on the mean of the last 10 epochs;')
    lines.append('            reported for comparison, NOT used as the verdict):')
    for key in ('G1', 'G2', 'G3'):
        block = res['robustness'][key]
        cells_txt = ' '.join(pct(block['deltas'][s]) for s in SUBSETS)
        lines.append('  %-4s mean=%-9s  %s' % (key, pct(block['mean']), cells_txt))
        base = res[key]['deltas']
        have = [s for s in SUBSETS if base.get(s) is not None
                and block['deltas'].get(s) is not None]
        order_a = sorted(have, key=lambda s: -base[s])
        order_b = sorted(have, key=lambda s: -block['deltas'][s])
        if len(have) >= 2 and order_a != order_b:
            lines.append('        RANKING FLIPPED (n=%d): best-epoch %s  vs  plateau %s'
                         % (len(have), '>'.join(order_a), '>'.join(order_b)))
        elif len(have) >= 2:
            lines.append('        ranking stable (n=%d): %s' % (len(have), '>'.join(order_a)))
    lines.append('')
    if res['integrity_problems']:
        lines.append('INTEGRITY PROBLEMS (%d):' % len(res['integrity_problems']))
        for p in res['integrity_problems']:
            lines.append('  - ' + p)
    else:
        lines.append('INTEGRITY: all collected cells pass baseline/written_by/preds/Jensen')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    cells = collect(args.root)
    res = evaluate(cells)
    if args.json:
        payload = dict(
            cells={'%s/%s' % k: v for k, v in cells.items()},
            results=res,
        )
        print(json.dumps(payload, indent=1, default=str))
    else:
        print(render(cells, res))

    verdicts = [res[k]['passed'] for k in ('G1', 'G2', 'G3')]
    if any(v is False for v in verdicts) or res['integrity_problems']:
        return 1
    if any(v is None for v in verdicts):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
