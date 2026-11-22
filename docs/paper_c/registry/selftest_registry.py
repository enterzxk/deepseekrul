"""Self-test for check_registry.py: 19 cases, every one asserted by exit code.

    python docs/paper_c/registry/selftest_registry.py

The previous version of the checker was validated with nine throwaway cases that
were never committed, so "the checker really blocks" was a claim in a document
rather than a thing anyone could re-run. This file is that claim, executable.

Fixtures are built from the real queue (so run_id patterns are the real ones) and
from the real frozen anchors (so segment-A baseline values are the real ones).
Nothing here writes to runs.csv or run_queue.csv: every case runs against files
in a temp directory via check_registry.py --runs/--queue/--grid.
"""
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import schema_cols

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = schema_cols.COLS


def load_queue():
    return list(csv.DictReader(io.open(os.path.join(HERE, 'run_queue.csv'), encoding='utf-8')))


def load_anchors():
    rows = list(csv.DictReader(io.open(os.path.join(HERE, 'FIXED40_ANCHORS.csv'), encoding='utf-8')))
    return {(r['model'], r['subset'], r['seed']): r for r in rows}


def base_row(q, anchor=None, shift=0.05):
    """A complete ok row for a queue cell; val = anchor - shift so delta = +shift."""
    r = {c: '' for c in COLS}
    r.update(run_id=q['run_id'], wave=q['wave'], stage='P2', segment=q['segment'],
             arm=q['arm'], base_model=q['base_model'], cadf_variant=q['variant_tag'],
             cadf_k0_mode=q['k0_mode'], cadf_k0_value=q['k0_resolved'],
             subset=q['subset'], seed=q['seed'], span=q['span'],
             split_protocol=q['split_protocol'], split_seed=q['split_seed'],
             gate='P2_cross_arch', in_gate_table='false',
             hypothesis='h', host='lab3090', git_sha='deadbeef', code_md5='{}',
             config_sha256='0' * 64, launched_utc='2026-11-23T00:00:00Z',
             closed_utc='2026-11-23T01:00:00Z', status='ok',
             epochs_run='27', best_epoch='15', budget_truncated='false',
             val_engine_mean_rmse='10.0', val_pooled_rmse='10.5',
             eol95_mae='3.0', ckpt_path='ckpt.pt', log_path='log.txt',
             preds_path='preds.csv', baseline_source=q['baseline_source'],
             baseline_ref=q['baseline_ref'], baseline_version=q['baseline_version'],
             registry_note='', audited_by='', audit_verdict='n/a',
             determinism_only='false')
    if anchor is not None:
        b = float(anchor['val_engine'])
        r['baseline_val_engine_mean_rmse'] = repr(b)
        c = b - shift
        r['val_engine_mean_rmse'] = repr(c)
        r['delta_val_rmse'] = repr(b - c)
        r['epochs_run'] = anchor['epochs_run']
        r['best_epoch'] = anchor['best_epoch']
        r['budget_truncated'] = anchor['truncated'].lower()
    if q['arm'] == 'cadf':
        r['gate_alpha_trace'] = 'traces/%s.csv' % q['run_id']
        r['gate_open_epoch'] = '12'
        r['gate_norm_last'] = '0.031'
        r['alpha_last'] = '0.72'
        r['in_gate_table'] = 'true'
        r['paired_win_rate'] = '0.62'
    return r


def write_runs(path, rows, cols=None):
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        w = csv.DictWriter(fh, fieldnames=cols or COLS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in (cols or COLS)})


def run_checker(tmp, runs, queue, grid):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'check_registry.py'),
                        '--runs', runs, '--queue', queue, '--grid', grid],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (p.stdout or '') + (p.stderr or '')
    return p.returncode, out


def main():
    queue = load_queue()
    anchors = load_anchors()
    v0q = [q for q in queue if q['arm'] == 'v0']
    cadfq = [q for q in queue if q['arm'] == 'cadf']
    tmp = tempfile.mkdtemp(prefix='p2reg_')
    grid = os.path.join(HERE, 'grid.json')
    # post-reconciliation queue: segment B is authorized once A reconciled
    qpath = os.path.join(tmp, 'queue_open.csv')
    with io.open(qpath, 'w', encoding='utf-8', newline='\n') as fh:
        w = csv.DictWriter(fh, fieldnames=list(queue[0].keys()), lineterminator='\n')
        w.writeheader()
        for q in queue:
            q = dict(q)
            q['launch_allowed'] = 'true'
            q['launch_gate'] = 'open'
            w.writerow(q)
    qpath_closed = os.path.join(tmp, 'queue_closed.csv')
    with io.open(qpath_closed, 'w', encoding='utf-8', newline='\n') as fh:
        w = csv.DictWriter(fh, fieldnames=list(queue[0].keys()), lineterminator='\n')
        w.writeheader()
        for q in queue:
            q = dict(q)
            q['launch_allowed'] = 'false'
            q['launch_gate'] = 'prereg_commit'
            w.writerow(q)

    a_ok = [base_row(q, anchors[(q['base_model'], q['subset'], q['seed'])]) for q in v0q]
    bq = [q for q in cadfq if q['base_model'] == 'aidgn' and q['subset'] == 'FD001' and q['seed'] == '101'][0]
    v0aidgn = [r for r in a_ok if r['run_id'] == bq['baseline_ref']][0]
    b_ok = base_row(bq, shift=0.45)
    b_ok['baseline_val_engine_mean_rmse'] = v0aidgn['val_engine_mean_rmse']
    b_ok['val_engine_mean_rmse'] = repr(float(v0aidgn['val_engine_mean_rmse']) - 0.45)
    b_ok['delta_val_rmse'] = repr(0.45)

    grid_drift = os.path.join(tmp, 'grid_drift.json')
    g = json.load(io.open(grid, encoding='utf-8'))
    g['prereg']['commit'] = '1b82b450'   # a commit where the prereg does not exist
    io.open(grid_drift, 'w', encoding='utf-8', newline='\n').write(json.dumps(g, ensure_ascii=False))

    # segment-B cases run against a grid where A has already been reconciled,
    # otherwise check 11 masks whatever the case is actually about
    grid_open = os.path.join(tmp, 'grid_open.json')
    g = json.load(io.open(grid, encoding='utf-8'))
    g['a_segment_reconciled'] = True
    io.open(grid_open, 'w', encoding='utf-8', newline='\n').write(json.dumps(g, ensure_ascii=False))

    cases = []

    def case(name, rows, want, queue_path=None, grid_path=None, cols=None):
        p = os.path.join(tmp, 'runs_%d.csv' % len(cases))
        write_runs(p, rows, cols)
        code, out = run_checker(tmp, p, queue_path or qpath, grid_path or grid)
        cases.append((name, want, code, out))

    def replaced(rows, run_id, **changes):
        """Swap one cell in place: a case that appends a mutated copy would trip
        over 'duplicate ok cell' and test the wrong assertion."""
        out = []
        for r in rows:
            if r['run_id'] == run_id:
                r = dict(r)
                r.update(changes)
            out.append(r)
        return out

    case('30 segment-A cells, no B yet', a_ok, 0)
    case('30 A + 1 B', a_ok + [b_ok], 0, grid_path=grid_open)

    bad = [dict(r) for r in a_ok]
    bad[0]['val_engine_mean_rmse'] = repr(float(bad[0]['baseline_val_engine_mean_rmse']) - 0.5)
    bad[0]['delta_val_rmse'] = repr(0.5)
    case('A cell out of tolerance + a B result', bad + [b_ok], 2, grid_path=grid_open)
    case('A missing one cell + a B result', a_ok[1:] + [b_ok], 2, grid_path=grid_open)

    # prereg R2 exempts the five budget-truncated cells: they were cut at the
    # ceiling while still improving, so their |delta| alone is not an ABORT
    tcell = [r for r in a_ok if r['run_id'] == 'pc-v0-two_phase-FD004-s101-W1']
    if not tcell:
        raise SystemExit('exemption fixture missing: pc-v0-two_phase-FD004-s101-W1')
    soft = replaced(a_ok, tcell[0]['run_id'],
                    val_engine_mean_rmse=repr(float(tcell[0]['baseline_val_engine_mean_rmse']) - 0.5),
                    delta_val_rmse=repr(0.5))
    case('only a budget-truncated A cell out of tolerance + B', soft + [b_ok], 0, grid_path=grid_open)

    x = dict(b_ok); x['baseline_source'] = 'frozen_ledger'
    x['baseline_ref'] = 'fixed40/aidgn/FD001/s101'
    case('B paired against the cross-machine ledger', a_ok + [x], 1, grid_path=grid_open)

    x = dict(b_ok); x['delta_val_rmse'] = repr(-0.45)
    case('B delta sign reversed', a_ok + [x], 1, grid_path=grid_open)

    x = dict(b_ok); x['test_rmse'] = '9.9'
    case('B with TEST filled while sealed', a_ok + [x], 1, grid_path=grid_open)

    x = dict(b_ok); x['gate_alpha_trace'] = ''
    case('B without the gate/alpha trace', a_ok + [x], 1, grid_path=grid_open)

    x = dict(b_ok); x['gate_open_epoch'] = 'none'
    case('B gate never opened but in the Gate table', a_ok + [x], 1, grid_path=grid_open)

    x = dict(b_ok); x['gate_open_epoch'] = 'none'; x['in_gate_table'] = 'false'
    case('B gate never opened, kept out of the Gate table', a_ok + [x], 0, grid_path=grid_open)

    case('duplicate ok cell', a_ok + [dict(a_ok[0])], 1)

    # the old dedup key (baseline_version, cadf_variant, subset, seed) had no
    # base_model, so these two legitimate cells looked like duplicates
    case('same variant/subset/seed, different model', a_ok, 0)

    x = dict(a_ok[0]); x['status'] = 'superseded'
    case('superseded correction shares the cell', a_ok + [x], 0)

    x = dict(a_ok[0]); x['status'] = 'done'
    case('bad status enum', a_ok + [x], 1)

    x = dict(a_ok[0]); x['run_id'] = 'pc-v0-aidgn-FD002-s101-W1'
    case('run_id outside the grid', a_ok + [x], 1)

    case('ok row whose queue cell was never authorized', a_ok, 1, queue_path=qpath_closed)

    x = dict(b_ok); x['cadf_k0_value'] = '15'
    case('k0p resolved to 15 instead of 10', a_ok + [x], 1, grid_path=grid_open)

    case('epochs_run above the fixed40 ceiling',
         replaced(a_ok, a_ok[0]['run_id'], epochs_run='41'), 1)

    case('prereg commit no longer holds the prereg', a_ok, 2, grid_path=grid_drift)

    case('header lost a column', a_ok, 2, cols=[c for c in COLS if c != 'gate_open_epoch'])

    width = max(len(n) for n, _, _, _ in cases)
    bad_n = 0
    for name, want, got, out in cases:
        ok = (want == got)
        bad_n += 0 if ok else 1
        first = ''
        if not ok or got != 0:
            lines = [l.strip() for l in out.splitlines() if l.strip().startswith('-')]
            first = lines[0] if lines else out.strip().splitlines()[0] if out.strip() else ''
        print('%-*s exit=%d want=%d %s  %s' % (width, name, got, want, 'OK' if ok else 'MISMATCH', first[:110]))
    print('%d/%d cases behaved as expected' % (len(cases) - bad_n, len(cases)))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if bad_n == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
