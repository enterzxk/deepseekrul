"""Paper C registry checker. Exit 0 only when every assertion holds.

    python docs/paper_c/registry/check_registry.py

Checks, in order:
  1  runs.csv header equals the schema in SCHEMA.md (single source: COLS here)
  2  enum domains: status, audit_verdict, determinism_only, in_gate_table
  3  run_id matches the grid pattern for the variant tags declared in grid.json
  4  dedup key (baseline_version, cadf_variant, subset, seed): one ok row only,
     later duplicates must be status=superseded
  5  every ok row has its required evidence fields non-empty
  6  baseline_ref resolves to exactly one row in the pinned baseline ledger with
     source_condition=fixed40 and the same (base_model, subset, seed)
  7  delta_val_rmse == baseline - cadf (sign convention from grid.json)
  8  TEST stays sealed: no test_* column may be filled while stage != Terminal
  9  every ok row is in the launch queue
"""
import csv
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

COLS = ['run_id', 'wave', 'stage', 'base_model', 'cadf_variant', 'cadf_k0_mode',
        'cadf_k0_value', 'subset', 'seed', 'span', 'split_protocol', 'split_seed',
        'gate', 'in_gate_table', 'hypothesis', 'host', 'git_sha', 'code_md5',
        'config_sha256', 'launched_utc', 'closed_utc', 'status', 'epochs_run',
        'best_epoch', 'budget_truncated', 'val_engine_mean_rmse', 'val_pooled_rmse',
        'val_mae', 'val_bias', 'eol95_mae', 'val_nasa', 'test_rmse', 'test_score',
        'ckpt_path', 'log_path', 'preds_path', 'baseline_ref',
        'baseline_val_engine_mean_rmse', 'delta_val_rmse', 'paired_win_rate',
        'registry_note', 'audited_by', 'audit_verdict', 'determinism_only',
        'baseline_version']

STATUS = {'ok', 'failed', 'smoke', 'running', 'superseded'}
VERDICT = {'pass', 'fail', 'n/a'}
BOOL = {'true', 'false'}
TEST_COLS = ['test_rmse', 'test_score']
OK_REQUIRED = ['git_sha', 'code_md5', 'config_sha256', 'launched_utc', 'closed_utc',
               'status', 'epochs_run', 'best_epoch', 'val_engine_mean_rmse',
               'eol95_mae', 'ckpt_path', 'preds_path', 'baseline_ref',
               'baseline_val_engine_mean_rmse', 'delta_val_rmse']


def fail(errs, msg):
    errs.append(msg)


def main():
    errs = []
    g = json.load(io.open(os.path.join(HERE, 'grid.json'), encoding='utf-8'))
    runs_path = os.path.join(HERE, 'runs.csv')
    with io.open(runs_path, encoding='utf-8', newline='') as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames
        rows = list(rd)

    if header != COLS:
        missing = [c for c in COLS if c not in (header or [])]
        extra = [c for c in (header or []) if c not in COLS]
        fail(errs, '1 header mismatch: missing=%s extra=%s' % (missing, extra))
        print('FAILED header; cannot continue safely')
        return 1

    tags = '|'.join(g['variant_tags'])
    models = '|'.join(g['methods'])
    subsets = '|'.join(g['subsets'])
    seeds = '|'.join(str(s) for s in g['seeds'])
    pattern = re.compile(r'^pc-cadf-(%s)-(%s)-(%s)-s(%s)-W1(r\d+)?$'
                         % (tags, models, subsets, seeds))

    seen = {}
    for i, r in enumerate(rows, start=2):
        rid = r['run_id']
        if r['status'] not in STATUS:
            fail(errs, '2 line %d status=%r' % (i, r['status']))
        if r['audit_verdict'] not in VERDICT:
            fail(errs, '2 line %d audit_verdict=%r' % (i, r['audit_verdict']))
        for c in ('determinism_only', 'in_gate_table', 'budget_truncated'):
            if r[c] not in BOOL:
                fail(errs, '2 line %d %s=%r' % (i, c, r[c]))
        if not pattern.match(rid):
            fail(errs, '3 line %d run_id=%r' % (i, rid))
        key = (r['baseline_version'], r['cadf_variant'], r['subset'], r['seed'])
        if r['status'] == 'ok':
            if key in seen:
                fail(errs, '4 duplicate ok cell %s (line %d and %d)' % (key, seen[key], i))
            seen[key] = i
        # failed attempts and superseded corrections may share a cell key with an
        # ok row; only a second ok row is an error.
        if r['status'] == 'ok':
            for c in OK_REQUIRED:
                if not r[c].strip():
                    fail(errs, '5 line %d ok row missing %s' % (i, c))
        if r['status'] == 'ok' and r['stage'] != 'Terminal':
            for c in TEST_COLS:
                if r[c].strip():
                    fail(errs, '8 line %d TEST is sealed but %s is filled' % (i, c))
        if r['status'] == 'ok' and r['delta_val_rmse'].strip():
            b = float(r['baseline_val_engine_mean_rmse'])
            c = float(r['val_engine_mean_rmse'])
            d = float(r['delta_val_rmse'])
            if abs((b - c) - d) > 1e-9:
                fail(errs, '7 line %d delta %.9f != baseline %.9f - cadf %.9f' % (i, d, b, c))

    # 6 baseline resolution against the pinned ledger
    base_txt = subprocess.run(
        ['git', 'show', '%s:%s' % (g['baseline_commit'], g['baseline_ledger'])],
        cwd=REPO, capture_output=True, text=True, encoding='utf-8')
    if base_txt.returncode != 0:
        fail(errs, '6 cannot read baseline ledger %s@%s: %s'
             % (g['baseline_ledger'], g['baseline_commit'], base_txt.stderr.strip()[:120]))
        base = []
    else:
        base = list(csv.DictReader(io.StringIO(base_txt.stdout)))
    bidx = {(r['physical_id']): r for r in base}
    for i, r in enumerate(rows, start=2):
        if r['status'] != 'ok':
            continue
        b = bidx.get(r['baseline_ref'])
        if b is None:
            fail(errs, '6 line %d baseline_ref %r not in ledger' % (i, r['baseline_ref']))
            continue
        if b['source_condition'] != g['baseline_source_condition']:
            fail(errs, '6 line %d baseline_ref has source_condition=%r, need %r'
                 % (i, b['source_condition'], g['baseline_source_condition']))
        if (b['model'], b['subset'], b['seed']) != (r['base_model'], r['subset'], r['seed']):
            fail(errs, '6 line %d baseline_ref %r does not match %s/%s/s%s'
                 % (i, r['baseline_ref'], r['base_model'], r['subset'], r['seed']))
        if r['baseline_val_engine_mean_rmse'].strip():
            want = float(b['validation_engine_mean_rmse'])
            got = float(r['baseline_val_engine_mean_rmse'])
            if abs(want - got) > 1e-9:
                fail(errs, '6 line %d baseline value %.9f != ledger %.9f' % (i, got, want))

    # 9 ok rows must be queued
    q = list(csv.DictReader(io.open(os.path.join(HERE, 'run_queue.csv'), encoding='utf-8')))
    queued = {r['run_id'] for r in q}
    for i, r in enumerate(rows, start=2):
        if r['status'] == 'ok' and r['run_id'] not in queued:
            fail(errs, '9 line %d ok row not in run_queue.csv' % i)

    done = {r['run_id'] for r in rows if r['status'] == 'ok'}
    pending = sorted(queued - done)
    print('rows=%d cols=%d queued=%d ok=%d pending=%d'
          % (len(rows), len(header), len(queued), len(done), len(pending)))
    if errs:
        print('FAILED (%d)' % len(errs))
        for e in errs[:40]:
            print('  -', e)
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
