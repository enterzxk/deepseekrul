"""Paper C registry checker. Exit 0 only when every assertion holds.

    python docs/paper_c/registry/check_registry.py

Exit codes (three, never reused):
    0  PASS   every assertion holds
    1  FAIL   a row or the ledger is wrong
    2  ABORT  a precondition is missing or an input is unusable (the frozen
              artifacts moved, the ledger header cannot be trusted, or segment B
              holds results while segment A did not reconcile)

Checks, in order:
  1  runs.csv header equals schema_cols.COLS
  2  enum domains: status, audit_verdict, booleans, arm, segment, and the
     segment<->arm pairing (A=v0, B=cadf)
  3  run_id matches the pattern of its own arm
  4  dedup key (baseline_version, arm, base_model, cadf_variant, subset, seed):
     exactly one ok row; failed retries and superseded corrections may share it
  5  ok rows carry their required evidence; the two arms have different extras
     (a cadf row without a gate/alpha trace is not evidence of anything)
  6  baseline resolution: segment A against the pinned frozen ledger
     (source_condition=fixed40), segment B against this batch's segment-A cell -
     a segment-B row that points at the cross-machine ledger is rejected
  7  delta_val_rmse == baseline - this cell
  8  TEST stays sealed while stage != Terminal
  9  every ok row is in the queue and its queue row says launch_allowed=true
 10  span/epochs/best_epoch/k0 agree with grid.json
 11  segment B gate: no cadf ok row unless all 30 segment-A cells are ok and
     every |delta| there is within recon_tolerance                       [ABORT]
 12  the frozen artifacts (prereg, manifest, wrapper) still hash to the values
     recorded at freeze time                                            [ABORT]
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys

import schema_cols

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
COLS = schema_cols.COLS

STATUS = {'ok', 'failed', 'smoke', 'running', 'superseded'}
VERDICT = {'pass', 'fail', 'n/a'}
BOOL = {'true', 'false'}
TEST_COLS = ['test_rmse', 'test_score']
OK_REQUIRED = ['git_sha', 'code_md5', 'config_sha256', 'launched_utc', 'closed_utc',
               'status', 'epochs_run', 'best_epoch', 'val_engine_mean_rmse',
               'eol95_mae', 'ckpt_path', 'preds_path', 'baseline_source',
               'baseline_ref', 'baseline_val_engine_mean_rmse', 'delta_val_rmse']
CADF_REQUIRED = ['gate_alpha_trace', 'gate_open_epoch', 'gate_norm_last', 'alpha_last']
V0_EMPTY = ['gate_alpha_trace', 'gate_open_epoch', 'gate_norm_last', 'alpha_last',
            'paired_win_rate', 'cadf_k0_value']


def git_show(commit, path):
    p = subprocess.run(['git', 'show', '%s:%s' % (commit, path)],
                       cwd=REPO, capture_output=True)
    if p.returncode != 0:
        return None
    return p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs.csv'))
    ap.add_argument('--queue', default=os.path.join(HERE, 'run_queue.csv'))
    ap.add_argument('--grid', default=os.path.join(HERE, 'grid.json'))
    a = ap.parse_args(argv)

    errs, aborts = [], []
    g = json.load(io.open(a.grid, encoding='utf-8'))
    with io.open(a.runs, encoding='utf-8', newline='') as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames
        rows = list(rd)

    # 1 header
    if header != COLS:
        missing = [c for c in COLS if c not in (header or [])]
        extra = [c for c in (header or []) if c not in COLS]
        aborts.append('1 header mismatch: missing=%s extra=%s' % (missing, extra))
        print('ABORT: runs.csv header is not the schema; refusing to read rows')
        print('  -', aborts[-1])
        return 2

    arms = {x['arm']: x for x in g['arms']}
    patterns = {k: re.compile('^%s$' % v['run_id_pattern']
                              .replace('<variant_tag>', '(%s)' % '|'.join(g['variant_tags']))
                              .replace('<base_model>', '(%s)' % '|'.join(g['methods']))
                              .replace('<subset>', '(%s)' % '|'.join(g['subsets']))
                              .replace('<seed>', '(%s)' % '|'.join(str(s) for s in g['seeds']))
                              + '(r\\d+)?')
                for k, v in arms.items()}

    seen = {}
    for i, r in enumerate(rows, start=2):
        rid = r['run_id']
        if r['status'] not in STATUS:
            errs.append('2 line %d status=%r' % (i, r['status']))
        if r['audit_verdict'] not in VERDICT:
            errs.append('2 line %d audit_verdict=%r' % (i, r['audit_verdict']))
        for c in ('determinism_only', 'in_gate_table', 'budget_truncated'):
            if r[c] not in BOOL:
                errs.append('2 line %d %s=%r' % (i, c, r[c]))
        if r['arm'] not in arms:
            errs.append('2 line %d arm=%r' % (i, r['arm']))
            continue
        if r['segment'] != arms[r['arm']]['segment']:
            errs.append('2 line %d segment=%r but arm=%r is segment %r'
                        % (i, r['segment'], r['arm'], arms[r['arm']]['segment']))
        if not patterns[r['arm']].match(rid):
            errs.append('3 line %d run_id=%r does not match the %r pattern'
                        % (i, rid, r['arm']))
        key = (r['baseline_version'], r['arm'], r['base_model'], r['cadf_variant'],
               r['subset'], r['seed'])
        if r['status'] == 'ok':
            if key in seen:
                errs.append('4 duplicate ok cell %s (line %d and %d)'
                            % (key, seen[key], i))
            seen[key] = i
            for c in OK_REQUIRED:
                if not r[c].strip():
                    errs.append('5 line %d ok row missing %s' % (i, c))
            if r['arm'] == 'v0':
                if r['cadf_variant'] != arms['v0']['cadf_variant']:
                    errs.append('5 line %d v0 row cadf_variant=%r, want %r'
                                % (i, r['cadf_variant'], arms['v0']['cadf_variant']))
                if r['cadf_k0_mode'] != 'none':
                    errs.append('5 line %d v0 row cadf_k0_mode=%r, want none'
                                % (i, r['cadf_k0_mode']))
                for c in V0_EMPTY:
                    if r[c].strip():
                        errs.append('5 line %d v0 row must leave %s empty' % (i, c))
            else:
                for c in CADF_REQUIRED:
                    if not r[c].strip():
                        errs.append('5 line %d cadf row missing %s' % (i, c))
                if r['gate_open_epoch'].strip() == 'none' and r['in_gate_table'] == 'true':
                    errs.append('5 line %d gate never opened (gate_open_epoch=none) '
                                'but in_gate_table=true; a cell whose gate never woke '
                                'is not evidence for or against the module' % i)
        if r['status'] == 'ok' and r['stage'] != 'Terminal':
            for c in TEST_COLS:
                if r[c].strip():
                    errs.append('8 line %d TEST is sealed but %s is filled' % (i, c))
        if r['status'] == 'ok' and r['delta_val_rmse'].strip():
            b = float(r['baseline_val_engine_mean_rmse'])
            c = float(r['val_engine_mean_rmse'])
            d = float(r['delta_val_rmse'])
            if abs((b - c) - d) > 1e-9:
                errs.append('7 line %d delta %.9f != baseline %.9f - cell %.9f'
                            % (i, d, b, c))

    # 10 grid agreement
    for i, r in enumerate(rows, start=2):
        if r['arm'] not in arms:
            continue
        want_span = g['windows'].get(r['base_model'], {}).get(r['subset'])
        if want_span is None:
            errs.append('10 line %d no window declared for %s/%s'
                        % (i, r['base_model'], r['subset']))
        elif r['span'].strip() and int(r['span']) != want_span:
            errs.append('10 line %d span=%s but grid says %d for %s/%s'
                        % (i, r['span'], want_span, r['base_model'], r['subset']))
        if r['status'] != 'ok':
            continue
        try:
            ep = int(r['epochs_run'])
            be = int(r['best_epoch'])
        except ValueError:
            errs.append('10 line %d epochs_run/best_epoch not integers' % i)
            continue
        if ep > g['epochs']:
            errs.append('10 line %d epochs_run=%d exceeds the fixed40 ceiling %d'
                        % (i, ep, g['epochs']))
        if be > ep:
            errs.append('10 line %d best_epoch=%d > epochs_run=%d' % (i, be, ep))
        if r['arm'] == 'cadf':
            variant = r['cadf_variant']
            if variant not in g['variant_tags']:
                errs.append('10 line %d cadf_variant=%r not in grid %s'
                            % (i, variant, g['variant_tags']))
                continue
            mode = g['k0_mode_by_variant'][variant]
            if r['cadf_k0_mode'] != mode:
                errs.append('10 line %d cadf_k0_mode=%r, grid says %r for %s'
                            % (i, r['cadf_k0_mode'], mode, variant))
            if mode == 'prop':
                want_k0 = int(math.ceil(g['k0_prop_fraction'] * int(r['span'])))
            elif mode == 'abs':
                want_k0 = g['k0_abs_value']
            else:
                want_k0 = None
            if want_k0 is not None and r['cadf_k0_value'].strip() != str(want_k0):
                errs.append('10 line %d cadf_k0_value=%r but %s on span %s resolves to %d'
                            % (i, r['cadf_k0_value'], variant, r['span'], want_k0))

    # 6 baseline resolution
    base_txt = git_show(g['baseline_commit'], g['baseline_ledger'])
    if base_txt is None:
        aborts.append('6 cannot read baseline ledger %s@%s'
                      % (g['baseline_ledger'], g['baseline_commit']))
        base = []
    else:
        base = list(csv.DictReader(io.StringIO(base_txt.decode('utf-8'))))
    bidx = {r['physical_id']: r for r in base}
    v0_ok = {}
    for i, r in enumerate(rows, start=2):
        if r['status'] == 'ok' and r['arm'] == 'v0':
            v0_ok[(r['base_model'], r['subset'], r['seed'])] = (i, r)

    for i, r in enumerate(rows, start=2):
        if r['status'] != 'ok':
            continue
        src = r['baseline_source']
        if src == 'frozen_ledger':
            b = bidx.get(r['baseline_ref'])
            if b is None:
                errs.append('6 line %d baseline_ref %r not in the frozen ledger'
                            % (i, r['baseline_ref']))
                continue
            if b['source_condition'] != g['baseline_source_condition']:
                errs.append('6 line %d baseline_ref has source_condition=%r, need %r'
                            % (i, b['source_condition'], g['baseline_source_condition']))
            if (b['model'], b['subset'], b['seed']) != (r['base_model'], r['subset'], r['seed']):
                errs.append('6 line %d baseline_ref %r does not match %s/%s/s%s'
                            % (i, r['baseline_ref'], r['base_model'], r['subset'], r['seed']))
            want = float(b['validation_engine_mean_rmse'])
            got = float(r['baseline_val_engine_mean_rmse'])
            if abs(want - got) > 1e-9:
                errs.append('6 line %d baseline value %.9f != ledger %.9f'
                            % (i, got, want))
        elif src == 'local_v0':
            k = (r['base_model'], r['subset'], r['seed'])
            hit = v0_ok.get(k)
            if hit is None:
                errs.append('6 line %d segment-B row pairs against local V0 %s but '
                            'there is no ok segment-A cell for it' % (i, list(k)))
                continue
            j, v0r = hit
            if r['baseline_ref'] != v0r['run_id']:
                errs.append('6 line %d baseline_ref=%r, want this batch\'s segment-A '
                            'run_id %r (a cross-machine ledger ref is not a pair)'
                            % (i, r['baseline_ref'], v0r['run_id']))
            want = float(v0r['val_engine_mean_rmse'])
            got = float(r['baseline_val_engine_mean_rmse'])
            if abs(want - got) > 1e-9:
                errs.append('6 line %d baseline value %.9f != segment-A %.9f (line %d)'
                            % (i, got, want, j))
        else:
            errs.append('6 line %d baseline_source=%r' % (i, src))

    # 9 queue coverage
    q = list(csv.DictReader(io.open(a.queue, encoding='utf-8')))
    qidx = {r['run_id']: r for r in q}
    for i, r in enumerate(rows, start=2):
        if r['status'] != 'ok':
            continue
        qq = qidx.get(r['run_id'])
        if qq is None:
            errs.append('9 line %d ok row not in run_queue.csv' % i)
        elif qq['launch_allowed'] != 'true':
            errs.append('9 line %d ok row was never authorized (queue launch_allowed=%r, '
                        'gate=%r)' % (i, qq['launch_allowed'], qq['launch_gate']))

    # 11 segment B gate
    b_ok = [r for r in rows if r['status'] == 'ok' and r['arm'] == 'cadf']
    recon = []
    for k, (i, r) in v0_ok.items():
        recon.append((abs(float(r['delta_val_rmse'])), k, i))
    recon_max = max([x[0] for x in recon], default=0.0)
    # the prereg exempts the budget-truncated cells from ABORT on their own: they
    # were cut at the 40-epoch ceiling while still improving, so a shift there is
    # expected. Read the set from the anchor manifest instead of hard-coding five
    # names, so the exemption follows the data if the manifest is regenerated.
    anchor_path = os.path.join(REPO, g['anchor_manifest'])
    truncated = set()
    if os.path.exists(anchor_path):
        for ar in csv.DictReader(io.open(anchor_path, encoding='utf-8')):
            if int(ar['epochs_run']) >= g['epochs']:
                truncated.add((ar['model'], ar['subset'], ar['seed']))
    elif b_ok:
        aborts.append('11 anchor manifest %s is missing, so the budget-truncated '
                      'exemption cannot be applied' % g['anchor_manifest'])
    over = [x for x in recon if x[0] > g['recon_tolerance']]
    over_hard = [x for x in over if x[1] not in truncated]
    over_soft = [x for x in over if x[1] in truncated]
    if b_ok:
        missing = [(m, s, sd) for m in g['methods'] for s in g['subsets']
                   for sd in (str(x) for x in g['seeds']) if (m, s, sd) not in v0_ok]
        if not g.get('a_segment_reconciled'):
            aborts.append('11 segment B holds %d ok cells but grid.json still has '
                          'a_segment_reconciled=false' % len(b_ok))
        if missing:
            aborts.append('11 segment B holds results but segment A is missing %d of %d '
                          'cells (first: %s)' % (len(missing),
                                                 len(g['methods']) * len(g['subsets']) * len(g['seeds']),
                                                 missing[0]))
        if over_hard:
            worst = max(over_hard)
            aborts.append('11 segment A did not reconcile: %d of %d non-truncated cells '
                          'exceed |delta| <= %.2f (worst %.4f at %s, line %d)'
                          % (len(over_hard), len(recon) - len(truncated),
                             g['recon_tolerance'], worst[0], '/'.join(worst[1]), worst[2]))

    # 12 frozen artifacts unchanged
    fz_path = os.path.join(REPO, g['freeze_hashes'])
    if not os.path.exists(fz_path):
        aborts.append('12 freeze manifest %s is missing' % g['freeze_hashes'])
    else:
        fz = json.load(io.open(fz_path, encoding='utf-8'))
        for name, ent in fz['artifacts'].items():
            if name == 'prereg' and (ent['commit'] != g['prereg']['commit']
                                     or ent['path'] != g['prereg']['path']):
                aborts.append('12 freeze recorded prereg as %s@%s but grid.json points at %s@%s'
                              % (ent['path'], ent['commit'], g['prereg']['path'],
                                 g['prereg']['commit']))
            blob = git_show(ent['commit'], ent['path'])
            if blob is None:
                aborts.append('12 %s: %s@%s cannot be read' % (name, ent['path'], ent['commit']))
                continue
            got = hashlib.sha256(blob).hexdigest()
            if got != ent['sha256']:
                aborts.append('12 %s changed since freeze: %s@%s hashes to %s, recorded %s'
                              % (name, ent['path'], ent['commit'], got[:16], ent['sha256'][:16]))

    seg = {}
    for r in rows:
        if r['status'] == 'ok':
            seg[r['segment']] = seg.get(r['segment'], 0) + 1
    done = {r['run_id'] for r in rows if r['status'] == 'ok'}
    print('rows=%d cols=%d queued=%d ok=%d (A=%d B=%d) pending=%d'
          % (len(rows), len(header), len(qidx), len(done), seg.get('A', 0), seg.get('B', 0),
             len(set(qidx) - done)))
    if v0_ok:
        print('segment A reconciliation: %d/%d cells, max |delta| = %.4f (tolerance %.2f)'
              % (len(v0_ok), len(g['methods']) * len(g['subsets']) * len(g['seeds']),
                 recon_max, g['recon_tolerance']))
        if over_soft:
            print('  exempt (budget-truncated, not ABORT on their own per prereg R2): %s'
                  % ', '.join('%s=%.4f' % ('/'.join(x[1]), x[0]) for x in over_soft))
    if aborts:
        print('ABORT (%d)' % len(aborts))
        for e in aborts:
            print('  -', e)
        return 2
    if errs:
        print('FAILED (%d)' % len(errs))
        for e in errs[:40]:
            print('  -', e)
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
