"""Gate A wave-1 evaluation: G1, G2a, G2b, G3, G4, G5 and the R0 self-check.

Implements the criteria pre-registered in
``docs/paper_b_hc/GATE_A_WAVE1_PLAN.md`` section 9. Every threshold below is
copied from that plan; this file must not become a place where a threshold is
tuned after seeing the numbers. If a criterion changes, change the plan first.

Conventions (from the plan):

* main metric is ``val_engine_mean_rmse`` from ``close.json`` (engine-level mean,
  select-on-val), falling back to ``metrics.json`` for runs that closed without
  a close file;
* ``delta(A -> B) = (RMSE_A - RMSE_B) / RMSE_A``, in percent; **positive means B
  has the lower RMSE**;
* single seed, direction only -- no significance claim is made anywhere here;
* A0 is a reference line and enters G3 only (R3);
* A3p is the frozen (pre-norm) arm and enters G2b and the V0-NoHI reference
  only, never a cycle-encoding comparison (hard constraint in the plan).

Usage::

    python -m paper_b.experiments.gate_a_report --root experiments/hc
    python -m paper_b.experiments.gate_a_report --root experiments/hc --md out.md

The tool is read-only with respect to run directories. It writes only the
``--csv`` / ``--md`` paths it is given.
"""
import argparse
import json
import re
from pathlib import Path

VARIANTS = ['A0', 'A1', 'A2', 'A3a', 'A3p', 'A4', 'A5']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
CYCLE_ARMS = ['A2', 'A3a', 'A4', 'A5']          # post-norm cycle arms
FROZEN_ARM = 'A3p'                               # pre-norm, reference only
REFERENCE_ARM = 'A0'                             # no backbone, G3 only
MAIN_TABLE = ['A0', 'A1', 'A2', 'A3a', 'A4', 'A5']

# --- pre-registered thresholds (GATE_A_WAVE1_PLAN.md section 9) -------------
G1_MEAN_MIN = 1.0        # percent
G1_SUBSET_FLOOR = -2.0   # percent, no subset may be worse than this
G2_MEAN_MIN = 1.0        # percent
G2_SIGN_MIN = 3          # of 4 subsets
G3_DELTA_MIN = -10.0     # percent; below this A0 is clearly worse -> G3 passes
G4_MEAN_ABS_MIN = 0.5
G4_FRAC_NEG_MIN = 0.5
G5_SIGN_MIN = 3          # of 4

R0_REFERENCE = 10.895096839289387   # HC-C-A3p-FD001-s101-20261112T0030Z
R0_RUN = ('A3p', 'FD001')
R0_BASELINE_RUN_ID = 'HC-C-A3p-FD001-s101-20261112T0030Z'

RUN_RE = re.compile(r'^HC-C-(?P<variant>[A-Za-z0-9]+)-(?P<subset>FD00[1-4])-'
                    r's(?P<seed>\d+)-(?P<suffix>.+)$')


def parse_run_id(run_id):
    m = RUN_RE.match(run_id)
    if not m:
        return None
    return m.group('variant'), m.group('subset'), int(m.group('seed')), m.group('suffix')


def load_run(d):
    """Read one run directory into a record, or None if it has no metrics."""
    parsed = parse_run_id(d.name)
    if not parsed:
        return None
    variant, subset, seed, suffix = parsed
    rec = dict(run_id=d.name, variant=variant, subset=subset, seed=seed,
               suffix=suffix, status=None, closed_utc=None, metric=None,
               intervention=None, baseline_version=None, init_sha256=None,
               path=str(d))
    man = d / 'manifest.json'
    if man.exists():
        try:
            mj = json.loads(man.read_text())
        except Exception:
            mj = {}
        rec['baseline_version'] = mj.get('baseline_version')
        rec['init_sha256'] = mj.get('init_sha256')
    close = d / 'close.json'
    metrics = d / 'metrics.json'
    m = {}
    if close.exists():
        try:
            c = json.loads(close.read_text())
        except Exception:
            c = {}
        rec['status'] = c.get('status')
        rec['closed_utc'] = c.get('closed_utc')
        rec['metric'] = (c.get('metrics') or {}).get('val_engine_mean_rmse')
    if metrics.exists():
        try:
            m = json.loads(metrics.read_text())
        except Exception:
            m = {}
        if rec['metric'] is None:
            rec['metric'] = m.get('val_engine_mean_rmse')
        rec['intervention'] = m.get('cycle_intervention')
    if rec['metric'] is None:
        return None
    return rec


def suffix_rank(suffix):
    """Prefer the canonical W1 run, then the W1r2 retry, then anything ok."""
    if suffix == 'W1':
        return 0
    if suffix.startswith('W1r'):
        return 1
    return 2


def collect(root, baseline=None):
    """Group runs by cell. With ``baseline`` set, only that baseline version is
    accepted; everything else is returned separately so it can be reported as
    excluded instead of silently disappearing."""
    by_cell, excluded = {}, []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or not d.name.startswith('HC-C-'):
            continue
        rec = load_run(d)
        if rec is None:
            continue
        if baseline is not None and rec['baseline_version'] != baseline:
            excluded.append(rec)
            continue
        by_cell.setdefault((rec['variant'], rec['subset'], rec['seed']), []).append(rec)
    return by_cell, excluded


def pick(by_cell, seed=101):
    """One run per (variant, subset): canonical suffix first, ok status first."""
    out = {}
    for v in VARIANTS:
        for s in SUBSETS:
            recs = by_cell.get((v, s, seed), [])
            if not recs:
                out[(v, s)] = None
                continue
            recs = sorted(recs, key=lambda r: (r['status'] != 'ok',
                                               suffix_rank(r['suffix']),
                                               r['closed_utc'] or '~'))
            out[(v, s)] = recs[0]
    return out


def rmse(chosen, variant, subset):
    r = chosen.get((variant, subset))
    return None if r is None else r['metric']


def delta(a, b):
    """Percent improvement of b over a; positive = b better."""
    if a is None or b is None or a == 0:
        return None
    return 100.0 * (a - b) / a


def signs(values):
    return [None if v is None else ('+' if v > 0 else ('-' if v < 0 else '0'))
            for v in values]


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'


def fmt_pct(x):
    return '—' if x is None else f'{x:+.2f}%'


def build(chosen, excluded=()):
    lines = []
    add = lines.append

    add('# Gate A wave-1 evaluation\n')
    add('Generated by `paper_b.experiments.gate_a_report`. Thresholds are the')
    add('pre-registered ones in `docs/paper_b_hc/GATE_A_WAVE1_PLAN.md` section 9.')
    add('Single seed, direction only: nothing here is a significance claim.\n')

    # provenance
    add('## 0. Runs used\n')
    add('| variant | ' + ' | '.join(SUBSETS) + ' |')
    add('|---|' + '---|' * len(SUBSETS))
    for v in VARIANTS:
        cells = []
        for s in SUBSETS:
            r = chosen.get((v, s))
            if r is None:
                cells.append('missing')
            elif r['status'] != 'ok':
                cells.append(f"`{r['suffix']}` {r['status']}")
            else:
                cells.append(f"`{r['suffix']}`")
        add(f'| {v} | ' + ' | '.join(cells) + ' |')
    add('')

    if excluded:
        add(f'Excluded (baseline_version != the wave baseline): '
            f'{len(excluded)} run(s) not used.\n')
        add('| run_id | baseline_version | metric |')
        add('|---|---|---:|')
        for r in sorted(excluded, key=lambda r: r['run_id']):
            add(f'| `{r["run_id"]}` | {r["baseline_version"]} | {fmt(r["metric"])} |')
        add('')

    # main table
    add('## 1. Main table: validation engine-mean RMSE\n')
    add('| variant | ' + ' | '.join(SUBSETS) + ' | mean |')
    add('|---|' + '---:|' * (len(SUBSETS) + 1))
    for v in MAIN_TABLE + [FROZEN_ARM]:
        vals = [rmse(chosen, v, s) for s in SUBSETS]
        add(f'| {v} | ' + ' | '.join(fmt(x) for x in vals) + f' | {fmt(mean_or_none(vals))} |')
    add('')
    add('A0 is a reference line (no backbone) and is excluded from every')
    add('treatment comparison except G3. A3p is the frozen pre-norm arm and is')
    add('reported here only for the R0 check and the V0-NoHI reference.\n')

    verdicts = {}

    # G1
    add('## 2. G1 — is the cycle path useful? A2 vs A1\n')
    d1 = [delta(rmse(chosen, 'A1', s), rmse(chosen, 'A2', s)) for s in SUBSETS]
    add('| | ' + ' | '.join(SUBSETS) + ' | mean |')
    add('|---|' + '---:|' * (len(SUBSETS) + 1))
    add('| d(A1->A2) | ' + ' | '.join(fmt_pct(x) for x in d1) + f' | {fmt_pct(mean_or_none(d1))} |')
    add('')
    m1 = mean_or_none(d1)
    complete1 = all(x is not None for x in d1)
    g1_pass = (complete1 and m1 is not None and m1 >= G1_MEAN_MIN
               and min(d1) >= G1_SUBSET_FLOOR)
    verdicts['G1'] = 'PASS' if g1_pass else ('PENDING' if not complete1 else 'FAIL')
    add(f'Criterion: mean >= {G1_MEAN_MIN:+.1f}% and no subset < {G1_SUBSET_FLOOR:+.1f}%.')
    add(f'**G1 = {verdicts["G1"]}**'
        + ('' if complete1 else ' (incomplete: some subsets missing)') + '\n')

    # G2a
    add('## 3. G2a — Fourier vs linear age at the same (post-norm) injection point: A3a vs A2\n')
    d2 = [delta(rmse(chosen, 'A2', s), rmse(chosen, 'A3a', s)) for s in SUBSETS]
    add('| | ' + ' | '.join(SUBSETS) + ' | mean |')
    add('|---|' + '---:|' * (len(SUBSETS) + 1))
    add('| d(A2->A3a) | ' + ' | '.join(fmt_pct(x) for x in d2) + f' | {fmt_pct(mean_or_none(d2))} |')
    add('')
    m2 = mean_or_none(d2)
    complete2 = all(x is not None for x in d2)
    same_sign2 = len([x for x in d2 if x is not None and x > 0])
    g2_pass = (complete2 and m2 is not None and m2 >= G2_MEAN_MIN
               and same_sign2 >= G2_SIGN_MIN)
    verdicts['G2a'] = 'PASS' if g2_pass else ('PENDING' if not complete2 else 'FAIL')
    add(f'Criterion: mean >= {G2_MEAN_MIN:+.1f}% and at least {G2_SIGN_MIN}/4 subsets')
    add(f'share the sign. Positive subsets: {same_sign2}/4.')
    add(f'**G2a = {verdicts["G2a"]}**'
        + ('' if complete2 else ' (incomplete: some subsets missing)') + '\n')
    if verdicts['G2a'] == 'FAIL':
        add('Per the plan, a failed G2a means Fourier may not be claimed as the')
        add('innovation over a plain age encoding.\n')

    # G2b
    add('## 4. G2b — injection point at equal encoding form: A3p vs A3a (no threshold)\n')
    d3 = [delta(rmse(chosen, 'A3a', s), rmse(chosen, FROZEN_ARM, s)) for s in SUBSETS]
    add('| | ' + ' | '.join(SUBSETS) + ' | mean |')
    add('|---|' + '---:|' * (len(SUBSETS) + 1))
    add('| d(A3a->A3p) | ' + ' | '.join(fmt_pct(x) for x in d3) + f' | {fmt_pct(mean_or_none(d3))} |')
    add('')
    add('Signs: ' + ' '.join(s or '·' for s in signs(d3)) + '. No threshold is')
    add('applied: this is a single-seed preliminary contrast for')
    add('`CYCLE_DEPENDENCE_STUDY.md`, not Gate A main-table evidence.\n')

    # G3
    add('## 5. G3 — age shortcut check: A0 vs A1\n')
    d4 = [delta(rmse(chosen, 'A1', s), rmse(chosen, REFERENCE_ARM, s)) for s in SUBSETS]
    add('| | ' + ' | '.join(SUBSETS) + ' | mean |')
    add('|---|' + '---:|' * (len(SUBSETS) + 1))
    add('| d(A1->A0) | ' + ' | '.join(fmt_pct(x) for x in d4) + f' | {fmt_pct(mean_or_none(d4))} |')
    add('')
    m4 = mean_or_none(d4)
    complete4 = all(x is not None for x in d4)
    # G3 passes when A0 is clearly worse than A1, i.e. the delta is very negative.
    g3_pass = complete4 and m4 is not None and m4 < G3_DELTA_MIN
    verdicts['G3'] = 'PASS' if g3_pass else ('PENDING' if not complete4 else 'FAIL')
    add(f'Criterion: d(A1->A0) < {G3_DELTA_MIN:+.1f}% (A0 clearly worse).')
    add(f'**G3 = {verdicts["G3"]}**'
        + ('' if complete4 else ' (incomplete: some subsets missing)') + '\n')
    if verdicts['G3'] == 'FAIL':
        add('A failed G3 means the benchmark can be solved by age alone, so the')
        add('cycle conclusions are void.\n')

    # G4
    add('## 6. G4 — is the cycle path actually used? intervention direction\n')
    add('| variant | ' + ' | '.join(SUBSETS) + ' |')
    add('|---|' + '---|' * len(SUBSETS))
    g4_rows = []
    for v in ['A1'] + CYCLE_ARMS + [FROZEN_ARM]:
        cells = []
        for s in SUBSETS:
            r = chosen.get((v, s))
            iv = (r or {}).get('intervention')
            if iv is None:
                cells.append('none' if r is not None else 'missing')
                g4_rows.append((v, s, None, None))
            else:
                cells.append(f"|d|={iv.get('mean_abs_delta', float('nan')):.2f} "
                             f"fn={iv.get('frac_negative', float('nan')):.2f}")
                g4_rows.append((v, s, iv.get('mean_abs_delta'), iv.get('frac_negative')))
        add(f'| {v} | ' + ' | '.join(cells) + ' |')
    add('')
    a1_ok = True
    for v, s, mad, _fn in g4_rows:
        if v == 'A1' and mad not in (None, 0.0):
            a1_ok = False
    main_ok, main_complete = True, True
    for v, s, mad, fn in g4_rows:
        if v not in CYCLE_ARMS:
            continue
        if mad is None or fn is None:
            main_complete = False
            continue
        if not (mad >= G4_MEAN_ABS_MIN and fn >= G4_FRAC_NEG_MIN):
            main_ok = False
    g4_pass = a1_ok and main_ok and main_complete
    verdicts['G4'] = 'PASS' if g4_pass else ('PENDING' if not main_complete else 'FAIL')
    add(f'Criterion: every cycle arm has mean|d| >= {G4_MEAN_ABS_MIN} and '
        f'frac_negative >= {G4_FRAC_NEG_MIN}; A1 has mean|d| = 0.')
    add(f'A1 clean: {a1_ok}. Cycle arms all meet the bound: {main_ok}.')
    add(f'**G4 = {verdicts["G4"]}**'
        + ('' if main_complete else ' (incomplete: some cells missing)') + '\n')
    if verdicts['G4'] == 'FAIL':
        add('A failed G4 means the model does not use the cycle path in the')
        add('physically correct direction, so that arm\'s conclusion is void.\n')

    # G5
    add('## 7. G5 — sign stability across subsets\n')
    s1 = signs(d1)
    s2 = signs(d2)
    n1 = len([x for x in s1 if x == '+'])
    n2 = len([x for x in s2 if x == '+'])
    complete5 = complete1 and complete2
    g5_pass = complete5 and n1 >= G5_SIGN_MIN and n2 >= G5_SIGN_MIN
    verdicts['G5'] = ('PASS' if g5_pass else
                      ('PENDING' if not complete5 else 'FAIL'))
    add(f'G1 signs {s1} -> {n1}/4 positive; G2a signs {s2} -> {n2}/4 positive.')
    add(f'Criterion: at least {G5_SIGN_MIN}/4 consistent for both.')
    add(f'**G5 = {verdicts["G5"]}**'
        + ('' if complete5 else ' (incomplete: G1 or G2a has missing subsets)') + '\n')

    # R0
    add('## 8. R0 — launch self-check\n')
    r0 = chosen.get(R0_RUN)
    if r0 is None:
        verdicts['R0'] = 'PENDING'
        add('R0 run missing.')
    else:
        got = r0['metric']
        exact = got == R0_REFERENCE
        verdicts['R0'] = 'PASS' if exact else 'FAIL'
        add(f'`{r0["run_id"]}` = {got!r}')
        add(f'Reference `{R0_BASELINE_RUN_ID}` = {R0_REFERENCE!r}')
        add(f'**R0 = {verdicts["R0"]}**' + ('' if exact else ' (bitwise mismatch: stop the wave)') + '\n')

    # summary
    add('## 9. Summary\n')
    add('| gate | verdict |')
    add('|---|---|')
    for k in ('R0', 'G1', 'G2a', 'G3', 'G4', 'G5'):
        add(f'| {k} | **{verdicts.get(k, "—")}** |')
    add('| G2b | informational (no threshold) |')
    add('')
    add('Any `PENDING` above means the wave is not complete yet; the verdict')
    add('belongs to 主控 once all 28 cells have landed.')
    return '\n'.join(lines) + '\n', verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiments/hc')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--baseline', default='v9',
                    help="only runs with this manifest baseline_version are used; "
                         "'' disables the filter")
    ap.add_argument('--md', default=None, help='write the markdown report here')
    ap.add_argument('--csv', default=None, help='write the chosen-run table here')
    a = ap.parse_args()

    by_cell, excluded = collect(a.root, baseline=(a.baseline or None))
    chosen = pick(by_cell, seed=a.seed)
    text, verdicts = build(chosen, excluded)

    print(text)
    if a.md:
        Path(a.md).write_text(text, encoding='utf-8')
        print(f'[gate_a_report] wrote {a.md}')
    if a.csv:
        rows = ['variant,subset,seed,run_id,status,suffix,val_engine_mean_rmse']
        for (v, s), r in sorted(chosen.items()):
            if r is None:
                rows.append(f'{v},{s},{a.seed},MISSING,,,')
            else:
                rows.append(f'{v},{s},{a.seed},{r["run_id"]},{r["status"]},'
                            f'{r["suffix"]},{r["metric"]!r}')
        Path(a.csv).write_text('\n'.join(rows) + '\n', encoding='utf-8')
        print(f'[gate_a_report] wrote {a.csv}')


if __name__ == '__main__':
    main()
