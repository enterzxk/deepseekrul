"""Gate 1: is CADF plug-and-play across the five reproduced SOTA backbones?

=============================================================================
SIGN CONVENTION -- read this first.
=============================================================================
The pre-registration states two thresholds that use OPPOSITE conventions:

    "not degraded"   :  delta <= +0.05     (reads as module - baseline)
    "positive gain"  :  delta  >  0        (reads as baseline - module)

If both are taken literally, the same delta has to be both negative and
positive. This script pins ONE convention and translates the other:

    delta := baseline - module            (POSITIVE = the module is better)

    not degraded   :  delta_EOL_MAE >= -0.05     (i.e. module - baseline <= 0.05)
    positive gain  :  delta_RMSE    >  0   AND  paired win rate > 0.50

Everything printed below is in this convention. Do not re-derive the guardrail
from the "+0.05" in the plan without translating it first.

=============================================================================
Exit codes -- three distinct meanings, never reuse one for another
=============================================================================
    0  PASS     gate passed, or the segment-A reconcile is within tolerance
    1  FAIL     the gate was measured and the criterion was not met
                (this is a RESULT: the module did not deliver)
    2  ABORT    a precondition is unmet: bad/missing input, or the segment-A
                reconcile is outside tolerance. NO result exists -- do not
                report a FAIL, and do not start segment B.

A driver that only checks "exit != 0" cannot tell "the module failed" from
"the environment drifted". That distinction is the whole point of the codes.

=============================================================================
Input contract (both files: a JSON list of rows)
=============================================================================
    {"method": str,   # aidgn | two_phase | dual_mixer | ffve | fcstgnn
     "subset": str,   # FD001 | FD004
     "seed":   int,   # 101 | 202 | 303
     "engine": int,   # engine id
     "cut":    float, # truncation fraction of that engine's own life, 0..1
     "y":      float, # true RUL at that cut
     "pred":   float} # predicted RUL at that cut

Per-engine, per-cut predictions are required because EOL(0.95) is defined at a
cut, and the paired win rate is defined per engine. A pre-aggregated file
cannot produce either quantity.

Gate: a backbone passes only if BOTH subsets pass (FD001 and FD004), because
B_min already fails FD004 -- so "4 of 6 backbones" is really "4 of the 5 SOTA
backbones pass on both subsets".

=============================================================================
Reconcile mode: segment A must reproduce the frozen baseline first
=============================================================================
    --reconcile-a DIR_OR_JSON --reconcile-frozen DIR [--tol 0.2]

Segment A retrains V0 on lab3090; V0 was originally trained on a40 with a
different Python/torch. Before any CADF cell is run, the 30 V0 cells must
reproduce the frozen report (commit d7bd9c17) per engine. The comparison is on
VALIDATION engine-level RMSE only -- official TEST stays sealed.

The frozen side is read from `validation_metrics.json` and cross-checked
against `validation_predictions.csv` (per-engine RMSE recomputed from
engine_squared_errors, and the mean recomputed from the CSV). A frozen report
that disagrees with its own predictions aborts before any comparison happens,
so a stale or truncated CSV cannot silently become the reference.

Tolerance is per ENGINE, not per aggregate: an aggregate can hide a swap of
+0.2 and -0.2 between two engines, and the signed mean is printed precisely so
that a constant offset (environment drift) is distinguishable from scatter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ('aidgn', 'two_phase', 'dual_mixer', 'ffve', 'fcstgnn')
SUBSETS = ('FD001', 'FD004')
SEEDS = (101, 202, 303)
EOL_CUT = 0.95
NOT_DEGRADED = -0.05      # delta >= this
MIN_PASS_METHODS = 4
BOOT = 10000
BOOT_SEED = 20261122

EXIT_OK = 0
EXIT_GATE_FAIL = 1
EXIT_ABORT = 2
RECON_TOL = 0.2
N_CELLS = len(METHODS) * len(SUBSETS) * len(SEEDS)


def die(msg):
    """Abort: no result exists. Never exit 1 for this -- that means FAIL."""
    print(msg, file=sys.stderr)
    raise SystemExit(EXIT_ABORT)


def load(path: Path):
    rows = json.loads(path.read_text())
    need = {'method', 'subset', 'seed', 'engine', 'cut', 'y', 'pred'}
    seen = set()
    for i, r in enumerate(rows):
        missing = need - set(r)
        if missing:
            die(f'{path}: row {i} missing {sorted(missing)}')
        key = (r['method'], r['subset'], int(r['seed']), int(r['engine']), round(float(r['cut']), 6))
        if key in seen:
            die(f'{path}: duplicate row {key}')
        seen.add(key)
    mx = max(float(r['cut']) for r in rows)
    if mx > 1.0 + 1e-9:
        # The frozen Paper-B CSVs store `cut` as an ABSOLUTE CYCLE (e.g. 75, 103,
        # 132 for one FD001 engine), not as a fraction. Feeding that here would
        # make "nearest to 0.95" pick the SMALLEST cycle -- i.e. the EOL guardrail
        # would be read at the earliest truncation instead of the latest, and the
        # gate would still print a verdict. Recovering the fraction needs each
        # engine's life; guessing it is how a silent inversion happens.
        die(f'{path}: cut looks like an ABSOLUTE CYCLE (max {mx:.1f}), not a fraction '
            f'in (0,1]. EOL({EOL_CUT}) is defined on the fraction; emit '
            f'cut = cycle / life per engine instead of converting by guesswork.')
    return rows


def by_cell(rows):
    """{(method, subset, seed): {engine: [(cut, y, pred), ...]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(r['method'], r['subset'], int(r['seed']))][int(r['engine'])].append(
            (float(r['cut']), float(r['y']), float(r['pred'])))
    return out


def cell_metrics(cell):
    """Per-engine RMSE over all cuts, and the |error| at the cut nearest 0.95."""
    rmse, eol = {}, {}
    for eng, pts in cell.items():
        pts = sorted(pts)
        rmse[eng] = float(np.sqrt(np.mean([(p - y) ** 2 for _, y, p in pts])))
        near = min(pts, key=lambda t: abs(t[0] - EOL_CUT))
        # Remaining RUL falls monotonically as the cut advances, so the LATEST
        # truncation of an engine is the one with the smallest y. If the point
        # nearest 0.95 does not carry that minimum, the guardrail is not being
        # read at the last truncation -- wrong cut space, reversed ordering, or a
        # missing final cut. Catch it here rather than in the verdict.
        y_min = min(p[1] for p in pts)
        if near[1] > y_min + 1e-6:
            die(f'engine {eng}: cut nearest {EOL_CUT} has y={near[1]:.1f}, but this engine '
                f'has a later cut with y={y_min:.1f} -- the EOL guardrail is being read at '
                f'the wrong truncation')
        eol[eng] = float(abs(near[2] - near[1]))
    return rmse, eol


def boot_ci(d, seed):
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------- reconcile

def seed_dir(parent: Path, seed: int) -> Path:
    for cand in (f's{seed}', f'seed{seed}', f'seed_{seed}', str(seed)):
        d = parent / cand
        if d.is_dir():
            return d
    hits = sorted(p for p in parent.glob(f'*{seed}') if p.is_dir())
    if len(hits) == 1:
        return hits[0]
    die(f'reconcile: cannot resolve seed {seed} under {parent} '
        f'(candidates={[p.name for p in hits]})')


def engine_rmse_from_csv(path: Path):
    """{unit: RMSE over that engine's cuts}, from validation_predictions.csv."""
    if not path.exists():
        die(f'reconcile: missing {path}')
    per = defaultdict(list)
    with path.open(newline='') as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames:
            die(f'{path}: empty file')
        miss = [c for c in ('unit', 'y_true', 'y_pred') if c not in rd.fieldnames]
        if miss:
            die(f'{path}: missing columns {miss} (have {rd.fieldnames})')
        for i, r in enumerate(rd):
            try:
                u = int(float(r['unit']))
                y = float(r['y_true'])
                p = float(r['y_pred'])
            except (TypeError, ValueError) as exc:
                die(f'{path}: row {i} unparsable ({exc})')
            per[u].append((y, p))
    if not per:
        die(f'{path}: no rows')
    return {u: float(np.sqrt(np.mean([(p - y) ** 2 for y, p in pts])))
            for u, pts in per.items()}


def frozen_engine_rmse(cell_dir: Path):
    """Per-engine RMSE from the frozen tree, cross-checked against its metrics."""
    mj = cell_dir / 'validation_metrics.json'
    csvp = cell_dir / 'validation_predictions.csv'
    if not mj.exists():
        die(f'reconcile: missing {mj}')
    meta = json.loads(mj.read_text())
    for key in ('val_engine_mean_rmse', 'engine_squared_errors'):
        if key not in meta:
            die(f'{mj}: missing {key}')
    per = engine_rmse_from_csv(csvp)
    ref = {int(e['unit']): float(np.sqrt(float(e['sum_squared_error']) / float(e['n'])))
           for e in meta['engine_squared_errors']}
    if set(per) != set(ref):
        die(f'{mj}: engine sets differ (csv={len(per)} metrics={len(ref)})')
    worst = max(abs(per[u] - ref[u]) for u in per)
    if worst > 1e-6:
        die(f'{mj}: predictions disagree with engine_squared_errors '
            f'(max |diff| = {worst:.3e}) -- stale or truncated CSV')
    claimed = float(meta['val_engine_mean_rmse'])
    mine = float(np.mean(list(per.values())))
    if abs(mine - claimed) > 1e-6:
        die(f'{mj}: val_engine_mean_rmse={claimed:.6f} but recomputed {mine:.6f}')
    pooled = meta.get('val_pooled_rmse')
    if isinstance(pooled, (int, float)) and abs(claimed - float(pooled)) < 1e-12:
        print(f'  WARN {cell_dir}: val_engine_mean_rmse == val_pooled_rmse '
              f'(unexpected on validation -- check the metric source)')
    return per


def a_segment_engine_rmse(root: Path):
    """{(method, subset, seed): {engine: rmse}} from a directory tree or a JSON."""
    if root.is_dir():
        out = {}
        for m in METHODS:
            for s in SUBSETS:
                for seed in SEEDS:
                    d = seed_dir(root / m / s, seed)
                    out[(m, s, seed)] = engine_rmse_from_csv(d / 'validation_predictions.csv')
        return out
    if root.exists():
        rows = load(root)
        return {k: cell_metrics(c)[0] for k, c in by_cell(rows).items()}
    die(f'reconcile: no such path {root}')


META_FIELDS = ('epochs', 'patience', 'min_delta', 'window', 'batch', 'lr', 'weight_decay',
               'val_frac', 'split_seed', 'rul_cap')


def _same(x, y):
    try:
        return abs(float(x) - float(y)) <= 1e-12
    except (TypeError, ValueError):
        return x == y


def cell_meta(cell_dir: Path, label: str):
    """Epoch anchor and hyperparameters of one cell.

    Sources, in order: training_metrics.json (authoritative); then history.json
    for epochs_run (one record per completed epoch) and FREEZE.json for
    best_epoch -- both verified equal to training_metrics.json on the frozen
    tree, and used only when the trimmed tree omits the latter. The source
    actually used is reported, so a reconcile that fell back to a weaker file is
    visible rather than implied.
    """
    tm = cell_dir / 'training_metrics.json'
    meta = json.loads(tm.read_text()) if tm.exists() else {}
    src = []
    epochs_run = meta.get('epochs_run')
    if epochs_run is not None:
        src.append('training_metrics')
    else:
        hist = cell_dir / 'history.json'
        if hist.exists():
            h = json.loads(hist.read_text())
            if isinstance(h, list) and h:
                epochs_run = len(h)
                src.append('history')
    best = meta.get('best_epoch')
    if best is not None:
        if 'training_metrics' not in src:
            src.append('training_metrics')
    else:
        fr = cell_dir / 'FREEZE.json'
        if fr.exists():
            best = json.loads(fr.read_text()).get('best_epoch')
            if best is not None:
                src.append('FREEZE')
    if epochs_run is None or best is None:
        die(f'{label}: cannot read epochs_run/best_epoch from training_metrics.json, '
            f'history.json or FREEZE.json -- the stopping epoch is the FIRST drift '
            f'signal, and without it a shifted stop stays invisible')
    cf = cell_dir / 'config.json'
    cfg = json.loads(cf.read_text()) if cf.exists() else {}
    if 'epochs' not in cfg:
        die(f'{label}: {cf} missing "epochs" -- without the ceiling a budget-truncated '
            f'cell cannot be told from a converged one')
    return dict(epochs_run=int(epochs_run), best_epoch=int(best), cfg=cfg,
                src='+'.join(src) or '?')


def check_metadata(a_root: Path, f_root: Path, no_meta: bool):
    """Compare stopping epochs and hyperparameters BEFORE looking at any RMSE.

    A cell that stops at a different epoch has already diverged: comparing its
    RMSE to the anchor only stacks noise on top of a known protocol drift. Returns
    the set of budget-truncated cells, whose RMSE is not decisive on its own --
    they were cut off while still improving, so |delta| there says little about
    the environment.
    """
    if no_meta:
        print('metadata: SKIPPED (--no-meta) -- RMSE-only reconcile, epoch drift invisible\n')
        return set()
    fragile, ep_bad, cfg_bad, n_cmp = set(), [], [], 0
    srcs = defaultdict(int)
    for m in METHODS:
        for s in SUBSETS:
            for seed in SEEDS:
                key = f'{m}/{s}/s{seed}'
                fm = cell_meta(seed_dir(f_root / m / s, seed), f'frozen {key}')
                am = cell_meta(seed_dir(a_root / m / s, seed), f'segment A {key}')
                srcs[f'{fm["src"]}|{am["src"]}'] += 1
                ceiling = fm['cfg'].get('epochs')
                if isinstance(ceiling, (int, float)) and fm['epochs_run'] >= float(ceiling):
                    fragile.add((m, s, seed))
                if (am['epochs_run'], am['best_epoch']) != (fm['epochs_run'], fm['best_epoch']):
                    ep_bad.append(f'{key}: epochs_run {am["epochs_run"]} vs {fm["epochs_run"]}, '
                                  f'best_epoch {am["best_epoch"]} vs {fm["best_epoch"]}')
                for f in META_FIELDS:
                    if f in fm['cfg'] and f in am['cfg']:
                        n_cmp += 1
                        if not _same(fm['cfg'][f], am['cfg'][f]):
                            cfg_bad.append(f'{key}: {f} {am["cfg"][f]!r} vs {fm["cfg"][f]!r}')
    print(f'metadata: epochs_run+best_epoch identical on {N_CELLS - len(ep_bad)}/{N_CELLS} cells; '
          f'{n_cmp} config-field comparisons')
    print(f'epoch sources (frozen|segment A): {dict(srcs)}')
    print(f'budget-truncated (fragile) cells: {len(fragile)} '
          f'{sorted(f"{m}/{s}/s{d}" for m, s, d in fragile)}')
    if ep_bad:
        print('\n'.join('  ' + b for b in ep_bad), file=sys.stderr)
        die(f'RECONCILE: ABORT -- {len(ep_bad)} cell(s) stopped at a different epoch. '
            f'The stopping epoch is the first drift signal; fix the environment or the '
            f'protocol before any RMSE comparison.')
    if cfg_bad:
        print('\n'.join('  ' + b for b in cfg_bad), file=sys.stderr)
        die(f'RECONCILE: ABORT -- {len(cfg_bad)} hyperparameter mismatch(es): wrong protocol')
    return fragile


def reconcile(a_seg: Path, frozen: Path, tol: float, no_meta: bool = False) -> int:
    print('RECONCILE  segment A (V0) vs frozen d7bd9c17  -- VALIDATION ONLY, TEST stays sealed')
    print(f'per-ENGINE tolerance: |delta| <= {tol:.3f} on engine-level RMSE')
    print('delta := A_segment - frozen  (signed; a constant offset = environment drift)\n')
    print(f'{"cell":22} {"eng":>4} {"max|d|":>8} {"mean d":>8} {"worst":>6}  verdict')

    got_all = a_segment_engine_rmse(Path(a_seg))
    fragile = check_metadata(Path(a_seg), Path(frozen), no_meta)
    n_bad = 0
    n_frag = 0
    worst_all = (-1.0, '')
    drift = []
    for m in METHODS:
        for s in SUBSETS:
            for seed in SEEDS:
                key = (m, s, seed)
                if key not in got_all:
                    die(f'reconcile: segment A has no cell {key}')
                got = got_all[key]
                ref = frozen_engine_rmse(seed_dir(Path(frozen) / m / s, seed))
                if set(got) != set(ref):
                    die(f'reconcile {m}/{s}/s{seed}: engine sets differ '
                        f'(A={len(got)} frozen={len(ref)}; '
                        f'only-A={sorted(set(got) - set(ref))[:5]} '
                        f'only-frozen={sorted(set(ref) - set(got))[:5]})')
                d = {u: got[u] - ref[u] for u in ref}
                arr = np.array([d[u] for u in sorted(d)])
                we = max(d, key=lambda u: abs(d[u]))
                over = int((np.abs(arr) > tol).sum())
                if over and key in fragile:
                    n_frag += 1
                    verdict = f'{over} OVER fragile'
                elif over:
                    n_bad += 1
                    verdict = f'{over} OVER'
                else:
                    verdict = 'OK'
                drift.append(float(arr.mean()))
                if abs(d[we]) > worst_all[0]:
                    worst_all = (abs(d[we]), f'{m}/{s}/s{seed} engine {we}')
                print(f'{m:11} {s} s{seed}  {len(ref):4d} {abs(d[we]):8.4f} '
                      f'{arr.mean():+8.4f} {we:6d}  {verdict}')

    print(f'\ncells: {N_CELLS}, outside tolerance: {n_bad} '
          f'(+ {n_frag} fragile, not decisive on their own)')
    print(f'worst per-engine |delta| = {worst_all[0]:.4f} at {worst_all[1]}')
    print(f'systematic drift (mean over all engines) = {np.mean(drift):+.4f}')
    if n_frag:
        print(f'NOTE: {n_frag} budget-truncated cell(s) exceeded tolerance. They were cut off '
              f'while still improving, so their |delta| is not evidence of environment drift; '
              f'they are kept out of the verdict but must be listed in the report.')
    if n_bad:
        print('RECONCILE: FAIL -- environment mismatch, DO NOT start segment B')
        return EXIT_ABORT
    print('RECONCILE: OK -- segment B may proceed')
    return EXIT_OK


# --------------------------------------------------------------------- gate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--module', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--reconcile-a', default=None, metavar='DIR_OR_JSON')
    ap.add_argument('--reconcile-frozen', default=None, metavar='DIR')
    ap.add_argument('--tol', type=float, default=RECON_TOL)
    ap.add_argument('--no-meta', action='store_true',
                    help='skip the epochs_run/best_epoch + hyperparameter check '
                         '(only for a tree that genuinely lacks training_metrics.json)')
    a = ap.parse_args()

    if a.reconcile_a or a.reconcile_frozen:
        if not (a.reconcile_a and a.reconcile_frozen):
            die('reconcile needs BOTH --reconcile-a and --reconcile-frozen')
        if a.baseline or a.module:
            die('reconcile mode does not take --baseline/--module')
        if Path(a.reconcile_a).is_file() and not a.no_meta:
            die('a JSON of rows carries no epochs_run/best_epoch; pass --no-meta to reconcile '
                'it, or point --reconcile-a at the run directory instead')
        return reconcile(Path(a.reconcile_a), Path(a.reconcile_frozen), a.tol, a.no_meta)
    if not (a.baseline and a.module):
        die('need --baseline and --module (or --reconcile-a/--reconcile-frozen)')

    base = by_cell(load(Path(a.baseline)))
    mod = by_cell(load(Path(a.module)))

    print(f'gate1.py sha256 = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}')
    print('convention: delta = baseline - module (POSITIVE = module better)')
    print(f'not degraded: delta_EOL_MAE >= {NOT_DEGRADED:+.2f}   '
          f'positive gain: delta_RMSE > 0 and win rate > 0.50\n')

    cells, per_method = {}, defaultdict(dict)
    for method in METHODS:
        for subset in SUBSETS:
            d_rmse, d_eol, wins, total = [], [], 0, 0
            for seed in SEEDS:
                key = (method, subset, seed)
                if key not in base or key not in mod:
                    die(f'missing cell {key} '
                        f'(baseline={key in base}, module={key in mod})')
                rb, eb = cell_metrics(base[key])
                rm_, em = cell_metrics(mod[key])
                eng = sorted(set(rb) & set(rm_))
                if len(eng) != len(rb) or len(eng) != len(rm_):
                    die(f'{key}: engine sets differ '
                        f'(base={len(rb)} mod={len(rm_)} common={len(eng)})')
                d_rmse.append(np.array([rb[u] - rm_[u] for u in eng]))
                d_eol.append(np.array([eb[u] - em[u] for u in eng]))
                wins += sum(1 for u in eng if rm_[u] < rb[u])
                total += len(eng)
            dr = np.mean(np.stack(d_rmse, axis=0), axis=0)
            de = np.mean(np.stack(d_eol, axis=0), axis=0)
            lo_r, hi_r = boot_ci(dr, BOOT_SEED)
            lo_e, hi_e = boot_ci(de, BOOT_SEED + 1)
            win = wins / total
            ok_eol = float(de.mean()) >= NOT_DEGRADED
            ok_gain = (float(dr.mean()) > 0) and (win > 0.50)
            cells[(method, subset)] = dict(
                delta_rmse=float(dr.mean()), rmse_ci=[lo_r, hi_r],
                delta_eol_mae=float(de.mean()), eol_ci=[lo_e, hi_e],
                win_rate=float(win), not_degraded=bool(ok_eol), positive_gain=bool(ok_gain),
                passed=bool(ok_eol and ok_gain), n_engines=int(len(eng)))
            per_method[method][subset] = cells[(method, subset)]
            c = cells[(method, subset)]
            print(f'{method:11} {subset}  dRMSE {c["delta_rmse"]:+.3f} '
                  f'[{lo_r:+.3f},{hi_r:+.3f}]  dEOL {c["delta_eol_mae"]:+.3f} '
                  f'[{lo_e:+.3f},{hi_e:+.3f}]  win {c["win_rate"]:.0%}  '
                  f'{"PASS" if c["passed"] else "FAIL"}'
                  f'{"  (eol)" if not c["not_degraded"] else ""}'
                  f'{"  (gain)" if not c["positive_gain"] else ""}')

    n_pass = sum(1 for m in METHODS
                 if all(per_method[m].get(s, {}).get('passed') for s in SUBSETS))
    print(f'\nbackbones passing on BOTH subsets: {n_pass}/{len(METHODS)} '
          f'(need >= {MIN_PASS_METHODS})')
    verdict = 'PASS' if n_pass >= MIN_PASS_METHODS else 'FAIL'
    print(f'GATE 1: {verdict}')

    if a.json:
        Path(a.json).write_text(json.dumps(dict(
            convention='delta = baseline - module (positive = module better)',
            thresholds=dict(not_degraded_ge=NOT_DEGRADED, win_rate_gt=0.50,
                            min_pass_methods=MIN_PASS_METHODS),
            cells={f'{m}/{s}': v for (m, s), v in cells.items()},
            n_pass=n_pass, verdict=verdict,
        ), indent=1), encoding='utf-8')
        print(f'wrote {a.json}')
    return EXIT_OK if verdict == 'PASS' else EXIT_GATE_FAIL


if __name__ == '__main__':
    raise SystemExit(main())
