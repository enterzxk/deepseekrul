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
    --reconcile-a DIR --reconcile-frozen DIR [--tol 0.2] [--no-meta]

Segment A retrains V0 on lab3090; V0 was originally trained on a40 with a
different Python/torch. Before any CADF cell is run, the 30 V0 cells must
reproduce the frozen report (commit d7bd9c17). The comparison uses VALIDATION
data only -- official TEST stays sealed.

WHY THE SELECTED CHECKPOINT IS NOT THE CRITERION
------------------------------------------------
Measured on the first A-segment cell (aidgn/FD001/s101): the selection metric's
per-epoch sd is 0.9377, and the run-to-run argmin differs by 0.3983. Taking the
argmin of a curve with sd ~1 is inherently unstable, so an |argmin delta| <= 0.2
rule fails a run whose *level* is off by only 0.1247. The argmin difference is
therefore a DIAGNOSTIC, not a gate. The gates are:

 1. internal consistency  validation_metrics.json must equal history.json at
                          best_epoch, and validation_predictions.csv must equal
                          engine_squared_errors (both sides)
 2. protocol              config.json fields identical (epochs, patience,
                          min_delta, window, batch, lr, weight_decay, ...)
 3. convergence           |epochs_run| and |best_epoch| within 2 epochs
 4. level                 mean of the selection metric over COMMON epochs
                          within +/-0.2 -- this is the environment signal
 5. structure             Spearman rank correlation of per-engine curve means
                          >= 0.90: the same engines must still be the hard ones

Gate 4 catches a data/scaler/numerics shift; gate 5 catches a change in which
engines are hard. Both are invariant to the argmin wandering, and gate 5 is
invariant to a uniform level shift as well.

Budget-truncated cells (epochs_run hit the ceiling while still improving) are
listed separately: their argmin is the most unstable, so their argmin delta is
reported but never decisive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
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
RECON_TOL = 0.2           # curve-level tolerance (gate 4)
EPOCH_TOL = 2             # gate 3
RANK_MIN = 0.90           # gate 5
N_CELLS = len(METHODS) * len(SUBSETS) * len(SEEDS)

META_FIELDS = ('epochs', 'patience', 'min_delta', 'window', 'batch', 'lr', 'weight_decay',
               'val_frac', 'split_seed', 'rul_cap')


def die(msg):
    """Abort: no result exists. Never exit 1 for this -- that means FAIL."""
    print(msg, file=sys.stderr)
    raise SystemExit(EXIT_ABORT)


# ------------------------------------------------------------------ the gate

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


def _same(x, y):
    try:
        return abs(float(x) - float(y)) <= 1e-12
    except (TypeError, ValueError):
        return x == y


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
            f'history.json or FREEZE.json -- the stopping epoch is a drift signal, and '
            f'without it a shifted stop stays invisible')
    cf = cell_dir / 'config.json'
    cfg = json.loads(cf.read_text()) if cf.exists() else {}
    if 'epochs' not in cfg:
        die(f'{label}: {cf} missing "epochs" -- without the ceiling a budget-truncated '
            f'cell cannot be told from a converged one')
    return dict(epochs_run=int(epochs_run), best_epoch=int(best), cfg=cfg,
                src='+'.join(src) or '?')


def cell_curves(cell_dir: Path, label: str):
    """{epoch: (val_engine_mean_rmse, {unit: rmse})} from history.json."""
    hp = cell_dir / 'history.json'
    if not hp.exists():
        die(f'{label}: missing {hp} -- the curve level is the environment signal; '
            f'without it only --no-meta (RMSE-only) reconcile is possible')
    hist = json.loads(hp.read_text())
    if not isinstance(hist, list) or not hist:
        die(f'{hp}: expected a non-empty list of epoch records')
    out = {}
    for rec in hist:
        for k in ('epoch', 'val_engine_mean_rmse', 'engine_squared_errors'):
            if k not in rec:
                die(f'{hp}: epoch record missing {k}')
        per = {int(x['unit']): float(np.sqrt(float(x['sum_squared_error']) / float(x['n'])))
               for x in rec['engine_squared_errors']}
        out[int(rec['epoch'])] = (float(rec['val_engine_mean_rmse']), per)
    return out


def check_consistency(cell_dir: Path, label: str, hist):
    """metrics.json must equal history at best_epoch, and the CSV must equal both.

    A predictions file that does not correspond to the reported selection is how
    a stale or half-written artifact passes for a good run.
    """
    mj = cell_dir / 'validation_metrics.json'
    if not mj.exists():
        die(f'{label}: missing {mj}')
    meta = json.loads(mj.read_text())
    for k in ('val_engine_mean_rmse', 'engine_squared_errors'):
        if k not in meta:
            die(f'{mj}: missing {k}')
    claimed = float(meta['val_engine_mean_rmse'])
    # The reported selection must BE one of the recorded epochs. On the frozen
    # tree it is bit-exact; anything else means the metrics and the curve were
    # produced by different runs, and every downstream comparison inherits it.
    if not any(abs(hist[e][0] - claimed) < 1e-6 for e in hist):
        die(f'{label}: validation_metrics.val_engine_mean_rmse={claimed:.6f} matches no '
            f'epoch in history.json -- the reported selection and the recorded curve '
            f'come from different runs')
    ref = {int(x['unit']): float(np.sqrt(float(x['sum_squared_error']) / float(x['n'])))
           for x in meta['engine_squared_errors']}
    per = engine_rmse_from_csv(cell_dir / 'validation_predictions.csv')
    if set(per) != set(ref):
        die(f'{label}: engine sets differ (csv={len(per)} metrics={len(ref)})')
    worst = max(abs(per[u] - ref[u]) for u in per)
    if worst > 1e-6:
        die(f'{label}: predictions disagree with engine_squared_errors '
            f'(max |diff| = {worst:.3e}) -- stale or truncated CSV')
    claimed = float(meta['val_engine_mean_rmse'])
    mine = float(np.mean(list(per.values())))
    if abs(mine - claimed) > 1e-6:
        die(f'{label}: val_engine_mean_rmse={claimed:.6f} but recomputed {mine:.6f}')
    return per, ref


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def check_cells(a_root: Path, f_root: Path, no_meta: bool):
    """Gates 1-5 over the 30 cells. Dies (exit 2) on the first failing gate."""
    if no_meta:
        print('cell checks: SKIPPED (--no-meta) -- RMSE-only, no epoch/curve/rank gate\n')
        return [], set(), {}
    bad, fragile, curves = [], set(), {}
    counts = defaultdict(int)
    n_cmp = 0
    srcs = defaultdict(int)
    for m in METHODS:
        for s in SUBSETS:
            for seed in SEEDS:
                key = f'{m}/{s}/s{seed}'
                fd = seed_dir(f_root / m / s, seed)
                ad = seed_dir(a_root / m / s, seed)
                fm = cell_meta(fd, f'frozen {key}')
                am = cell_meta(ad, f'segment A {key}')
                srcs[f'{fm["src"]}|{am["src"]}'] += 1

                for f in META_FIELDS:
                    if f in fm['cfg'] and f in am['cfg']:
                        n_cmp += 1
                        if not _same(fm['cfg'][f], am['cfg'][f]):
                            counts['protocol'] += 1
                            bad.append(f'{key}: config.{f} {am["cfg"][f]!r} != {fm["cfg"][f]!r}')

                if (abs(am['epochs_run'] - fm['epochs_run']) > EPOCH_TOL
                        or abs(am['best_epoch'] - fm['best_epoch']) > EPOCH_TOL):
                    counts['stopping epoch'] += 1
                    bad.append(f'{key}: stopping epoch drifted (epochs_run '
                               f'{am["epochs_run"]} vs {fm["epochs_run"]}, best '
                               f'{am["best_epoch"]} vs {fm["best_epoch"]}, tol {EPOCH_TOL})')

                ceiling = fm['cfg'].get('epochs')
                if isinstance(ceiling, (int, float)) and fm['epochs_run'] >= float(ceiling):
                    fragile.add((m, s, seed))

                fc, ac = cell_curves(fd, f'frozen {key}'), cell_curves(ad, f'segment A {key}')
                check_consistency(fd, f'frozen {key}', fc)
                per_a, per_f = check_consistency(ad, f'segment A {key}', ac)

                common = sorted(set(fc) & set(ac))
                if len(common) < 5:
                    bad.append(f'{key}: only {len(common)} common epochs')
                    continue
                lev_f = float(np.mean([fc[e][0] for e in common]))
                lev_a = float(np.mean([ac[e][0] for e in common]))
                dlev = lev_a - lev_f
                units = sorted(fc[common[0]][1])
                if units != sorted(ac[common[0]][1]):
                    bad.append(f'{key}: engine sets differ between curves')
                    continue
                cf_ = np.array([float(np.mean([fc[e][1][u] for e in common])) for u in units])
                ca_ = np.array([float(np.mean([ac[e][1][u] for e in common])) for u in units])
                rho = _spearman(cf_, ca_)
                # diagnostic only: what the SELECTED checkpoints differ by
                arg_f = float(min(fc[e][0] for e in fc))
                arg_a = float(min(ac[e][0] for e in ac))
                d_sel = float(np.mean([per_a[u] - per_f[u] for u in sorted(per_f)]))
                curves[key] = dict(level=dlev, rho=rho, n_common=len(common),
                                   argmin_gap=arg_a - arg_f, selected_delta=d_sel)
                if abs(dlev) > RECON_TOL:
                    counts['curve level'] += 1
                    bad.append(f'{key}: curve level {dlev:+.4f} outside +/-{RECON_TOL}')
                if not (rho >= RANK_MIN):
                    counts['rank'] += 1
                    bad.append(f'{key}: per-engine rank correlation {rho:.3f} < {RANK_MIN}')

    print(f'protocol: {n_cmp} config-field comparisons')
    print(f'epoch sources (frozen|segment A): {dict(srcs)}')
    print(f'budget-truncated (fragile) cells: {len(fragile)} '
          f'{sorted(f"{m}/{s}/s{d}" for m, s, d in fragile)}')
    # Per-gate counts: a cumulative total under a single gate's label reads as
    # "the protocol changed" even when only the level drifted.
    print(f'gate violations: {dict(counts) if counts else "none"} (total {len(bad)})')
    if bad:
        print('\n'.join('  ' + b for b in bad), file=sys.stderr)
        print(f'RECONCILE: ABORT -- {len(bad)} gate violation(s); segment A does not '
              f'reproduce the anchor, so segment B would measure the environment',
              file=sys.stderr)
    return bad, fragile, curves


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


def write_abort_record(path: Path, batch, violations):
    """Record a failed reconcile so "ran and failed" stays distinct from "not run".

    Deliberately NOT provenance-verified: an abort record must never be able to
    open segment B, and the ledger gates on verdict == 'pass' alone.
    """
    here = Path(__file__).resolve()
    repo = here.parent.parent
    rel = f'{here.parent.name}/{here.name}'
    r = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(
        batch=batch, verdict='abort', cells=N_CELLS,
        gate1_path=rel,
        git_sha=r.stdout.strip() if r.returncode == 0 else None,
        gate1_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
        provenance_verified=False,
        violations=list(violations),
    ), indent=1), encoding='utf-8')
    print(f'wrote {path}  (verdict=abort, {len(violations)} violation(s) -- segment B stays closed)')


def write_record(path: Path, a_seg: Path, frozen: Path, curves, fragile, tol, batch):
    """Write the pass record the run ledger reads, after proving its own provenance.

    The ledger opens segment B by checking that `git show <git_sha>:paper_c/gate1.py`
    hashes to `gate1_sha256`. So the record must not be written unless the script
    on disk IS the committed one -- otherwise it names a commit that does not
    contain the code that produced the verdict, and the check would pass on a lie.
    """
    here = Path(__file__).resolve()
    repo = here.parent.parent
    rel = f'{here.parent.name}/{here.name}'
    disk = hashlib.sha256(here.read_bytes()).hexdigest()
    r = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=repo)
    if r.returncode != 0:
        die('record: cannot resolve git HEAD -- the ledger attributes the verdict to a '
            'commit, so an unattributed record is worse than none')
    sha = r.stdout.strip()
    blob = subprocess.run(['git', 'show', f'{sha}:{rel}'], capture_output=True, cwd=repo)
    if blob.returncode != 0:
        die(f'record: {sha}:{rel} is not readable -- the judging script is not committed '
            f'at HEAD, so the record would cite a commit that lacks it')
    blob_sha = hashlib.sha256(blob.stdout).hexdigest()
    if blob_sha != disk:
        die(f'record: working-tree {rel} sha256 {disk[:12]} != HEAD blob {blob_sha[:12]} -- '
            f'commit the judging script before recording a verdict')

    lv = np.array([c['level'] for c in curves.values()])
    rh = np.array([c['rho'] for c in curves.values()])
    ag = np.array([c['argmin_gap'] for c in curves.values()])
    rec = dict(
        batch=batch, verdict='pass', cells=N_CELLS,
        gate1_path=rel, git_sha=sha, gate1_sha256=disk,
        gates=dict(protocol_fields='identical', epoch_tol=EPOCH_TOL,
                   level_tol=tol, rank_min=RANK_MIN,
                   level_statistic='mean of the selection metric over common epochs'),
        level=dict(mean=float(lv.mean()), min=float(lv.min()), max=float(lv.max())),
        rank=dict(mean=float(rh.mean()), min=float(rh.min())),
        argmin_gap=dict(mean=float(ag.mean()), sd=float(ag.std(ddof=1)),
                        max_abs=float(np.abs(ag).max()), role='diagnostic, not a gate'),
        fragile_cells=sorted(f'{m}/{s}/s{d}' for m, s, d in fragile),
        segment_a=str(a_seg), frozen=str(frozen),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(f'wrote {path}  (verdict=pass, git_sha={sha[:8]}, gate1_sha256={disk[:12]})')


def reconcile(a_seg: Path, frozen: Path, tol: float, no_meta: bool = False,
              record: Path | None = None, batch: str = 'P2/W1/segment-A') -> int:
    print('RECONCILE  segment A (V0) vs frozen d7bd9c17  -- VALIDATION ONLY, TEST stays sealed')
    print(f'gates: protocol identical | stopping epoch within {EPOCH_TOL} | curve level '
          f'within +/-{tol:.2f} | per-engine rank >= {RANK_MIN}')
    print('delta := segment A - frozen. The selected-checkpoint gap is a DIAGNOSTIC, '
          'not a gate (argmin of a curve with sd ~1 is unstable)\n')

    got_all = a_segment_engine_rmse(Path(a_seg))
    bad, fragile, curves = check_cells(Path(a_seg), Path(frozen), no_meta)
    if bad:
        # An abort record keeps "ran and failed" distinguishable from "not run".
        # It can never open segment B: the ledger gates on verdict == 'pass'.
        if record is not None:
            write_abort_record(Path(record), batch, bad)
        return EXIT_ABORT

    print(f'\n{"cell":22} {"lvl":>7} {"rho":>6} {"argmin":>7} {"sel":>7} {"eng":>4}  note')
    for m in METHODS:
        for s in SUBSETS:
            for seed in SEEDS:
                key = f'{m}/{s}/s{seed}'
                k = (m, s, seed)
                if k not in got_all:
                    die(f'reconcile: segment A has no cell {k}')
                c = curves.get(key)
                note = 'fragile' if k in fragile else ''
                if c is None:
                    print(f'{m:11} {s} s{seed}  {"-":>7} {"-":>6} {"-":>7} {"-":>7} '
                          f'{len(got_all[k]):4d}  {note}')
                else:
                    print(f'{m:11} {s} s{seed}  {c["level"]:+7.4f} {c["rho"]:6.3f} '
                          f'{c["argmin_gap"]:+7.4f} {c["selected_delta"]:+7.4f} '
                          f'{len(got_all[k]):4d}  {note}')

    if curves:
        lv = np.array([c['level'] for c in curves.values()])
        rh = np.array([c['rho'] for c in curves.values()])
        ag = np.array([c['argmin_gap'] for c in curves.values()])
        print(f'\ncurve level   mean {lv.mean():+.4f}  min {lv.min():+.4f}  max {lv.max():+.4f}')
        print(f'rank (rho)    mean {rh.mean():.4f}   min {rh.min():.4f}')
        print(f'argmin gap    mean {ag.mean():+.4f}  sd {ag.std(ddof=1):.4f}  '
              f'max|.| {np.abs(ag).max():.4f}   (diagnostic, not a gate)')
    if record is not None:
        if not curves:
            die('record: a --no-meta reconcile cannot certify provenance -- no curve or '
                'rank gate ran, so "pass" would only mean the RMSE table loaded')
        write_record(Path(record), Path(a_seg), Path(frozen), curves, fragile, tol, batch)
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
                    help='skip the protocol/epoch/curve/rank gates (RMSE-only reconcile)')
    ap.add_argument('--record', default=None, metavar='PATH',
                    help='write the pass record the run ledger reads (requires a committed '
                         'gate1.py and the full gate set)')
    ap.add_argument('--batch', default='P2/W1/segment-A')
    a = ap.parse_args()

    if a.reconcile_a or a.reconcile_frozen:
        if not (a.reconcile_a and a.reconcile_frozen):
            die('reconcile needs BOTH --reconcile-a and --reconcile-frozen')
        if a.baseline or a.module:
            die('reconcile mode does not take --baseline/--module')
        if Path(a.reconcile_a).is_file() and not a.no_meta:
            die('a JSON of rows carries no history/epochs; pass --no-meta to reconcile it, '
                'or point --reconcile-a at the run directory instead')
        return reconcile(Path(a.reconcile_a), Path(a.reconcile_frozen), a.tol, a.no_meta,
                         a.record, a.batch)
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
