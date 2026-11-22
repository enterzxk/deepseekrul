"""Gate 1: is CADF plug-and-play across the five reproduced SOTA backbones?

=============================================================================
SIGN CONVENTION -- read this first.
=============================================================================
    delta := baseline - module            (POSITIVE = the module is better)
    not degraded   :  delta_EOL_MAE >= -0.05     (module - baseline <= 0.05)
    positive gain  :  delta_RMSE    >  0   AND  paired win rate > 0.50

The pre-registration states "not degraded" as delta <= +0.05 (module - baseline)
and "positive gain" as delta > 0 (baseline - module). Taken literally the same
delta would have to be both signs, so this script pins ONE convention and
translates the other. Do not re-derive the guardrail from the "+0.05" in the
plan without translating it first.

=============================================================================
Exit codes -- three distinct meanings, never reuse one for another
=============================================================================
    0  PASS     gate passed, or the reconcile is clean
    1  FAIL     the gate was measured and the criterion was not met
    2  ABORT    a precondition is unmet: bad input, or a reconcile gate fired.
                No module result exists -- do not report a FAIL.

=============================================================================
Input contract (--baseline / --module: a JSON list of rows)
=============================================================================
    {"method","subset","seed","engine","cut","y","pred"}
`cut` is a FRACTION of that engine's own life (0..1]. Per-engine, per-cut rows
are required because EOL(0.95) is defined at a cut and the win rate is per
engine; a pre-aggregated file can produce neither.

=============================================================================
Reconcile mode (--reconcile-a DIR --reconcile-frozen DIR --record PATH)
=============================================================================
Segment A retrains V0 on lab3090. V0 was trained on a40, so this measures the
drift between the two, on VALIDATION data only -- official TEST stays sealed.

THE SELECTED CHECKPOINT IS NOT THE CRITERION. Measured: the selection metric's
per-epoch sd is ~0.94 and the run-to-run argmin differs by ~0.40. On a curve
with sd ~1 the argmin wanders by several epochs, so "|argmin delta| <= 0.2"
fails runs whose level is fine, and epochs_run differs by >2 in 20 of 30 cells.
Those are properties of argmin, not of the environment. Gates, per prereg v5.1:

 1. consistency   validation_metrics.json == history.json at the selected epoch;
                  validation_predictions.csv == engine_squared_errors (both sides)
 2. protocol      config.json fields identical
 3. curve level   |mean(selection metric over COMMON epochs) - anchor| <= band,
                  where band comes from RECON_BANDS.csv per (model, subset, seed).
                  A missing band is an INPUT ERROR (exit 2, no record), not an
                  ABORT -- an absent threshold must never read as "passed".
 4. aggregate     mean(|curve level delta|) <= 0.2 AND mean(|argmin delta|) <= 0.2
 5. argmin gross  |argmin delta| > 0.40 is a gross error and aborts
 6. rank          Spearman of per-engine curve means >= 0.90: the same engines
                  must still be the hard ones
 7. runtime       R2.1(5): segment A's training_metrics.json must attest the TF32
                  flags and min_delta. The committed anchor carries that file with
                  runtime_flags {matmul_tf32, cudnn_tf32, ...}, so the TF32 half is
                  a comparison against the anchor; min_delta has no anchor field and
                  is checked against the config. What it cannot prove: that the
                  training loop USED min_delta (pinned by the runner source hash,
                  reviewed once, not per cell).

Diagnostics, NOT gates: argmin delta per cell, epochs_run/best_epoch drift, and
the tail-window level (last 25% of the common epochs). The tail is reported
because the full-window mean is dominated by the early descent for models whose
curve starts high (two_phase: 78 -> 24), so a fast early descent moves the
full-window mean by several units while the converged level is unchanged.

The record stores BOTH verdicts the prereg asks for: `band` (the gate) and
`strict_0.2` (every cell within +/-0.2, reported for the original guardrail).
`provenance_verified` is checked on every written record, abort included: the
record names a commit, and the ledger re-hashes `<commit>:paper_c/gate1.py`.

`--exclude` (a JSON list of "model/subset/sNNN") removes cells from the DRIFT
statistics only -- both the per-engine drift reading AND R2.2's level drift,
the one that can stop segment B. A cell that is known to be un-reproducible on
this machine, or whose early stop never fired, is not evidence about the
environment, and letting it into R2.2 turns "this cell cannot be reproduced
here" into "the machine drifted". band and strict_0.2 always use all 30 cells,
and the record lists every excluded cell plus the unfiltered level drift. An
exclusion that matches no cell is refused (exit 2, no record): a typo must not
silently no-op, which is exactly how v6.1's exclusion failed.

validation_predictions.csv must carry the frozen schema
`unit,y_true,y_pred,signed_error,cut` (prereg §5) and history.json must carry
`engine_squared_errors` in every epoch record. The segment-A runner wrote
`unit,cut,y,pred` and epoch aggregates only; the judge refuses both and names
the missing piece, because a judge that accepts either cannot enforce §5. The
fix belongs in the runner.
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
NOT_DEGRADED = -0.05
MIN_PASS_METHODS = 4
BOOT = 10000
BOOT_SEED = 20261122

EXIT_OK = 0
EXIT_GATE_FAIL = 1
EXIT_ABORT = 2
RECON_TOL = 0.2           # aggregate gate + the strict_0.2 reference verdict
ARG_TOL = 0.40            # per-cell argmin gross-error gate
RANK_MIN = 0.90           # per-engine rank gate
EPOCH_TOL = 2             # diagnostic only
TAIL_FRAC = 0.25          # tail window for the level diagnostic
N_CELLS = len(METHODS) * len(SUBSETS) * len(SEEDS)
BANDS_REL = 'docs/paper_c/registry/RECON_BANDS.csv'

META_FIELDS = ('epochs', 'patience', 'min_delta', 'window', 'batch', 'lr', 'weight_decay',
               'val_frac', 'split_seed', 'rul_cap')

# R2.1(7) training-window count. This is the ONLY check that can catch a
# wrong data module: both prepare() variants label the training set
# 'all_cycle_left_padded' and agree on every config field, but the variant
# that drops the first window-1 windows per engine trains on 80*(window-1)
# fewer windows (-23% on FD001). The anchor records n_train_windows per cell,
# so the expected value comes from the anchor AND is cross-checked against
# this pre-registered table -- neither side is trusted alone.
# n_train_windows is read from training_metrics.json and n_target_train_windows
# from pretraining_metrics.json; never cross them. On FD002/FD004 the target
# count and the paired count are each other's values (43251 / 48691), so a
# cross-wired read is silent rather than loud.
EXPECTED_WINDOWS = {'FD001': 16762, 'FD002': 43251, 'FD003': 20486, 'FD004': 48691}
# n_validation_cuts is NOT constant across subsets (the anchor records 100 on
# FD001 and 250 on FD004), so it is compared cell-by-cell against the anchor
# rather than against a table here.


def die(msg):
    """Abort: no result exists. Never exit 1 for this -- that means FAIL."""
    print(msg, file=sys.stderr)
    raise SystemExit(EXIT_ABORT)


def _read_json(path: Path):
    """Read JSON tolerant of a UTF-8 BOM.

    The frozen artifacts carry a BOM, and Path.read_text() defaults to the
    locale encoding on Windows (GBK here), which dies on it. One helper, used
    everywhere, so no reader can drift back to the default.
    """
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ the gate

def load(path: Path):
    rows = _read_json(path)
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
        # The frozen Paper-B CSVs store `cut` as an ABSOLUTE CYCLE. Feeding that
        # here would make "nearest to 0.95" pick the SMALLEST cycle -- the EOL
        # guardrail would be read at the earliest truncation and the gate would
        # still print a verdict.
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
        # truncation of an engine is the one with the smallest y.
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


def load_bands(path: Path):
    """{(model, subset, seed): {band, curve_mean, epochs_run}} from RECON_BANDS.csv."""
    if not path.exists():
        die(f'reconcile: missing {path} -- the per-cell bands are part of the '
            f'pre-registration; without them the tiered gate cannot run')
    bands, raw = {}, {}
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        need = ('model', 'subset', 'seed', 'band')
        miss = [c for c in need if c not in (rd.fieldnames or [])]
        if miss:
            die(f'{path}: missing columns {miss} (have {rd.fieldnames})')
        for i, r in enumerate(rd):
            try:
                k = (r['model'], r['subset'], int(r['seed']))
                bands[k] = dict(band=float(r['band']),
                                curve_mean=float(r['curve_mean_1L'])
                                if r.get('curve_mean_1L') else None,
                                epochs_run=int(r['epochs_run']) if r.get('epochs_run') else None)
            except (TypeError, ValueError) as exc:
                die(f'{path}: row {i} unparsable ({exc})')
            raw[k] = r
    missing = [(m, s, d) for m in METHODS for s in SUBSETS for d in SEEDS if (m, s, d) not in bands]
    if missing:
        die(f'{path}: no band for {len(missing)} cell(s): {missing[:5]} -- a missing '
            f'threshold is an INPUT ERROR, not a pass')
    return bands


def engine_rmse_from_csv(path: Path):
    """{unit: RMSE over that engine's cuts}, from validation_predictions.csv.

    The frozen schema is the contract (prereg §5): `unit,y_true,y_pred,...`. The
    segment-A runner wrote `unit,cut,y,pred`, so the error names that schema and
    says where the fix belongs -- a judge that accepted both could not enforce
    the contract, and "the runner records it but never executes it" is the exact
    failure mode that produced the min_delta bug.
    """
    if not path.exists():
        die(f'reconcile: missing {path}')
    per = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames:
            die(f'{path}: empty file')
        miss = [c for c in ('unit', 'y_true', 'y_pred') if c not in rd.fieldnames]
        if miss:
            hint = ''
            if {'y', 'pred'} <= set(rd.fieldnames):
                hint = (' -- this is the segment-A runner schema (unit,cut,y,pred); prereg §5 '
                        'requires the frozen schema unit,y_true,y_pred,signed_error,cut, so the '
                        'fix belongs in the runner, not in the judge')
            die(f'{path}: missing columns {miss} (have {rd.fieldnames}){hint}')
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
    """Epoch anchor and hyperparameters, with the source actually used reported."""
    tm = cell_dir / 'training_metrics.json'
    meta = _read_json(tm) if tm.exists() else {}
    src = []
    epochs_run = meta.get('epochs_run')
    if epochs_run is not None:
        src.append('training_metrics')
    else:
        hist = cell_dir / 'history.json'
        if hist.exists():
            h = _read_json(hist)
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
            best = _read_json(fr).get('best_epoch')
            if best is not None:
                src.append('FREEZE')
    if epochs_run is None or best is None:
        die(f'{label}: cannot read epochs_run/best_epoch from training_metrics.json, '
            f'history.json or FREEZE.json')
    cf = cell_dir / 'config.json'
    cfg = _read_json(cf) if cf.exists() else {}
    if 'epochs' not in cfg:
        die(f'{label}: {cf} missing "epochs" -- without the ceiling a budget-truncated '
            f'cell cannot be told from a converged one')
    return dict(epochs_run=int(epochs_run), best_epoch=int(best), cfg=cfg,
                src='+'.join(src) or '?')


def cell_curves(cell_dir: Path, label: str):
    """{epoch: (val_engine_mean_rmse, {unit: rmse})} from history.json."""
    hp = cell_dir / 'history.json'
    if not hp.exists():
        die(f'{label}: missing {hp} -- the curve level is the environment signal')
    hist = _read_json(hp)
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


def _replay_best(hist, min_delta):
    """Replay the acceptance rule: accepted iff score < best - min_delta.

    With min_delta = 0 this degenerates to the plain argmin (first minimum wins),
    which is the assertion the reconcile has always relied on. With min_delta > 0
    it is the only form that both (a) admits a legitimate last-accepted epoch that
    is not the curve minimum and (b) still rejects a genuinely wrong epoch: a
    relaxed ">= curve minimum" bound would let a wrong epoch through.
    """
    order = sorted(hist)
    best_e, best_v = order[0], hist[order[0]][0]
    for e in order[1:]:
        v = hist[e][0]
        if v < best_v - min_delta:
            best_e, best_v = e, v
    return best_e, best_v


def check_consistency(cell_dir: Path, label: str, hist, best_epoch=None):
    """metrics.json must equal history at the selected epoch, and the CSV must
    equal engine_squared_errors. A predictions file that does not correspond to
    the reported selection is how a stale artifact passes for a good run."""
    mj = cell_dir / 'validation_metrics.json'
    if not mj.exists():
        die(f'{label}: missing {mj}')
    meta = _read_json(mj)
    for k in ('val_engine_mean_rmse', 'engine_squared_errors'):
        if k not in meta:
            die(f'{mj}: missing {k}')
    claimed = float(meta['val_engine_mean_rmse'])
    if not any(abs(hist[e][0] - claimed) < 1e-6 for e in hist):
        die(f'{label}: validation_metrics.val_engine_mean_rmse={claimed:.6f} matches no '
            f'epoch in history.json -- the reported selection and the recorded curve '
            f'come from different runs')
    # The reconcile takes the anchor's argmin from the curve, so the recorded
    # selection must be reproducible from the curve under the acceptance rule the
    # config declares. A relaxed "recorded value >= curve minimum" bound would
    # lose the ability to catch a wrong epoch, which is the reason this check
    # exists; replaying the rule keeps both properties.
    cf = cell_dir / 'config.json'
    md = float((_read_json(cf).get('min_delta') or 0.0)) if cf.exists() else 0.0
    r_epoch, r_val = _replay_best(hist, md)
    if abs(r_val - claimed) > 1e-9:
        die(f'{label}: recorded selection {claimed:.6f} != min_delta replay {r_val:.6f} '
            f'(epoch {r_epoch}, min_delta={md:g}) -- the curve and the recorded selection '
            f'do not come from the same acceptance rule')
    if best_epoch is not None and int(best_epoch) != r_epoch:
        die(f'{label}: recorded best_epoch {best_epoch} != min_delta replay {r_epoch} '
            f'(min_delta={md:g}) -- the selection rule that produced this cell is not the '
            f'one its config declares')
    ref = {int(x['unit']): float(np.sqrt(float(x['sum_squared_error']) / float(x['n'])))
           for x in meta['engine_squared_errors']}
    per = engine_rmse_from_csv(cell_dir / 'validation_predictions.csv')
    if set(per) != set(ref):
        die(f'{label}: engine sets differ (csv={len(per)} metrics={len(ref)})')
    worst = max(abs(per[u] - ref[u]) for u in per)
    # The CSV stores y_pred at printed precision, so an engine RMSE recomputed
    # from it differs from the full-precision metric by ~1e-6 (measured 1.7e-6 on
    # v5 aidgn/FD001/s101). This check exists to catch a stale or truncated file,
    # which differs by orders of magnitude (the tampered fixture is +0.5), so the
    # tolerance is 1e-4: far below any real staleness, above printing rounding.
    if worst > 1e-4:
        die(f'{label}: predictions disagree with engine_squared_errors '
            f'(max |diff| = {worst:.3e}) -- stale or truncated CSV')
    mine = float(np.mean(list(per.values())))
    if abs(mine - claimed) > 1e-4:
        die(f'{label}: val_engine_mean_rmse={claimed:.6f} but recomputed {mine:.6f}')
    return per, ref


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_cells(rows, bands, exclude_agg=()):
    """The decision function, kept pure so it can be tested without any files.

    `exclude_agg` drops cells from the AGGREGATE readings only (budget-truncated
    and shape-violating cells): their deltas are structurally different, not
    drift. The per-cell readings still cover every row, and the caller records
    exactly which cells were dropped.
    """
    viol, strict = [], []
    for r in rows:
        k = (r['model'], r['subset'], r['seed'])
        tag = f"{r['model']}/{r['subset']}/s{r['seed']}"
        band = bands[k]['band']
        if abs(r['dlev']) > band:
            viol.append(f"{tag}: curve level {r['dlev']:+.4f} outside band {band:.4f}")
        if abs(r['dlev']) > RECON_TOL:
            strict.append(f"{tag}: curve level {r['dlev']:+.4f} outside {RECON_TOL}")
        if abs(r['dargmin']) > ARG_TOL:
            viol.append(f"{tag}: argmin gap {r['dargmin']:+.4f} outside {ARG_TOL}")
        if not (r['rho'] >= RANK_MIN):
            viol.append(f"{tag}: per-engine rank {r['rho']:.3f} < {RANK_MIN}")
    drop = {tuple(x) for x in exclude_agg}
    keep = [r for r in rows if (r['model'], r['subset'], r['seed']) not in drop]
    ml = float(np.mean([abs(r['dlev']) for r in keep])) if keep else float('nan')
    ma = float(np.mean([abs(r['dargmin']) for r in keep])) if keep else float('nan')
    if keep and ml > RECON_TOL:
        viol.append(f'aggregate: mean|dlev| {ml:.4f} > {RECON_TOL} '
                    f'(over {len(keep)} of {len(rows)} cells)')
    if keep and ma > RECON_TOL:
        viol.append(f'aggregate: mean|dargmin| {ma:.4f} > {RECON_TOL} '
                    f'(over {len(keep)} of {len(rows)} cells)')
    return dict(violations=viol, strict_violations=strict,
                mean_abs_dlev=ml, mean_abs_dargmin=ma, agg_n=len(keep),
                agg_excluded=sorted(f'{m}/{s}/s{d}' for m, s, d in drop),
                pass_band=not viol, pass_strict=not strict)


def check_cells(a_root: Path, f_root: Path, bands, no_meta: bool):
    """Read the 30 cells and build the per-cell rows the gates and record need."""
    if no_meta:
        print('cell checks: SKIPPED (--no-meta) -- RMSE-only, no gates\n')
        return [], [], set(), {}, [], set(), [], [], []
    rows, proto, fragile, curves = [], [], set(), {}
    shape_bad = set()
    ceil_stats = {}
    eng = []
    runtime = []
    pretraining = []
    windows = []
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
                            proto.append(f'{key}: config.{f} {am["cfg"][f]!r} != {fm["cfg"][f]!r}')

                # R2.1(2) data fingerprint: the same engines must have been read
                fa = fm['cfg'].get('training_data_sha256')
                aa = am['cfg'].get('training_data_sha256')
                if fa is None or aa is None:
                    proto.append(f'{key}: training_data_sha256 present on '
                                 f'{"neither side" if fa is None and aa is None else "one side only"} '
                                 f'-- which engines were read cannot be verified')
                elif {k: str(v) for k, v in fa.items()} != {k: str(v) for k, v in aa.items()}:
                    proto.append(f'{key}: training_data_sha256 differs '
                                 f'(anchor {sorted(fa)} vs segment A {sorted(aa)})')

                # R2.1(5) runtime switches. The committed anchor DOES carry
                # training_metrics.json -- the 7-file copy in the working tree is a
                # pruned checkout -- and that file records
                # runtime_flags {matmul_tf32, cudnn_tf32, cudnn_deterministic,
                # cudnn_benchmark}. So the TF32 half is a COMPARISON against the
                # anchor, not a self-attestation; only min_delta has no anchor field
                # and is checked against the config. What none of this proves: that
                # the training loop USED min_delta -- that is pinned by the runner
                # source hash and reviewed once, not per cell.
                tm_path = ad / 'training_metrics.json'
                anchor_tm = fd / 'training_metrics.json'
                a_tm = _read_json(anchor_tm) if anchor_tm.exists() else {}
                a_flags = a_tm.get('runtime_flags') or {}
                tm = _read_json(tm_path) if tm_path.exists() else None
                if tm is None:
                    runtime.append(dict(model=m, subset=s, seed=seed, attested=False,
                                        why='no training_metrics.json',
                                        anchor_runtime_flags=a_flags or None))
                    proto.append(f'{key}: R2.1(5) segment A has no training_metrics.json -- '
                                 f'min_delta/tf32_matmul/tf32_cudnn cannot be attested')
                else:
                    md_local = tm.get('min_delta')
                    md_cfg = am['cfg'].get('min_delta')
                    bad = []
                    if md_cfg is not None and not _same(md_local, md_cfg):
                        bad.append(f'min_delta {md_local!r} != config {md_cfg!r}')
                    if md_cfg is None and md_local is None:
                        bad.append('min_delta absent on both sides')
                    for lk, ak in (('tf32_matmul', 'matmul_tf32'),
                                   ('tf32_cudnn', 'cudnn_tf32')):
                        lv = tm.get(lk)
                        if ak in a_flags:
                            if lv is None or bool(lv) is not bool(a_flags[ak]):
                                bad.append(f'{lk}={lv!r} but anchor '
                                           f'runtime_flags.{ak}={a_flags[ak]!r}')
                        elif lv is not False:
                            bad.append(f'{lk}={lv!r} != False (anchor recorded no runtime_flags)')
                    runtime.append(dict(model=m, subset=s, seed=seed, attested=not bad,
                                        min_delta=md_local,
                                        tf32_matmul=tm.get('tf32_matmul'),
                                        tf32_cudnn=tm.get('tf32_cudnn'),
                                        anchor_runtime_flags=a_flags or None,
                                        mode='comparison' if a_flags else 'self_attestation'))
                    for b in bad:
                        proto.append(f'{key}: R2.1(5) training_metrics {b}')

                # R2.1(7) training-window count (prereg v6.5.6). Which windows
                # were trained on is not in config -- the two data modules are
                # field-for-field identical there -- so it is read from
                # training_metrics.json, where the anchor records it per cell.
                exp_n = EXPECTED_WINDOWS.get(s)
                an = a_tm.get('n_train_windows')
                if exp_n is None:
                    proto.append(f'{key}: R2.1(7) no pre-registered window count for {s}')
                elif an is None:
                    proto.append(f'{key}: R2.1(7) anchor training_metrics.json has no '
                                 f'n_train_windows -- which windows were trained on is '
                                 f'unverifiable')
                elif not _same(an, exp_n):
                    proto.append(f'{key}: R2.1(7) anchor n_train_windows {an!r} != '
                                 f'pre-registered {exp_n!r} for {s} (anchor and table '
                                 f'disagree -- one of them is wrong)')
                if tm is not None:
                    ln = tm.get('n_train_windows')
                    lv = tm.get('n_validation_cuts')
                    if ln is None:
                        proto.append(f'{key}: R2.1(7) segment A training_metrics.json has '
                                     f'no n_train_windows -- which windows were trained on '
                                     f'is unverifiable (the wrong data module wrote the same '
                                     f'9 fields, so absence is not neutral)')
                    elif exp_n is not None and not _same(ln, exp_n):
                        try:
                            d = f'{int(ln) - int(exp_n):+d} windows'
                        except (TypeError, ValueError):
                            d = 'delta unknown'
                        proto.append(f'{key}: R2.1(7) n_train_windows {ln!r} != '
                                     f'pre-registered {exp_n!r} for {s} ({d}) -- the '
                                     f'training set was built by a different data module')
                    av = a_tm.get('n_validation_cuts')
                    if lv is None:
                        proto.append(f'{key}: R2.1(7) segment A training_metrics.json has '
                                     f'no n_validation_cuts')
                    elif av is None:
                        proto.append(f'{key}: R2.1(7) anchor training_metrics.json has no '
                                     f'n_validation_cuts -- the cut count cannot be '
                                     f'compared')
                    elif not _same(lv, av):
                        proto.append(f'{key}: R2.1(7) n_validation_cuts {lv!r} != '
                                     f'anchor {av!r}')
                    windows.append(dict(model=m, subset=s, seed=seed,
                                        n_train_windows=ln, n_validation_cuts=lv,
                                        anchor_n_train_windows=an,
                                        anchor_n_validation_cuts=av,
                                        expected=exp_n))

                # R2.1(6) two_phase pretraining. The backbone's protocol is
                # "pretrain on the paired subset, then freeze the encoder", and
                # until v6.5.5 the judge never looked at it. Both sides write
                # pretraining_metrics.json, so the protocol fields are COMPARED
                # rather than hardcoded.
                if m == 'two_phase':
                    pa = ad / 'pretraining_metrics.json'
                    pf = fd / 'pretraining_metrics.json'
                    if not pf.exists():
                        proto.append(f'{key}: R2.1(6) anchor has no pretraining_metrics.json '
                                     f'-- paired pretraining cannot be checked')
                    elif not pa.exists():
                        proto.append(f'{key}: R2.1(6) segment A has no pretraining_metrics.json '
                                     f'-- paired pretraining cannot be checked')
                    else:
                        ja, jf = _read_json(pa), _read_json(pf)
                        for fld in ('epochs', 'n_paired_train_windows',
                                    'n_target_train_windows'):
                            if fld not in ja or fld not in jf:
                                proto.append(f'{key}: R2.1(6) pretraining_metrics.{fld} missing '
                                             f'(A={fld in ja}, anchor={fld in jf})')
                            elif not _same(ja[fld], jf[fld]):
                                proto.append(f'{key}: R2.1(6) pretraining {fld} '
                                             f'{ja[fld]!r} != anchor {jf[fld]!r}')
                        pe = am['cfg'].get('pretraining_epochs')
                        if pe is not None and not _same(ja.get('epochs'), pe):
                            proto.append(f'{key}: R2.1(6) pretraining epochs '
                                         f'{ja.get("epochs")!r} != config.pretraining_epochs '
                                         f'{pe!r}')
                        # The target set IS the fine-tuning set, so these two
                        # counts must agree; if they do not, one of them was
                        # read from the wrong file (see EXPECTED_WINDOWS).
                        nt = ja.get('n_target_train_windows')
                        ntr = tm.get('n_train_windows') if tm else None
                        if nt is not None and ntr is not None and not _same(nt, ntr):
                            proto.append(f'{key}: R2.1(6) pretraining '
                                         f'n_target_train_windows {nt!r} != training '
                                         f'n_train_windows {ntr!r} -- one of the two counts '
                                         f'came from the wrong file')
                        pretraining.append(dict(
                            model=m, subset=s, seed=seed, epochs=ja.get('epochs'),
                            n_paired=ja.get('n_paired_train_windows'),
                            n_target=ja.get('n_target_train_windows'),
                            encoder_frozen=ja.get('encoder_before_finetune_sha256')
                            == ja.get('encoder_after_finetune_sha256')))

                # R2.1(3) early-stop shape. Per prereg v6.5.5 this is a GROUP
                # check on (backbone, subset) x 3 seeds -- "all three local cells
                # hit the ceiling while the anchor's did not" -- not a per-cell
                # epoch tolerance. Per-cell epochs_run/best_epoch differences are
                # drift readings (R2.2) and never enter `proto`; they only mark the
                # cell shape_bad for the aggregate reading.
                try:
                    f_ceil = fm['epochs_run'] >= int(fm['cfg']['epochs'])
                    a_ceil = am['epochs_run'] >= int(am['cfg']['epochs'])
                except (TypeError, ValueError):
                    f_ceil = a_ceil = None
                if f_ceil is not None:
                    st = ceil_stats.setdefault((m, s), dict(n=0, a=0, f=0, cells=[]))
                    st['n'] += 1
                    st['a'] += int(bool(a_ceil))
                    st['f'] += int(bool(f_ceil))
                    st['cells'].append((seed, fm['epochs_run'], am['epochs_run']))
                    if bool(f_ceil) != bool(a_ceil):
                        shape_bad.add((m, s, seed))

                ceiling = fm['cfg'].get('epochs')
                if isinstance(ceiling, (int, float)) and fm['epochs_run'] >= float(ceiling):
                    fragile.add((m, s, seed))

                fc, ac = cell_curves(fd, f'frozen {key}'), cell_curves(ad, f'segment A {key}')
                check_consistency(fd, f'frozen {key}', fc, fm['best_epoch'])
                per_a, _ = check_consistency(ad, f'segment A {key}', ac, am['best_epoch'])

                # R2.1(4) numerical sanity
                for side, cv, mm in (('anchor', fc, fm), ('segment A', ac, am)):
                    if len(cv) != mm['epochs_run']:
                        proto.append(f'{key}: {side} history has {len(cv)} epochs but '
                                     f'epochs_run={mm["epochs_run"]}')
                    if not (1 <= mm['best_epoch'] <= mm['epochs_run']):
                        proto.append(f'{key}: {side} best_epoch {mm["best_epoch"]} outside '
                                     f'[1,{mm["epochs_run"]}]')
                    if any(not np.isfinite(v[0]) for v in cv.values()):
                        proto.append(f'{key}: {side} curve contains non-finite values')

                common = sorted(set(fc) & set(ac))
                if len(common) < 5:
                    die(f'{key}: only {len(common)} common epochs -- cannot compare curves')
                units = sorted(fc[common[0]][1])
                if units != sorted(ac[common[0]][1]):
                    die(f'{key}: engine sets differ between curves')
                lev_f = float(np.mean([fc[e][0] for e in common]))
                lev_a = float(np.mean([ac[e][0] for e in common]))
                tail_n = max(1, int(round(len(common) * TAIL_FRAC)))
                tail_f = float(np.mean([fc[e][0] for e in common[-tail_n:]]))
                tail_a = float(np.mean([ac[e][0] for e in common[-tail_n:]]))
                cf_ = np.array([float(np.mean([fc[e][1][u] for e in common])) for u in units])
                ca_ = np.array([float(np.mean([ac[e][1][u] for e in common])) for u in units])
                rho = _spearman(cf_, ca_)
                for i, u in enumerate(units):
                    eng.append((key, int(u), float(ca_[i] - cf_[i])))
                arg_f = float(min(fc[e][0] for e in fc))
                arg_a = float(min(ac[e][0] for e in ac))
                d_sel = float(np.mean([per_a[u] for u in sorted(per_a)])) - arg_f
                band = bands[(m, s, seed)]['band']
                curves[key] = dict(level=lev_a - lev_f, rho=rho, n_common=len(common),
                                   argmin_gap=arg_a - arg_f, selected_delta=d_sel)
                rows.append(dict(
                    model=m, subset=s, seed=seed,
                    epochs_run=am['epochs_run'], best_epoch=am['best_epoch'],
                    epochs_run_anchor=fm['epochs_run'], best_epoch_anchor=fm['best_epoch'],
                    curve_mean=lev_a, curve_mean_anchor=lev_f,
                    curve_tail_delta=tail_a - tail_f,
                    band=band, dlev=lev_a - lev_f,
                    argmin=arg_a, argmin_anchor=arg_f, dargmin=arg_a - arg_f,
                    selected_delta=d_sel, rho=rho, n_common=len(common),
                    within_band=bool(abs(lev_a - lev_f) <= band),
                    within_strict=bool(abs(lev_a - lev_f) <= RECON_TOL),
                    within_argmin=bool(abs(arg_a - arg_f) <= ARG_TOL)))

    print(f'protocol: {n_cmp} config-field comparisons, {len(proto)} mismatch(es)')
    print(f'epoch sources (frozen|segment A): {dict(srcs)}')
    print(f'budget-truncated (fragile) cells: {len(fragile)} '
          f'{sorted(f"{m}/{s}/s{d}" for m, s, d in fragile)}')
    print('early-stop shape, per (backbone, subset) -- R2.1(3) fires only when ALL '
          'local cells hit the ceiling and the anchor did not:')
    for (m, s), st in sorted(ceil_stats.items()):
        print(f'  {m:11} {s}  local {st["f"]}/{st["n"]} ceiling   '
              f'anchor {st["a"]}/{st["n"]}')
        if st['f'] == st['n'] and st['a'] < st['n']:
            proto.append(f'{m}/{s}: R2.1(3) all {st["n"]} segment-A cells hit the epoch '
                         f'ceiling while the anchor had {st["a"]}/{st["n"]} -- early stop '
                         f'never fired')
    print(f'shape-divergent cells (diagnostic, not protocol): {len(shape_bad)} '
          f'{sorted(f"{m}/{s}/s{d}" for m, s, d in shape_bad)}')
    return rows, proto, fragile, curves, eng, shape_bad, runtime, pretraining, windows


def level_drift_summary(rows, seed=BOOT_SEED, exclude=()):
    """R2.2: the only drift reading that can stop segment B.

    A systematic machine shift moves every cell the same way, so the mean of the
    signed per-cell level deltas has a CI away from zero AND the signs agree. If
    the CI covers zero the offset is training noise, which is the case the
    pre-registration says to continue through.

    `exclude` holds canonical "model/subset/sNNN" ids. Filtering happens HERE,
    not only in the per-engine reading: a cell that is known un-reproducible on
    this machine must not be able to stop segment B. The caller records both the
    filtered and the unfiltered numbers, so the removal is never silent.
    """
    keep = [r for r in rows
            if f"{r['model']}/{r['subset']}/s{r['seed']}" not in set(exclude)]
    if not keep:
        die('R2.2: every cell was excluded -- no drift statistic exists. '
            'Refusing to report a verdict computed from zero cells.')
    d = np.array([r['dlev'] for r in keep], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    frac = max(pos, neg) / (pos + neg) if pos + neg else float('nan')
    return dict(n=int(d.size), n_cells=len(keep), mean=float(d.mean()), ci=[lo, hi],
                same_sign_frac=float(frac),
                excluded_cells=sorted(f"{r['model']}/{r['subset']}/s{r['seed']}"
                                      for r in rows if r not in keep),
                stops=bool((lo > 0 or hi < 0) and frac >= 0.80))



def level_drift_by_backbone(rows, exclude=()):
    """The R2.2 statistic computed per backbone. Descriptive, never a gate.

    The pooled rule fires only on a machine-WIDE shift: one backbone moving by
    several RMSE cannot reach 80% same-sign across 30 cells, because the healthy
    backbones' signs are close to a coin flip. Measured on segment A: pooled
    +1.0109 at 70% same-sign (continue) while two_phase alone is +4.8305 and the
    other four are +0.0560. Reporting the same statistic per backbone keeps a
    one-backbone offset from reading as "the machine drifted" -- it changes no
    threshold, no verdict and no exit code.
    """
    drop = set(exclude)
    out = {}
    for m in METHODS:
        sub = [r for r in rows if r['model'] == m
               and f"{r['model']}/{r['subset']}/s{r['seed']}" not in drop]
        if not sub:
            out[m] = dict(n=0)
            continue
        pb = level_drift_summary(sub)
        out[m] = dict(n=pb['n_cells'], mean=pb['mean'], ci=pb['ci'],
                      same_sign_frac=pb['same_sign_frac'], stops=pb['stops'])
    return out


def drift_summary(deltas):
    """Is the per-ENGINE offset a systematic shift, or zero-mean noise?

    A real machine/environment shift moves every engine the same way by a
    similar amount; argmin wander mixes signs. So the test is on the pooled
    per-engine signed deltas: a robust centre far from zero AND a near-unanimous
    sign. Recorded either way -- the numbers are the finding, not the label.
    """
    d = np.asarray(list(deltas), dtype=float)
    if d.size == 0:
        return dict(n=0, mean=None, median=None, same_sign_frac=None,
                    p10=None, p90=None, systematic=None)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    nz = pos + neg
    frac = max(pos, neg) / nz if nz else float('nan')
    med = float(np.median(d))
    return dict(n=int(d.size), mean=float(d.mean()), median=med,
                same_sign_frac=float(frac),
                p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)),
                systematic=bool(abs(med) > RECON_TOL and frac >= 0.90))


# ---------------------------------------------------------------- provenance

def provenance():
    """(path_in_repo, commit, sha256 of the COMMITTED blob).

    The hash is taken from `git show`, never from the working-tree bytes: a CRLF
    checkout has different bytes for the same committed file, so a byte
    comparison would refuse to record on a perfectly clean tree. Whether the
    script is committed is asked of git itself (`status --porcelain`), and the
    recorded hash is the blob the ledger will re-hash.
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
    st = subprocess.run(['git', 'status', '--porcelain', '--', rel],
                        capture_output=True, text=True, cwd=repo)
    if st.returncode != 0:
        die(f'record: cannot read git status for {rel}')
    if st.stdout.strip():
        die(f'record: {rel} has uncommitted changes ({st.stdout.strip()[:80]}) -- commit the '
            f'judging script before recording a verdict')
    blob = subprocess.run(['git', 'show', f'{sha}:{rel}'], capture_output=True, cwd=repo)
    if blob.returncode != 0:
        die(f'record: {sha}:{rel} is not readable -- the judging script is not committed '
            f'at HEAD, so the record would cite a commit that lacks it')
    blob_sha = hashlib.sha256(blob.stdout).hexdigest()
    if blob_sha != disk:
        print(f'note: working-tree bytes differ from the blob ({disk[:12]} vs '
              f'{blob_sha[:12]}) -- line endings only; recording the blob hash',
              file=sys.stderr)
    return rel, sha, blob_sha


def _bands_provenance(path: Path):
    rel = BANDS_REL
    repo = repo_root()
    r = subprocess.run(['git', 'show', f'HEAD:{rel}'], capture_output=True, cwd=repo)
    blob = hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
    disk = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return dict(bands_path=str(path), bands_sha256_blob=blob, bands_sha256_disk=disk)


def _record_common(batch, rel, sha, blob_sha, bands_path):
    return dict(batch=batch, cells=N_CELLS, gate1_path=rel, git_sha=sha,
                gate1_sha256=blob_sha, provenance_verified=True,
                gates=dict(per_cell_curve='RECON_BANDS.csv band',
                           aggregate=f'mean|dlev| and mean|dargmin| <= {RECON_TOL}',
                           per_cell_argmin_le=ARG_TOL, rank_min=RANK_MIN,
                           epoch_tol=EPOCH_TOL, epoch_role='diagnostic',
                           argmin_role='per-cell gross error only',
                           level_statistic='mean of the selection metric over common epochs'),
                **_bands_provenance(bands_path))


def _drift_block(drift, drift_ex, excluded, level_drift, level_all, by_backbone):
    """One definition of the drift evidence, used by BOTH record paths.

    The abort path used to write only `violations`, so an abort forced by a
    protocol violation (fcstgnn's early-stop shape is the live case) threw away
    the 30-row drift evidence at exactly the moment it mattered. Same dict here,
    with a role marker on the abort side.
    """
    return dict(
        all_cells=drift, excluded_cells=drift_ex, exclusion_list=sorted(excluded or []),
        level=level_drift, level_unfiltered=level_all,
        per_backbone=by_backbone,
        per_backbone_path='drift.per_backbone -- NOT the ledger top-level per_backbone, '
                          'which is the R2.1 protocol gate {method: pass|fail}',
        level_drift_role='descriptive only: same R2.2 statistic per backbone, so a '
                         'method-specific offset cannot hide inside the pooled '
                         'same-sign rule; drives no threshold and no exit code',
        rule='R2.2 stop iff signed dlev CI excludes 0 AND same-sign >= 80%; '
             'per-engine median rule reported alongside. `level` is computed '
             'after exclusion_list; `level_unfiltered` keeps the all-cell number')


def write_abort_record(path: Path, batch, violations, bands_path, evidence=None):
    """An abort record keeps "ran and failed" distinguishable from "not run".
    Provenance is verified here too: the record names a commit either way.

    `evidence` carries the computed blocks (drift, detail, band, protocol,
    runtime) so a protocol abort does not erase what the run measured. It is
    marked `computed, not decisive`: the verdict stays abort on `violations`.
    """
    rel, sha, blob_sha = provenance()
    rec = _record_common(batch, rel, sha, blob_sha, bands_path)
    rec.update(verdict='abort', violations=list(violations))
    if evidence:
        rec['computed_evidence'] = dict(
            evidence,
            role='computed, not decisive: the verdict is abort on the violations above; '
                 'these blocks are recorded so the drift evidence is not lost when a '
                 'protocol violation forces the abort path')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(f'wrote {path}  (verdict=abort, {len(violations)} violation(s) -- segment B stays closed'
          f'{", computed evidence kept" if evidence else ""})')


def write_record(path: Path, batch, res, proto, rows, curves, fragile, bands_path,
                 a_seg, frozen, drift=None, drift_ex=None, excluded=None, level_drift=None,
                 level_all=None, per_backbone=None, runtime=None, pretraining=None,
                 windows=None):
    rel, sha, blob_sha = provenance()
    lv = np.array([c['level'] for c in curves.values()])
    rh = np.array([c['rho'] for c in curves.values()])
    ag = np.array([c['argmin_gap'] for c in curves.values()])
    tl = np.array([r['curve_tail_delta'] for r in rows])
    rec = _record_common(batch, rel, sha, blob_sha, bands_path)
    rec.update(
        verdict='pass',
        # verdict answers "is there a systematic offset?"; these two answer
        # "does it reproduce to 0.2 / within band?" and are reported in full.
        drift=_drift_block(drift, drift_ex, excluded, level_drift, level_all, per_backbone),
        band={'pass': res['pass_band'], 'violations': res['violations']},
        strict_0_2={'pass': res['pass_strict'], 'violations': res['strict_violations']},
        protocol={'pass': not proto, 'violations': proto},
        runtime_switches=dict(
            cells=runtime or [],
            anchor_has_training_metrics=any(r.get('anchor_runtime_flags')
                                            for r in (runtime or [])),
            mode='TF32 flags: comparison against the anchor runtime_flags; min_delta: '
                 'checked against the config (the anchor file has no min_delta field)',
            proves='the recorded TF32 flags equal the anchor\'s and min_delta equals the config',
            cannot_prove='that the training loop USED min_delta; that is pinned by the runner '
                         'source hash and reviewed once, not per cell'),
        pretraining=dict(
            cells=pretraining or [],
            rule='R2.1(6), two_phase only: pretraining epochs and the paired/target window '
                 'counts must equal the anchor, and epochs must equal '
                 'config.pretraining_epochs; encoder_frozen is reported, not gated'),
        window_counts=dict(
            cells=windows or [],
            expected=EXPECTED_WINDOWS,
            rule='R2.1(7): n_train_windows must equal the anchor AND the pre-registered '
                 'per-subset count; n_validation_cuts must equal the anchor\'s (it is '
                 'per-subset: 100 on FD001, 250 on FD004)',
            proves='the number of training windows was the anchor\'s, so a prepare() that '
                   'drops the first window-1 windows per engine cannot pass silently',
            cannot_prove='that they were the same windows in the same order -- count plus '
                         'the data fingerprint plus the split file cover identity, order '
                         'is pinned by the runner source hash'),
        aggregate=dict(mean_abs_dlev=res['mean_abs_dlev'],
                       mean_abs_dargmin=res['mean_abs_dargmin'],
                       n_cells=res['agg_n'],
                       excluded=res['agg_excluded'],
                       excluded_reason='budget-truncated and early-stop-shape cells are '
                                       'structurally different, not drift'),
        level=dict(mean=float(lv.mean()), min=float(lv.min()), max=float(lv.max())),
        rank=dict(mean=float(rh.mean()), min=float(rh.min())),
        argmin_gap=dict(mean=float(ag.mean()), sd=float(ag.std(ddof=1)),
                        max_abs=float(np.abs(ag).max()), role='diagnostic'),
        curve_tail=dict(mean=float(tl.mean()), min=float(tl.min()), max=float(tl.max()),
                        role='diagnostic: full-window mean is dominated by the early descent'),
        fragile_cells=sorted(f'{m}/{s}/s{d}' for m, s, d in fragile),
        detail=rows, segment_a=str(a_seg), frozen=str(frozen),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print(f'wrote {path}  (verdict={rec["verdict"]}, git_sha={sha[:8]}, '
          f'gate1_sha256={blob_sha[:12]})')


def a_segment_engine_rmse(root: Path):
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


def reconcile(a_seg: Path, frozen: Path, bands_path: Path, tol: float,
              no_meta: bool = False, record: Path | None = None,
              batch: str = 'P2/W1/segment-A', exclude=None) -> int:
    print('RECONCILE  segment A (V0) vs frozen d7bd9c17  -- VALIDATION ONLY, TEST stays sealed')
    print(f'gates: consistency | protocol | curve level <= band | aggregate <= {tol:.2f} '
          f'| argmin gross <= {ARG_TOL:.2f} | rank >= {RANK_MIN}')
    print('delta := segment A - anchor. argmin and epochs_run are DIAGNOSTICS: '
          'on a curve with sd ~1 the argmin wanders by several epochs\n')

    bands = load_bands(bands_path)
    got_all = a_segment_engine_rmse(Path(a_seg))
    rows, proto, fragile, curves, eng, shape_bad, runtime, pretraining, windows = check_cells(
        Path(a_seg), Path(frozen), bands, no_meta)

    if no_meta:
        if record is not None:
            die('record: a --no-meta reconcile cannot certify provenance -- no gate ran')
        print('RECONCILE: OK (no gates ran)')
        return EXIT_OK

    res = evaluate_cells(rows, bands, fragile | shape_bad)
    print(f'\n{"cell":22} {"dlev":>8} {"band":>6} {"tail":>8} {"darg":>8} {"rho":>6} '
          f'{"dEpo":>5}  flags')
    for r in rows:
        flags = []
        if not r['within_band']:
            flags.append('LEVEL')
        if not r['within_strict']:
            flags.append('strict')
        if not r['within_argmin']:
            flags.append('ARG')
        if not (r['rho'] >= RANK_MIN):
            flags.append('RANK')
        print(f"{r['model']:11} {r['subset']} s{r['seed']}  {r['dlev']:+8.4f} "
              f"{r['band']:6.4f} {r['curve_tail_delta']:+8.4f} {r['dargmin']:+8.4f} "
              f"{r['rho']:6.3f} {r['best_epoch'] - r['best_epoch_anchor']:+5d}  "
              f"{' '.join(flags) or 'ok'}")

    lv = np.array([r['dlev'] for r in rows])
    ag = np.array([r['dargmin'] for r in rows])
    tl = np.array([r['curve_tail_delta'] for r in rows])
    ep = np.array([r['best_epoch'] - r['best_epoch_anchor'] for r in rows])
    print(f'\ncurve level   mean {lv.mean():+.4f}  mean|.| {np.abs(lv).mean():.4f}  '
          f'max|.| {np.abs(lv).max():.4f}   (aggregate gate {tol:.2f})')
    print(f'curve tail    mean {tl.mean():+.4f}  mean|.| {np.abs(tl).mean():.4f}  '
          f'(diagnostic)')
    print(f'argmin gap    mean {ag.mean():+.4f}  mean|.| {np.abs(ag).mean():.4f}  '
          f'max|.| {np.abs(ag).max():.4f}')
    print(f'epoch drift   mean {ep.mean():+.2f}  median|.| {np.median(np.abs(ep)):.1f}  '
          f'cells beyond {EPOCH_TOL}: {(np.abs(ep) > EPOCH_TOL).sum()}/{len(ep)}  (diagnostic)')
    print(f'band gate: {"PASS" if res["pass_band"] else "FAIL"}   '
          f'strict_0.2: {"pass" if res["pass_strict"] else "fail"}   '
          f'protocol: {"ok" if not proto else "MISMATCH"}')

    # The 2026-11-23 decision demoted segment A from a stop-gate to a drift
    # measurement: "reproduce the a40 run to within 0.2" stopped being a
    # requirement once all 60 cells were retrained locally. The blocking question
    # is whether 3090 shows a SYSTEMATIC offset, so `verdict` answers that --
    # while band and strict_0.2 keep reporting the reproducibility gates in full,
    # so a wide band can never hide a known-bad cell.
    # Canonical cell ids: the engine list is keyed "model/subset/sNNN", so the
    # exclusion set must use that same string. v6.1 built tuples here and the
    # membership test could never match -- "excluding 9 declared cell(s)" printed
    # n=1050 while nothing was dropped, and R2.2 never saw the exclusion at all.
    excl = {str(x) for x in (exclude or [])}
    known = {k for k, _, _ in eng}
    unknown = sorted(excl - known)
    if unknown:
        die(f'--exclude names {len(unknown)} cell(s) that are not in this batch: '
            f'{unknown}. An exclusion that matches nothing is a silent no-op, so it is '
            f'refused; valid ids look like "aidgn/FD001/s101".')
    drift = drift_summary(d for _, _, d in eng)
    drift_ex = drift_summary(d for k, _, d in eng if k not in excl) if excl else None
    print(f'drift (per-engine, all cells): n={drift["n"]} median {drift["median"]:+.4f} '
          f'mean {drift["mean"]:+.4f} same-sign {drift["same_sign_frac"]:.1%} '
          f'p10/p90 {drift["p10"]:+.3f}/{drift["p90"]:+.3f}')
    if drift_ex is not None:
        print(f'drift (per-engine, excluding {len(excl)} declared cell(s)): '
              f'n={drift["n"]} -> n={drift_ex["n"]} '
              f'({drift["n"] - drift_ex["n"]} engine readings dropped) '
              f'median {drift_ex["median"]:+.4f} '
              f'same-sign {drift_ex["same_sign_frac"]:.1%}')
    # Reported diagnostics follow the same eligible set as R2.2, so the printed
    # "systematic offset" cannot be produced by the cells we just declared bad.
    systematic = (drift_ex or drift)['systematic']
    level_all = level_drift_summary(rows)
    level_drift = level_drift_summary(rows, exclude=excl) if excl else level_all
    rank_bad = [r for r in rows if not (r['rho'] >= RANK_MIN)]
    print(f'systematic offset (per-engine, median rule, '
          f'{"after exclusions" if excl else "all cells"}): '
          f'{"YES" if systematic else "no"}   (reported)')
    if excl:
        print(f'drift (per-cell level, R2.2, all {level_all["n_cells"]} cells, '
              f'reported only): mean {level_all["mean"]:+.4f} '
              f'CI [{level_all["ci"][0]:+.4f},{level_all["ci"][1]:+.4f}] '
              f'same-sign {level_all["same_sign_frac"]:.0%} -> '
              f'{"STOPS" if level_all["stops"] else "no stop"}')
    excl_note = f', {len(excl)} excluded' if excl else ''
    print(f'drift (per-cell level, R2.2, deciding: {level_drift["n_cells"]} cells'
          f'{excl_note}): mean {level_drift["mean"]:+.4f} '
          f'CI [{level_drift["ci"][0]:+.4f},{level_drift["ci"][1]:+.4f}] '
          f'same-sign {level_drift["same_sign_frac"]:.0%} -> '
          f'{"STOPS segment B" if level_drift["stops"] else "training noise, continue"}')

    # The pooled rule only fires on a machine-WIDE shift: a single backbone that
    # moves by several RMSE cannot reach 80% same-sign across 30 cells, because
    # the healthy backbones' signs are a coin flip. Measured on segment A: the
    # pooled mean is +1.0109 with 70% same-sign (continue) while two_phase alone
    # is +4.8305 and the other four are +0.0560. So the same statistic is
    # reported per backbone -- descriptive only: it changes no threshold, no
    # verdict and no exit code, it just stops a one-backbone offset from reading
    # as "the machine drifted".
    by_backbone = level_drift_by_backbone(rows, excl)
    print('\nper-backbone level drift (same R2.2 statistic, descriptive only):')
    print(f'  {"backbone":11} {"n":>2} {"mean":>9} {"CI":>21} {"same-sign":>10}')
    for m in METHODS:
        pb = by_backbone[m]
        if not pb.get('n'):
            print(f'  {m:11}  0   (all cells excluded)')
            continue
        print(f"  {m:11} {pb['n']:2d} {pb['mean']:+9.4f} "
              f"[{pb['ci'][0]:+9.4f},{pb['ci'][1]:+9.4f}] "
              f"{pb['same_sign_frac']:9.0%}")
    print(f'rank: {len(rank_bad)}/{len(rows)} cell(s) below {RANK_MIN} (reported, not blocking)')

    print('\nper-backbone protocol fidelity (6 cells each):')
    backbone_ok = {}
    for m in METHODS:
        bad = [p for p in proto if p.startswith(f'{m}/')]
        backbone_ok[m] = not bad
        print(f'  {m:11} {"OK" if not bad else f"FAIL ({len(bad)} violation(s))"}')

    if proto:
        print('\n'.join('  ' + p for p in proto), file=sys.stderr)
    if res['violations']:
        print('\n'.join('  ' + v for v in res['violations']), file=sys.stderr)

    verdict_ok = (not level_drift['stops']) and not proto
    blocking = []
    if proto:
        blocking.append(f'protocol fidelity R2.1 ({len(proto)})')
    if level_drift['stops']:
        blocking.append('systematic level drift R2.2')
    print(f'blocking gates: {", ".join(blocking) or "none"}   '
          f'reported only: band / strict_0.2 / aggregate / argmin-gross / rank')
    if record is not None:
        if verdict_ok:
            write_record(Path(record), batch, res, proto, rows, curves, fragile, bands_path,
                         Path(a_seg), Path(frozen), drift, drift_ex, sorted(excl), level_drift,
                         level_all, by_backbone, runtime, pretraining, windows)
        else:
            why = proto + res['violations']
            if systematic:
                why = why + [f'systematic per-engine offset: median {drift["median"]:+.4f}, '
                             f'same-sign {drift["same_sign_frac"]:.1%}']
            evidence = dict(
                drift=_drift_block(drift, drift_ex, excl, level_drift, level_all, by_backbone),
                detail=rows,
                band={'pass': res['pass_band'], 'violations': res['violations']},
                strict_0_2={'pass': res['pass_strict'],
                            'violations': res['strict_violations']},
                protocol={'pass': not proto, 'violations': proto},
                aggregate=dict(mean_abs_dlev=res['mean_abs_dlev'],
                               mean_abs_dargmin=res['mean_abs_dargmin'],
                               n_cells=res['agg_n'], excluded=res['agg_excluded']),
                runtime_switches=dict(cells=runtime),
                pretraining=dict(cells=pretraining),
                window_counts=dict(cells=windows),
            )
            write_abort_record(Path(record), batch, why, bands_path, evidence)
    if not verdict_ok:
        print(f'RECONCILE: ABORT -- {", ".join(blocking)}', file=sys.stderr)
        return EXIT_ABORT
    if not res['pass_band']:
        print('RECONCILE: OK (drift) -- no systematic offset, so segment B may proceed; '
              'the band gate did NOT pass and is recorded as such')
    else:
        print('RECONCILE: OK')
    return EXIT_OK


# --------------------------------------------------------------------- gate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--module', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--reconcile-a', default=None, metavar='DIR_OR_JSON')
    ap.add_argument('--reconcile-frozen', default=None, metavar='DIR')
    ap.add_argument('--bands', default=str(repo_root() / BANDS_REL))
    ap.add_argument('--tol', type=float, default=RECON_TOL)
    ap.add_argument('--no-meta', action='store_true',
                    help='skip every reconcile gate (RMSE-only smoke)')
    ap.add_argument('--record', default=None, metavar='PATH',
                    help='write the record the run ledger reads (needs a committed gate1.py)')
    ap.add_argument('--batch', default='P2/W1/segment-A')
    ap.add_argument('--exclude', default=None, metavar='JSON',
                    help='JSON list of "model/subset/seed" cells excluded from the DRIFT '
                         'statistic only; band and strict_0.2 always use all 30 cells')
    a = ap.parse_args()

    exclude = None
    if a.exclude:
        raw = _read_json(Path(a.exclude))
        if not isinstance(raw, list):
            die('--exclude: expected a JSON list of "model/subset/seed" cell ids')
        exclude = []
        for x in raw:
            if isinstance(x, str):
                parts = x.split('/')
            elif isinstance(x, (list, tuple)):
                parts = list(x)
            else:
                parts = None
            if parts is None or len(parts) != 3 or not str(parts[2]).lstrip('s').isdigit():
                die(f'--exclude: cannot parse cell id {x!r}; expected "model/subset/seed"')
            m, s, d = str(parts[0]), str(parts[1]), int(str(parts[2]).lstrip('s'))
            # canonical form, so the id is comparable with the engine list's keys
            exclude.append(f'{m}/{s}/s{d}')

    if a.reconcile_a or a.reconcile_frozen:
        if not (a.reconcile_a and a.reconcile_frozen):
            die('reconcile needs BOTH --reconcile-a and --reconcile-frozen')
        if a.baseline or a.module:
            die('reconcile mode does not take --baseline/--module')
        if Path(a.reconcile_a).is_file() and not a.no_meta:
            die('a JSON of rows carries no history/epochs; pass --no-meta to reconcile it, '
                'or point --reconcile-a at the run directory instead')
        return reconcile(Path(a.reconcile_a), Path(a.reconcile_frozen), Path(a.bands),
                         a.tol, a.no_meta, a.record, a.batch, exclude)
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
