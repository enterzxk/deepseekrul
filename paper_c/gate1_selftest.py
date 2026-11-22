"""Self-test for gate1.py.

Three suites, because a checker that has only ever been seen to pass is not a
checker:

 1. gate scenarios (synthetic) -- the gate must PASS when every backbone
    improves, and must FAIL when a sub-criterion (EOL guardrail, win rate,
    coverage) vetoes it. Exit code must be 1 for FAIL, never 2.
 2. reconcile (synthetic tree) -- must return 0 on an exact match and ABORT (2)
    on drift, on a stale metrics.json, and on a cut column in the wrong space.
 3. reconcile (real frozen fixed40, commit d7bd9c17) -- the actual reference
    artifact must reconcile with itself at delta 0, and a single perturbed
    engine must abort. Skipped loudly when the commit is unavailable.

Nothing here writes into the repo; everything happens in a temp directory.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

METHODS = ('aidgn', 'two_phase', 'dual_mixer', 'ffve', 'fcstgnn')
SUBSETS = ('FD001', 'FD004')
SEEDS = (101, 202, 303)
CUTS = (0.40, 0.55, 0.70, 0.85, 0.95)
N_ENG = 20
FROZEN = 'd7bd9c17'
FROZEN_SUB = 'paper_b/experiments/open_source_matched_20261112/physical_runs/fixed40'

GATE = Path('paper_c/gate1.py')
results = []


def record(name, ok, detail):
    results.append((name, ok))
    print(f'{name:34} {"OK" if ok else "MISMATCH"}   {detail}')


# --------------------------------------------------------------- suite 1

def build_gate(kind):
    """Return (baseline_rows, module_rows) for one synthetic scenario."""
    rng = np.random.default_rng(20261122)
    base, mod = [], []
    for method in METHODS:
        for subset in SUBSETS:
            for seed in SEEDS:
                for eng in range(1, N_ENG + 1):
                    life = 180 + 4 * eng
                    for cut in CUTS:
                        y = max(1.0, life * (1.0 - cut))
                        eb = rng.normal(0.0, 6.0)
                        if kind == 'all_good':
                            em = eb * 0.5
                        elif kind == 'three_good':
                            em = eb * (0.5 if method in METHODS[:3] else 1.5)
                        elif kind == 'eol_break':
                            # Better everywhere except the EOL cut, where the
                            # absolute error must be genuinely larger: halving
                            # then adding 3 still beats the baseline
                            # (E|N(3,3)|=3.8 < E|N(0,6)|=4.8), so the offset has
                            # to be big enough to cross over.
                            em = eb * 0.5 if cut < 0.95 else eb * 0.5 + 8.0
                        elif kind == 'win_rate_break':
                            # mean RMSE better, but only half the engines improve
                            em = eb * (0.2 if eng % 2 == 0 else 1.4)
                        elif kind == 'cycle_cuts':
                            em = eb * 0.5
                        else:
                            raise SystemExit(kind)
                        for store, pred in ((base, y + eb), (mod, y + em)):
                            c = round(cut * life) if kind == 'cycle_cuts' else cut
                            store.append(dict(method=method, subset=subset, seed=seed,
                                              engine=eng, cut=c, y=y, pred=pred))
    return base, mod


def run_gate(kind, td):
    base, mod = build_gate(kind)
    bp, mp = Path(td) / 'base.json', Path(td) / 'mod.json'
    bp.write_text(json.dumps(base))
    mp.write_text(json.dumps(mod))
    r = subprocess.run([sys.executable, str(GATE), '--baseline', str(bp), '--module', str(mp),
                        '--json', str(Path(td) / 'out.json')],
                       capture_output=True, text=True)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.startswith(('GATE', 'backbones'))]
    got = tail[-1].split(':')[-1].strip() if tail else '?'
    return r.returncode, got, (r.stderr or '').strip().splitlines()[-1:] 


# --------------------------------------------------------------- suite 2

def write_frozen_tree(root, life_of=lambda eng: 180 + 4 * eng):
    """Synthetic frozen tree in the layout gate1 expects."""
    idx = 0
    for method in METHODS:
        for subset in SUBSETS:
            for seed in SEEDS:
                idx += 1
                d = root / method / subset / f's{seed}'
                d.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(90000 + idx)
                rows, ese = [], []
                for eng in range(1, N_ENG + 1):
                    life = life_of(eng)
                    sse = 0.0
                    for cut in CUTS:
                        y = max(1.0, life * (1.0 - cut))
                        e = rng.normal(0.0, 6.0)
                        p = y + e
                        rows.append((eng, y, p, round(cut * life)))
                        sse += (p - y) ** 2
                    ese.append(dict(unit=eng, n=len(CUTS), sum_squared_error=sse))
                write_csv(d, rows)
                write_metrics(d, ese)
                write_meta(d, epochs_run=27, best_epoch=15)


def write_meta(cell_dir, epochs_run, best_epoch, epochs=40, window=30):
    """The epoch anchor and the hyperparameters the reconcile compares."""
    (cell_dir / 'config.json').write_text(json.dumps(dict(
        epochs=epochs, patience=12, min_delta=1e-4, window=window, batch=128,
        lr=1e-3, weight_decay=0.0, val_frac=0.2, split_seed=7, rul_cap=125)))
    tm = cell_dir / 'training_metrics.json'
    meta = json.loads(tm.read_text()) if tm.exists() else {}
    meta.update(epochs_run=epochs_run, best_epoch=best_epoch)
    tm.write_text(json.dumps(meta))


def patch_meta(root, cell, **kw):
    write_meta(root / cell[0] / cell[1] / f's{cell[2]}', **kw)


def write_csv(cell_dir, rows):
    with (cell_dir / 'validation_predictions.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['unit', 'y_true', 'y_pred', 'signed_error', 'cut'])
        for eng, y, p, c in rows:
            w.writerow([eng, y, p, p - y, c])


def write_metrics(cell_dir, ese):
    per = {e['unit']: (e['sum_squared_error'] / e['n']) ** 0.5 for e in ese}
    n_tot = sum(e['n'] for e in ese)
    sse_tot = sum(e['sum_squared_error'] for e in ese)
    (cell_dir / 'validation_metrics.json').write_text(json.dumps(dict(
        val_engine_mean_rmse=float(np.mean(list(per.values()))),
        val_pooled_rmse=float(np.sqrt(sse_tot / n_tot)),
        engine_squared_errors=ese)))


def reread_metrics(cell_dir):
    """Recompute engine_squared_errors from the CSV, leaving the CSV untouched."""
    per = {}
    with (cell_dir / 'validation_predictions.csv').open(newline='') as fh:
        for r in csv.DictReader(fh):
            u = int(float(r['unit']))
            per.setdefault(u, []).append((float(r['y_true']), float(r['y_pred'])))
    ese = [dict(unit=u, n=len(v), sum_squared_error=float(sum((p - y) ** 2 for y, p in v)))
           for u, v in sorted(per.items())]
    write_metrics(cell_dir, ese)


def perturb(root, cell, eng, delta, fix_metrics):
    """Shift one engine's predictions in one cell, optionally keeping metrics honest."""
    d = root / cell[0] / cell[1] / f's{cell[2]}'
    rows = []
    with (d / 'validation_predictions.csv').open(newline='') as fh:
        for r in csv.DictReader(fh):
            u, y, p = int(float(r['unit'])), float(r['y_true']), float(r['y_pred'])
            if u == eng:
                p += delta
            rows.append((u, y, p, r['cut']))
    write_csv(d, rows)
    if fix_metrics:
        reread_metrics(d)


def run_recon(a_seg, frozen, td, tag, extra=()):
    r = subprocess.run([sys.executable, str(GATE), '--reconcile-a', str(a_seg),
                        '--reconcile-frozen', str(frozen), *extra],
                       capture_output=True, text=True)
    (Path(td) / f'{tag}.log').write_text((r.stdout or '') + (r.stderr or ''))
    tail = [ln for ln in r.stdout.strip().splitlines()
            if ln.startswith(('RECONCILE', 'worst per-engine', 'cells:'))]
    err = (r.stderr or '').strip().splitlines()[-1:]
    return r.returncode, tail, err


# ------------------------------------------------------------------ main

def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print('--- suite 1: gate verdicts ---')
        for kind, want_rc, want in (('all_good', 0, 'PASS'), ('three_good', 1, 'FAIL'),
                                    ('eol_break', 1, 'FAIL'), ('win_rate_break', 1, 'FAIL'),
                                    ('cycle_cuts', 2, '?')):
            rc, got, err = run_gate(kind, td)
            if kind == 'cycle_cuts':
                # absolute-cycle cuts must ABORT, not silently misplace the guardrail
                ok = (rc == 2) and any('ABSOLUTE CYCLE' in e for e in err)
                record(kind, ok, f'rc={rc} {err}')
            else:
                record(kind, rc == want_rc and got == want,
                       f'expect {want} rc={want_rc}, got {got} rc={rc}')

        print('\n--- suite 2: reconcile on a synthetic tree ---')
        frozen = td / 'frozen'
        write_frozen_tree(frozen)
        rc, tail, err = run_recon(frozen, frozen, td, 'self')
        record('recon self-match', rc == 0 and any('RECONCILE: OK' in t for t in tail),
               f'rc={rc} {tail}')

        drift = td / 'drift'
        shutil.copytree(frozen, drift)
        cell = ('aidgn', 'FD001', 101)
        perturb(drift, cell, eng=3, delta=0.5, fix_metrics=True)
        rc, tail, err = run_recon(drift, frozen, td, 'drift')
        log = (td / 'drift.log').read_text()
        ok = rc == 2 and 'aidgn       FD001 s101' in log and '1 OVER' in log
        record('recon drift -> ABORT', ok, f'rc={rc} flagged={ok}')

        stale = td / 'stale'
        shutil.copytree(frozen, stale)
        # The metrics/CSV cross-check guards the REFERENCE, so the stale side
        # has to be the frozen one: a reference that contradicts its own
        # predictions must abort before any comparison happens.
        perturb(stale, cell, eng=3, delta=0.5, fix_metrics=False)
        rc, tail, err = run_recon(frozen, stale, td, 'stale')
        ok = rc == 2 and any('disagree with engine_squared_errors' in e for e in err)
        record('recon stale reference -> ABORT', ok, f'rc={rc} {err}')

        jsonin = td / 'a_seg.json'
        rows = []
        for method in METHODS:
            for subset in SUBSETS:
                for seed in SEEDS:
                    d = frozen / method / subset / f's{seed}'
                    with (d / 'validation_predictions.csv').open(newline='') as fh:
                        for i, r in enumerate(csv.DictReader(fh)):
                            rows.append(dict(method=method, subset=subset, seed=seed,
                                             engine=int(float(r['unit'])),
                                             cut=(i % len(CUTS) + 1) / len(CUTS),
                                             y=float(r['y_true']), pred=float(r['y_pred'])))
        jsonin.write_text(json.dumps(rows))
        rc, tail, err = run_recon(jsonin, frozen, td, 'jsonin', extra=('--no-meta',))
        record('recon accepts JSON (--no-meta)', rc == 0, f'rc={rc} {err}')
        rc, tail, err = run_recon(jsonin, frozen, td, 'jsonin_strict')
        ok = rc == 2 and any('carries no epochs_run' in e for e in err)
        record('JSON without --no-meta -> ABORT', ok, f'rc={rc} {err}')

        epi = td / 'epoch_drift'
        shutil.copytree(frozen, epi)
        patch_meta(epi, ('dual_mixer', 'FD001', 202), epochs_run=33, best_epoch=21)
        rc, tail, err = run_recon(epi, frozen, td, 'epoch_drift')
        ok = rc == 2 and any('different epoch' in e for e in err)
        record('recon epoch drift -> ABORT', ok, f'rc={rc} {err}')

        # A budget-truncated anchor was cut off while still improving, so its
        # |delta| must not be read as environment drift.
        fcell = ('ffve', 'FD004', 303)
        patch_meta(frozen, fcell, epochs_run=40, best_epoch=35)
        frag = td / 'fragile'
        shutil.copytree(frozen, frag)
        perturb(frag, fcell, eng=5, delta=0.6, fix_metrics=True)
        rc, tail, err = run_recon(frag, frozen, td, 'fragile')
        log = (td / 'fragile.log').read_text()
        ok = rc == 0 and 'OVER fragile' in log and 'outside tolerance: 0' in log
        record('fragile cell over tol -> OK', ok,
               f'rc={rc} fragile_flagged={"OVER fragile" in log}')

        print('\n--- suite 3: reconcile against the real frozen fixed40 ---')
        have = subprocess.run(['git', 'cat-file', '-e', FROZEN], capture_output=True).returncode == 0
        if not have:
            record('real frozen fixed40', False, f'SKIPPED -- {FROZEN} not in this repo')
        else:
            real = td / 'real'
            real.mkdir()
            tar = real / 'f.tar'
            a = subprocess.run(['git', 'archive', '--format=tar', '--output', str(tar),
                                FROZEN, FROZEN_SUB], capture_output=True, text=True)
            if a.returncode != 0:
                record('real frozen fixed40', False, f'git archive failed: {a.stderr.strip()}')
            else:
                subprocess.run(['tar', '-xf', str(tar), '-C', str(real)], check=True)
                root = real / FROZEN_SUB
                rc, tail, err = run_recon(root, root, td, 'real_self')
                n_cells = sum(1 for p in root.glob('*/*/*') if p.is_dir())
                record('real self-match', rc == 0 and any('cells: 30,' in t for t in tail),
                       f'rc={rc} tree_cells={n_cells} {tail[:2]}')
                # the fragile set must be derived from the data, not hardcoded:
                # the anchor dump independently reported 5 budget-truncated cells
                log = (td / 'real_self.log').read_text()
                frag_line = [ln for ln in log.splitlines() if 'budget-truncated' in ln]
                record('real fragile set = 5', bool(frag_line) and 'cells: 5' in frag_line[0],
                       frag_line[0] if frag_line else 'no fragile line')

                bad = td / 'real_bad'
                shutil.copytree(root, bad)
                # a non-fragile cell: two_phase/FD004 is budget-truncated and is
                # deliberately not decisive on its own
                victim = ('aidgn', 'FD001', 101)
                # perturb an engine that actually exists in that cell; picking a
                # fixed id can silently edit nothing and pass for the wrong reason
                probe = bad / victim[0] / victim[1] / f's{victim[2]}' / 'validation_predictions.csv'
                with probe.open(newline='') as fh:
                    eng = int(float(next(csv.DictReader(fh))['unit']))
                perturb(bad, victim, eng=eng, delta=0.5, fix_metrics=True)
                rc, tail, err = run_recon(bad, root, td, 'real_bad')
                log = (td / 'real_bad.log').read_text()
                # assert on the victim's OWN line: every cell is printed, so
                # "the cell appears in the log" would pass even with no drift
                vline = [ln for ln in log.splitlines()
                         if ln.startswith(f'{victim[0]:11} {victim[1]} s{victim[2]}')]
                ok = rc == 2 and len(vline) == 1 and 'OVER' in vline[0]
                record('real single-engine drift', ok,
                       f'rc={rc} engine={eng} victim={vline}')

                # a trimmed tree may omit training_metrics.json; epochs_run must
                # then come from history.json and best_epoch from FREEZE.json
                trim = td / 'real_trim'
                shutil.copytree(root, trim)
                (trim / victim[0] / victim[1] / f's{victim[2]}'
                 / 'training_metrics.json').unlink()
                rc, tail, err = run_recon(trim, root, td, 'real_trim')
                log = (td / 'real_trim.log').read_text()
                src_line = [ln for ln in log.splitlines() if 'epoch sources' in ln]
                ok = rc == 0 and any('training_metrics|history+FREEZE' in ln for ln in src_line)
                record('real trim fallback (history+FREEZE)', ok,
                       f'rc={rc} {src_line or err}')

    n_bad = sum(1 for _, ok in results if not ok)
    print(f'\nSELFTEST {"PASSED" if not n_bad else "FAILED"}  '
          f'({len(results) - n_bad}/{len(results)} checks)')
    return 0 if not n_bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
