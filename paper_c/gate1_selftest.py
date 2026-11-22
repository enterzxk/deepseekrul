"""Self-test for gate1.py.

Three suites, because a checker that has only ever been seen to pass is not a
checker:

 1. gate verdicts (synthetic) -- PASS when every backbone improves; FAIL (rc 1)
    when the EOL guardrail, the win rate, or coverage vetoes it; ABORT (rc 2)
    when the cut column is in the wrong space.
 2. reconcile gates (synthetic) -- each of the five gates must be able to fire
    on its own: protocol mismatch, stopping-epoch drift, curve level, per-engine
    rank, and an artifact whose metrics do not correspond to its own curve.
 3. reconcile against the real frozen fixed40 (commit d7bd9c17) -- the actual
    reference must reconcile with itself, report the 5 budget-truncated cells,
    still work when training_metrics.json is missing, and catch a tampered
    predictions file.

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
    print(f'{name:36} {"OK" if ok else "MISMATCH"}   {detail}')


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

def write_csv(cell_dir, rows):
    with (cell_dir / 'validation_predictions.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['unit', 'y_true', 'y_pred', 'signed_error', 'cut'])
        for eng, y, p, c in rows:
            w.writerow([eng, y, p, p - y, c])


def write_history(cell_dir, hist):
    (cell_dir / 'history.json').write_text(json.dumps(hist))


def write_metrics(cell_dir, rec):
    sse = [e['sum_squared_error'] for e in rec['engine_squared_errors']]
    n = [e['n'] for e in rec['engine_squared_errors']]
    (cell_dir / 'validation_metrics.json').write_text(json.dumps(dict(
        val_engine_mean_rmse=rec['val_engine_mean_rmse'],
        val_pooled_rmse=float(np.sqrt(sum(sse) / sum(n))),
        engine_squared_errors=rec['engine_squared_errors'])))


def write_meta(cell_dir, epochs_run, best_epoch, epochs=40, window=30, lr=1e-3):
    (cell_dir / 'config.json').write_text(json.dumps(dict(
        epochs=epochs, patience=12, min_delta=1e-4, window=window, batch=128,
        lr=lr, weight_decay=0.0, val_frac=0.2, split_seed=7, rul_cap=125)))
    tm = cell_dir / 'training_metrics.json'
    meta = json.loads(tm.read_text()) if tm.exists() else {}
    meta.update(epochs_run=epochs_run, best_epoch=best_epoch)
    tm.write_text(json.dumps(meta))


def build_cell(d, idx, epochs_run=27, epochs=40, best_epoch=None, level=0.0, rank_flip=False,
               lr=1e-3):
    """One cell whose CSV, history and metrics are mutually consistent."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(90000 + idx)
    engs = list(range(1, N_ENG + 1))
    base = {}
    for u in engs:
        scale = (1.0 + (N_ENG - u) / N_ENG * 3.0) if rank_flip else 1.0
        for c in CUTS:
            base[(u, c)] = rng.normal(0.0, 5.0) * scale
    drift = {e: rng.normal(0.0, 1.0) + level for e in range(1, epochs_run + 1)}
    hist = []
    for e in range(1, epochs_run + 1):
        ese, per = [], {}
        for u in engs:
            sse = sum((base[(u, c)] + drift[e]) ** 2 for c in CUTS)
            ese.append(dict(unit=u, n=len(CUTS), sum_squared_error=sse))
            per[u] = (sse / len(CUTS)) ** 0.5
        hist.append(dict(epoch=e, val_engine_mean_rmse=float(np.mean(list(per.values()))),
                         engine_squared_errors=ese))
    if best_epoch is None:
        best_epoch = min(hist, key=lambda r: r['val_engine_mean_rmse'])['epoch']
    rows = []
    for u in engs:
        life = 180 + 4 * u
        for c in CUTS:
            y = max(1.0, life * (1.0 - c))
            rows.append((u, y, y + base[(u, c)] + drift[best_epoch], round(c * life)))
    write_csv(d, rows)
    write_history(d, hist)
    write_metrics(d, [r for r in hist if r['epoch'] == best_epoch][0])
    write_meta(d, epochs_run=epochs_run, best_epoch=best_epoch, epochs=epochs, lr=lr)
    return hist


def write_tree(root, **kw):
    idx = 0
    for m in METHODS:
        for s in SUBSETS:
            for sd in SEEDS:
                idx += 1
                build_cell(root / m / s / f's{sd}', idx, **kw)


def run_recon(a_seg, frozen, td, tag, extra=()):
    r = subprocess.run([sys.executable, str(GATE), '--reconcile-a', str(a_seg),
                        '--reconcile-frozen', str(frozen), *extra],
                       capture_output=True, text=True)
    log = (r.stdout or '') + (r.stderr or '')
    (Path(td) / f'{tag}.log').write_text(log)
    return r.returncode, log, (r.stderr or '').strip().splitlines()[-1:]


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
                ok = (rc == 2) and any('ABSOLUTE CYCLE' in e for e in err)
                record(kind, ok, f'rc={rc} {err}')
            else:
                record(kind, rc == want_rc and got == want,
                       f'expect {want} rc={want_rc}, got {got} rc={rc}')

        print('\n--- suite 2: reconcile gates (synthetic) ---')
        frozen = td / 'frozen'
        write_tree(frozen)
        rc, log, err = run_recon(frozen, frozen, td, 'self')
        record('self-match', rc == 0 and 'RECONCILE: OK' in log, f'rc={rc} {err}')

        def variant(name, expect_rc, needle, **kw):
            slug = ''.join(ch if ch.isalnum() else '_' for ch in name).strip('_')
            root = td / slug
            write_tree(root, **kw)
            rc, log, err = run_recon(root, frozen, td, slug)
            ok = rc == expect_rc and needle in log
            record(name, ok, f'rc={rc} needle={needle!r} found={needle in log}')

        variant('level shift +0.5 -> ABORT', 2, 'curve level', level=0.5)
        variant('rank scramble -> ABORT', 2, 'rank correlation', rank_flip=True)
        variant('epoch drift -> ABORT', 2, 'stopping epoch', epochs_run=33)
        variant('protocol lr -> ABORT', 2, 'config.lr', lr=5e-3)

        # an artifact whose metrics no longer correspond to its own curve
        tam = td / 'tampered'
        shutil.copytree(frozen, tam)
        cell = ('aidgn', 'FD001', 101)
        cd = tam / cell[0] / cell[1] / f's{cell[2]}'
        with (cd / 'validation_predictions.csv').open(newline='') as fh:
            rows = list(csv.DictReader(fh))
        with (cd / 'validation_predictions.csv').open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['unit', 'y_true', 'y_pred', 'signed_error', 'cut'])
            for r in rows:
                p = float(r['y_pred']) + (0.5 if int(float(r['unit'])) == 3 else 0.0)
                w.writerow([r['unit'], r['y_true'], p, p - float(r['y_true']), r['cut']])
        rc, log, err = run_recon(tam, frozen, td, 'tampered')
        ok = rc == 2 and 'disagree with engine_squared_errors' in log
        record('tampered CSV -> ABORT', ok, f'rc={rc} {err}')

        jsonin = td / 'a_seg.json'
        rows = []
        for m in METHODS:
            for s in SUBSETS:
                for sd in SEEDS:
                    with (frozen / m / s / f's{sd}' / 'validation_predictions.csv').open(
                            newline='') as fh:
                        for i, r in enumerate(csv.DictReader(fh)):
                            rows.append(dict(method=m, subset=s, seed=sd,
                                             engine=int(float(r['unit'])),
                                             cut=(i % len(CUTS) + 1) / len(CUTS),
                                             y=float(r['y_true']), pred=float(r['y_pred'])))
        jsonin.write_text(json.dumps(rows))
        rc, log, err = run_recon(jsonin, frozen, td, 'jsonin', extra=('--no-meta',))
        record('JSON input with --no-meta', rc == 0, f'rc={rc} {err}')
        rc, log, err = run_recon(jsonin, frozen, td, 'jsonin_strict')
        ok = rc == 2 and 'carries no history' in log
        record('JSON without --no-meta -> ABORT', ok, f'rc={rc} {err}')

        print('\n--- suite 3: real frozen fixed40 ---')
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
                rc, log, err = run_recon(root, root, td, 'real_self')
                record('real self-match', rc == 0 and 'RECONCILE: OK' in log, f'rc={rc} {err}')
                frag = [ln for ln in log.splitlines() if 'budget-truncated' in ln]
                record('real fragile set = 5', bool(frag) and 'cells: 5' in frag[0],
                       frag[0] if frag else 'no fragile line')

                victim = ('aidgn', 'FD001', 101)
                trim = td / 'real_trim'
                shutil.copytree(root, trim)
                (trim / victim[0] / victim[1] / f's{victim[2]}'
                 / 'training_metrics.json').unlink()
                rc, log, err = run_recon(trim, root, td, 'real_trim')
                src = [ln for ln in log.splitlines() if 'epoch sources' in ln]
                ok = rc == 0 and any('training_metrics|history+FREEZE' in ln for ln in src)
                record('trim fallback (history+FREEZE)', ok, f'rc={rc} {src or err}')

                bad = td / 'real_bad'
                shutil.copytree(root, bad)
                probe = bad / victim[0] / victim[1] / f's{victim[2]}' / 'validation_predictions.csv'
                with probe.open(newline='') as fh:
                    eng = int(float(next(csv.DictReader(fh))['unit']))
                with probe.open(newline='') as fh:
                    rr = list(csv.DictReader(fh))
                with probe.open('w', newline='') as fh:
                    w = csv.writer(fh)
                    w.writerow(['unit', 'y_true', 'y_pred', 'signed_error', 'cut'])
                    for r in rr:
                        p = float(r['y_pred']) + (0.5 if int(float(r['unit'])) == eng else 0.0)
                        w.writerow([r['unit'], r['y_true'], p, p - float(r['y_true']), r['cut']])
                rc, log, err = run_recon(bad, root, td, 'real_bad')
                ok = rc == 2 and 'disagree with engine_squared_errors' in log
                record('real tampered cell -> ABORT', ok, f'rc={rc} engine={eng}')

    n_bad = sum(1 for _, ok in results if not ok)
    print(f'\nSELFTEST {"PASSED" if not n_bad else "FAILED"}  '
          f'({len(results) - n_bad}/{len(results)} checks)')
    return 0 if not n_bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
