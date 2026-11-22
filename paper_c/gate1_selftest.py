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

# R2.1(7): the per-subset counts the committed anchor records. The synthetic
# fixture must carry them on BOTH sides, otherwise every case aborts on (7).
N_WINDOWS = {'FD001': 16762, 'FD004': 48691}
N_CUTS = {'FD001': 100, 'FD004': 250}


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
        lr=lr, weight_decay=0.0, val_frac=0.2, split_seed=7, rul_cap=125,
        training_data_sha256={'synthetic': '00000000'})))
    tm = cell_dir / 'training_metrics.json'
    meta = json.loads(tm.read_text()) if tm.exists() else {}
    sub = cell_dir.parent.name
    meta.update(epochs_run=epochs_run, best_epoch=best_epoch,
                min_delta=1e-4, tf32_matmul=False, tf32_cudnn=False,
                n_train_windows=N_WINDOWS[sub], n_validation_cuts=N_CUTS[sub])
    tm.write_text(json.dumps(meta))


def stamp_runtime(root):
    """ADD the attestation fields a conforming segment-A run writes.

    The committed anchor already carries training_metrics.json (with
    runtime_flags, but without min_delta/tf32_matmul/tf32_cudnn). This fixture
    ADDS only the missing fields so the rc=0 path can be exercised on real data;
    everything the anchor recorded is preserved, and the case name says the
    attestation fields are synthesized.
    """
    for m in METHODS:
        for s in SUBSETS:
            for sd in SEEDS:
                d = root / m / s / f's{sd}'
                fr = json.loads((d / 'FREEZE.json').read_text(encoding='utf-8-sig'))
                cfg = json.loads((d / 'config.json').read_text(encoding='utf-8-sig'))
                hist = json.loads((d / 'history.json').read_text(encoding='utf-8-sig'))
                p = d / 'training_metrics.json'
                meta = json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
                meta.setdefault('epochs_run', fr.get('epochs_run') or len(hist))
                meta.setdefault('best_epoch', fr.get('best_epoch'))
                meta.update(min_delta=cfg.get('min_delta'),
                            tf32_matmul=False, tf32_cudnn=False)
                p.write_text(json.dumps(meta))


ANCHOR_FILES = ('config.json', 'environment.json', 'FREEZE.json', 'history.json',
                'split.json', 'training_metrics.json', 'validation_metrics.json',
                'validation_predictions.csv', 'pretraining_metrics.json')


def extract_anchor(root):
    """Materialise ONLY the files the judge reads, straight from the blob.

    The fixture used to `git archive` the whole subtree and then delete the sealed
    files -- the one place in the pipeline that put 30 checkpoint.pt (43 MB) and
    six classes of TEST output on disk. The allowlist is the judge's own input
    list, so nothing the judge never reads is ever created.

    Returns (n_written, error). Byte-exact: blobs are written with write_bytes, so
    no line-ending translation can creep in.
    """
    tree = subprocess.run(['git', 'ls-tree', '-r', '--name-only', FROZEN, '--', FROZEN_SUB],
                          capture_output=True, text=True)
    if tree.returncode != 0:
        return 0, f'ls-tree failed: {tree.stderr.strip()}'
    seeds = {f's{x}' for x in SEEDS}
    keep = []
    for line in tree.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line[len(FROZEN_SUB):].lstrip('/').split('/')
        if len(parts) != 4:      # <model>/<subset>/s<seed>/<file>
            continue
        m, sub, sd = parts[0], parts[1], parts[2]
        if m in METHODS and sub in SUBSETS and sd in seeds and Path(line).name in ANCHOR_FILES:
            keep.append((f'{FROZEN}:{line}', line[len(FROZEN_SUB):].lstrip('/')))
    if not keep:
        return 0, 'allowlist matched no file'
    proc = subprocess.run(['git', 'cat-file', '--batch'],
                          input=('\n'.join(k for k, _ in keep) + '\n').encode(),
                          capture_output=True)
    if proc.returncode != 0:
        return 0, f'cat-file failed: {proc.stderr.decode(errors="replace").strip()}'
    data, pos, n = proc.stdout, 0, 0
    for spec, rel in keep:
        nl_at = data.index(b'\n', pos)
        head = data[pos:nl_at].split()
        if len(head) != 3:
            return 0, f'bad batch header for {spec}: {head}'
        size = int(head[2])
        start = nl_at + 1
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data[start:start + size])
        n += 1
        pos = start + size + 1
    return n, ''


def prune_sealed(root):
    """Drop the sealed TEST outputs and checkpoints the committed tree carries.

    `git archive d7bd9c17 fixed40` restores 22 files per cell (~56.5 MB/run),
    including test_predictions.csv, TEST_*.json and checkpoint.pt. gate1 never
    reads them, and the pruned 7-file working-tree copy is the intended state, so
    they are removed right after extraction -- this fixture was otherwise the one
    place in the pipeline that materialised sealed TEST artifacts on disk.
    """
    removed = 0
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if (p.name.startswith(('test_', 'TEST_'))
                or p.name in ('ACTUAL_TEST_INPUTS.json', 'checkpoint.pt')):
            p.unlink()
            removed += 1
    return removed


def replay_best(hist, min_delta):
    """The acceptance rule the config declares: accepted iff score < best - min_delta.

    The fixture must select by the SAME rule gate1 replays, otherwise it declares
    min_delta=1e-4 in config while selecting by plain argmin -- and gate1's replay
    check correctly flags the fixture as inconsistent.
    """
    best = hist[0]
    for rec in hist[1:]:
        if rec['val_engine_mean_rmse'] < best['val_engine_mean_rmse'] - min_delta:
            best = rec
    return best


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
            # a multiplicative shift changes the RMSE; adding a constant to the
            # noise mean barely moves it (mean error is ~0)
            base[(u, c)] = rng.normal(0.0, 5.0) * scale * (1.0 + level)
    drift = {e: rng.normal(0.0, 1.0) for e in range(1, epochs_run + 1)}
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
        best_epoch = replay_best(hist, 1e-4)['epoch']
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
    model = d.parent.parent.name
    if model == 'two_phase':
        # R2.1(6): the backbone pretrains on its paired subset, so the fixture must
        # carry the same pretraining record the anchor does
        cfgp = d / 'config.json'
        cfg = json.loads(cfgp.read_text())
        cfg['pretraining_epochs'] = 20
        cfgp.write_text(json.dumps(cfg))
        sub = d.parent.name
        (d / 'pretraining_metrics.json').write_text(json.dumps(dict(
            epochs=20, n_target_train_windows=N_WINDOWS[sub],
            n_paired_train_windows=N_WINDOWS[sub] + 3700,
            encoder_before_finetune_sha256='a' * 64,
            encoder_after_finetune_sha256='a' * 64)))
    return hist


def write_tree(root, levels=None, level=0.0, **kw):
    idx = 0
    for m in METHODS:
        for s in SUBSETS:
            for sd in SEEDS:
                idx += 1
                build_cell(root / m / s / f's{sd}', idx,
                           level=(levels or {}).get((m, s, sd), level), **kw)


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
            return root

        lvl_root = variant('level shift -> ABORT', 2, 'curve level', level=0.05)
        variant('protocol lr -> ABORT', 2, 'config.lr', lr=5e-3)

        # R2.1(7): the dropped-window data module. Both prepare() variants label the
        # training set 'all_cycle_left_padded' and agree on every config field; only
        # the window count separates them (-23% on FD001). This is the case that
        # would have caught the real v5/v6 batch.
        wc = td / 'window_wrong'
        write_tree(wc)
        wp = wc / 'aidgn' / 'FD001' / 's101' / 'training_metrics.json'
        wj = json.loads(wp.read_text())
        wj['n_train_windows'] = 12842
        wp.write_text(json.dumps(wj))
        rc, log, err = run_recon(wc, frozen, td, 'window_wrong')
        ok = (rc == 2 and 'R2.1(7)' in log and '12842' in log and '-3920 windows' in log)
        record('wrong data module -> ABORT', ok, f'rc={rc} {err}')

        # ... and the field being absent is not neutral either: the wrong module wrote
        # the same 9 fields, so "no n_train_windows" is a violation, not a skip.
        wm = td / 'window_missing'
        write_tree(wm)
        wmp = wm / 'aidgn' / 'FD001' / 's101' / 'training_metrics.json'
        wmj = json.loads(wmp.read_text())
        wmj.pop('n_train_windows')
        wmp.write_text(json.dumps(wmj))
        rc, log, err = run_recon(wm, frozen, td, 'window_missing')
        ok = rc == 2 and 'R2.1(7)' in log and 'no n_train_windows' in log
        record('window count missing -> ABORT', ok, f'rc={rc} {err}')

        # R2.1(6): two_phase pretraining is a protocol, so a mismatched paired-window
        # count must abort (before v6.5.5 the judge never looked at it)
        pt = td / 'pretrain_break'
        write_tree(pt)
        pj = pt / 'two_phase' / 'FD001' / 's101' / 'pretraining_metrics.json'
        jj = json.loads(pj.read_text())
        jj['n_paired_train_windows'] = 999
        pj.write_text(json.dumps(jj))
        rc, log, err = run_recon(pt, frozen, td, 'pretrain_break')
        ok = rc == 2 and 'R2.1(6)' in log
        record('pretraining mismatch -> ABORT', ok, f'rc={rc} {err}')

        # v6.2: the exclusion must reach R2.2, the statistic that decides the
        # verdict -- not just the per-engine reading. 9 shifted cells (the real
        # exclusion set's shape: fcstgnn x6 + two_phase/FD004 x3) stop segment B
        # when counted and must not when excluded.
        hot = ([('two_phase', 'FD004', d) for d in SEEDS]
               + [('fcstgnn', s, d) for s in SUBSETS for d in SEEDS])
        drift_root = td / 'r22_drift'
        write_tree(drift_root, levels={k: 0.30 for k in hot})
        rc, log, err = run_recon(drift_root, frozen, td, 'r22_drift')
        stops_when_counted = rc == 2 and 'STOPS segment B' in log
        excl9 = td / 'excl9.json'
        excl9.write_text(json.dumps([f'{m}/{s}/s{d}' for m, s, d in hot]))
        rc2, log2, err2 = run_recon(drift_root, frozen, td, 'r22_excl',
                                    extra=('--exclude', str(excl9)))
        ok = (stops_when_counted and rc2 == 0 and 'RECONCILE: OK' in log2
              and 'n=600 -> n=420' in log2)
        record('exclude flips R2.2 STOP -> OK', ok,
               f'counted rc={rc} excluded rc={rc2} '
               f'dropped={"n=600 -> n=420" in log2}')

        # a typo must be refused, not silently no-op (that was the v6.1 failure)
        bad_excl = td / 'excl_bad.json'
        bad_excl.write_text(json.dumps(['aidgn/FD001/s999']))
        rc, log, err = run_recon(frozen, frozen, td, 'excl_bad',
                                 extra=('--exclude', str(bad_excl)))
        ok = rc == 2 and 'not in this batch' in log and 's999' in log
        record('unknown exclusion -> ABORT', ok, f'rc={rc} {err}')

        # v6: epochs_run and rank are REPORTED, not blocking. Only protocol
        # fidelity (R2.1) and systematic level drift (R2.2) stop segment B.
        epi = td / 'epoch_drift'
        write_tree(epi, epochs_run=33)
        rc, log, err = run_recon(epi, frozen, td, 'epoch_drift')
        ok = rc == 0 and 'epoch drift' in log
        record('epoch drift is reported, not blocking', ok, f'rc={rc}')

        # the decision function, tested directly: one row per case, no plumbing
        sys.path.insert(0, str(GATE.parent))
        import gate1 as G

        def urow(**kw):
            base = dict(model='aidgn', subset='FD001', seed=101,
                        dlev=0.0, dargmin=0.0, rho=0.99)
            base.update(kw)
            return base

        ub = {k: dict(band=0.2) for k in
              ((m, s, d) for m in METHODS for s in SUBSETS for d in SEEDS)}

        def urows(**kw):
            out = []
            for m in METHODS:
                for s in SUBSETS:
                    for d in SEEDS:
                        row = urow(**kw)
                        row.update(model=m, subset=s, seed=d)
                        out.append(row)
            return out

        r = G.evaluate_cells([urow(rho=0.40)], ub)
        record('unit: low rank is reported', any('per-engine rank' in v for v in r['violations']),
               f'{r["violations"]}')

        wide = {k: dict(band=0.5) for k in ub}
        rows1 = urows()
        rows1[0]['dlev'] = 0.30
        r = G.evaluate_cells(rows1, wide)
        ok = (not r['pass_strict']) and not any('outside band' in v for v in r['violations'])
        record('unit: inside band but outside strict_0.2', ok,
               f'band_pass={not any("outside band" in v for v in r["violations"])} '
               f'strict_pass={r["pass_strict"]}')

        r = G.evaluate_cells([urow(dargmin=0.45)], ub)
        record('unit: argmin 0.45 is a gross error',
               any('argmin gap' in v for v in r['violations']), f'{r["violations"]}')

        r = G.evaluate_cells(urows(dlev=0.30), wide)
        ok = (not any('outside band' in v for v in r['violations'])
              and any('aggregate: mean|dlev|' in v for v in r['violations']))
        record('unit: all inside band, aggregate over line', ok,
               f'mean|dlev|={r["mean_abs_dlev"]:.3f}')

        # excluded cells must be listed, not just counted away
        r = G.evaluate_cells(urows(dlev=0.30), wide,
                             exclude_agg=[('aidgn', 'FD001', 101)])
        ok = r['agg_n'] == 29 and r['agg_excluded'] == ['aidgn/FD001/s101']
        record('unit: aggregate exclusions are listed', ok,
               f'n={r["agg_n"]} excluded={r["agg_excluded"]}')

        # the unit-level counterpart: R2.2 must honour the same ids
        rows9 = urows()
        hot9 = ([('two_phase', 'FD004', d) for d in SEEDS]
                + [('fcstgnn', s, d) for s in SUBSETS for d in SEEDS])
        for row in rows9:
            if (row['model'], row['subset'], row['seed']) in hot9:
                row['dlev'] = 0.30
        ld_all = G.level_drift_summary(rows9)
        ld_ex = G.level_drift_summary(
            rows9, exclude={f'{m}/{s}/s{d}' for m, s, d in hot9})
        ok = ld_all['stops'] and not ld_ex['stops'] and ld_ex['n_cells'] == 21
        record('unit: exclusion reaches R2.2', ok,
               f"all stops={ld_all['stops']} n_cells={ld_all['n_cells']}; "
               f"excluded stops={ld_ex['stops']} n_cells={ld_ex['n_cells']}")

        # v6.6: a one-backbone offset must be visible per backbone even when the
        # pooled same-sign rule cannot fire on it -- that is the segment-A shape
        rows_pb = urows()
        for i, row in enumerate(rows_pb):
            if row['model'] == 'two_phase':
                row['dlev'] = 0.30
            else:
                row['dlev'] = 0.10 if i % 2 == 0 else -0.10
        pooled = G.level_drift_summary(rows_pb)
        pb = G.level_drift_by_backbone(rows_pb)
        ok = ((not pooled['stops']) and pb['two_phase']['stops']
              and abs(pb['two_phase']['mean'] - 0.30) < 1e-12
              and abs(pb['aidgn']['mean']) < 1e-12)
        record('unit: per-backbone exposes one-backbone offset', ok,
               f"pooled stops={pooled['stops']} same-sign={pooled['same_sign_frac']:.0%}; "
               f"two_phase {pb['two_phase']['mean']:+.2f} stops={pb['two_phase']['stops']}")

        # min_delta replay: an improvement smaller than min_delta must NOT be
        # accepted, so the earlier epoch stays the selection -- and with
        # min_delta=0 the same curve picks the later one
        curve = {1: (10.0, {}), 2: (9.99999, {}), 3: (10.5, {})}
        e0, _ = G._replay_best(curve, 0.0)
        e1, _ = G._replay_best(curve, 1e-4)
        record('unit: min_delta replay picks the right epoch', e0 == 2 and e1 == 1,
               f'md=0 -> ep{e0}, md=1e-4 -> ep{e1}')

        # the frozen schema is the contract: a runner schema must be refused with
        # a message that says where the fix belongs, and a history without
        # per-epoch engine_squared_errors must be refused too (v0/v3 wrote only
        # epoch aggregates, and that silently removes the per-engine gates)
        def csv_probe(name, header, row):
            p = td / f'{name}.csv'
            with p.open('w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerow(row)
            return p

        seg = csv_probe('segmentA', ['unit', 'cut', 'y', 'pred'], [1, 90, 1.0, 3.0])
        try:
            G.engine_rmse_from_csv(seg)
            refused, hint = False, False
        except SystemExit as exc:
            refused, hint = (exc.code == 2), True
        record('unit: runner CSV schema refused', refused and hint,
               'unit,cut,y,pred -> exit 2 with the §5 hint')

        thin = td / 'thin_history'
        build_cell(thin / 'aidgn' / 'FD001' / 's101', 1)
        hp = thin / 'aidgn' / 'FD001' / 's101' / 'history.json'
        h = json.loads(hp.read_text())
        for rec in h:
            rec.pop('engine_squared_errors', None)
        hp.write_text(json.dumps(h))
        try:
            G.cell_curves(thin / 'aidgn' / 'FD001' / 's101', 'thin')
            thin_refused = False
        except SystemExit as exc:
            thin_refused = (exc.code == 2)
        record('unit: history without per-epoch SSE refused', thin_refused,
               f'refused={thin_refused}')

        # a missing band is an INPUT ERROR: exit 2 and no record, never "passed"
        short = td / 'bands_short.csv'
        lines = ['model,subset,seed,band']
        for i, k in enumerate(ub):
            if i == 0:
                continue
            lines.append(f'{k[0]},{k[1]},{k[2]},0.2')
        short.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        rc, log, err = run_recon(frozen, frozen, td, 'bands_short',
                                 extra=('--bands', str(short)))
        ok = rc == 2 and 'no band for 1 cell' in log
        record('missing band -> input error', ok, f'rc={rc} {err}')

        # an abort must still leave a record: otherwise the ledger cannot tell
        # "segment A ran and failed" from "segment A never ran"
        arec = td / 'abort.json'
        rc, log, err = run_recon(lvl_root, frozen, td, 'abortrec',
                                 extra=('--record', str(arec)))
        dirty0 = subprocess.run(['git', 'diff', '--quiet', '--', 'paper_c/gate1.py'],
                                capture_output=True).returncode != 0
        if dirty0:
            # provenance is verified on the abort path too, so a dirty tree
            # yields no record at all -- and exit 2, which is the input-error code
            ok = rc == 2 and not arec.exists()
            record('abort: dirty tree writes nothing', ok, f'rc={rc} exists={arec.exists()}')
        else:
            recj = json.loads(arec.read_text(encoding='utf-8')) if arec.exists() else {}
            ev = recj.get('computed_evidence', {})
            ok = (rc == 2 and recj.get('verdict') == 'abort'
                  and isinstance(ev.get('drift', {}).get('level'), dict)
                  and len(ev.get('detail', [])) == 30)
            record('abort writes verdict=abort + evidence', ok,
                   f'rc={rc} exists={arec.exists()} detail={len(ev.get("detail", []))}')

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
            root = real / FROZEN_SUB
            written, why = extract_anchor(root)
            if written == 0:
                record('real frozen fixed40', False, f'extract failed: {why}')
            else:
                # tripwire: the extractor must never have created a sealed file
                pruned = prune_sealed(root)
                sealed_left = sum(1 for _ in root.rglob('checkpoint.pt'))
                record('fixture writes the allowlist only (no sealed file on disk)',
                       pruned == 0 and sealed_left == 0,
                       f'{written} file(s), pruned={pruned}, checkpoint.pt={sealed_left}')
                # PURE anchor bytes first: no synthetic file anywhere. The committed
                # anchor DOES carry training_metrics.json (the 7-file copy in the
                # working tree is a pruned checkout), so item 5 fires 3x per cell --
                # min_delta absent, and both TF32 fields absent against the anchor's
                # recorded runtime_flags -- and nothing else may fire.
                rc, log, err = run_recon(root, root, td, 'real_pure')
                ok = (rc == 2 and 'protocol fidelity R2.1 (90)' in log
                      and log.count('R2.1(5)') == 90 and 'runtime_flags' in log
                      and 'config.' not in log and 'early-stop shape differs' not in log)
                record('pure frozen tree: only item 5 fires (3/cell)', ok,
                       f'rc={rc} item5={log.count("R2.1(5)")}')

                stamp_runtime(root)
                rc, log, err = run_recon(root, root, td, 'real_self')
                record('real self-match (stamped attestation)', rc == 0 and 'RECONCILE: OK' in log,
                       f'rc={rc} {err}')

                # the number the prereg's exclusion list must produce on the real
                # tree: 9 cells = 210 (fcstgnn FD001+FD004) + 150 (two_phase FD004)
                # engine readings, so 1050 -> 690 and R2.2 sees 21 cells.
                nine = ([f'fcstgnn/{s}/s{d}' for s in SUBSETS for d in SEEDS]
                        + [f'two_phase/FD004/s{d}' for d in SEEDS])
                ex9 = td / 'real_excl9.json'
                ex9.write_text(json.dumps(nine))
                rc, log, err = run_recon(root, root, td, 'real_excl9',
                                         extra=('--exclude', str(ex9)))
                line = next((ln for ln in log.splitlines() if 'n=1050' in ln), '')
                ok = rc == 0 and 'n=1050 -> n=690' in log and 'deciding: 21 cells' in log
                record('real 9-cell exclusion: 1050 -> 690', ok,
                       f'rc={rc} {line[:80] or err}')
                frag = [ln for ln in log.splitlines() if 'budget-truncated' in ln]
                record('real fragile set = 5', bool(frag) and 'cells: 5' in frag[0],
                       frag[0] if frag else 'no fragile line')

                # the ledger attributes the verdict to a commit, so a record must
                # never be written while the working tree differs from HEAD
                rec = td / 'rec.json'
                rc, log, err = run_recon(root, root, td, 'rec',
                                         extra=('--record', str(rec)))
                dirty = subprocess.run(['git', 'diff', '--quiet', '--', 'paper_c/gate1.py'],
                                       capture_output=True).returncode != 0
                if dirty:
                    ok = rc == 2 and 'commit the judging script' in log
                    record('record refuses a dirty tree', ok, f'rc={rc} dirty=True')
                else:
                    ok = rc == 0 and rec.exists()
                    detail = json.loads(rec.read_text()) if rec.exists() else {}
                    record('record written on clean tree', ok,
                           f'rc={rc} keys={sorted(detail)[:5]}')

                # R2.1(5): the runtime switches must be attested, and a missing
                # file or a True TF32 flag is a protocol violation, not a warning
                victim = ('aidgn', 'FD001', 101)
                bad_rt = td / 'real_runtime'
                shutil.copytree(root, bad_rt)
                p = bad_rt / victim[0] / victim[1] / f's{victim[2]}' / 'training_metrics.json'
                tmj = json.loads(p.read_text())
                tmj['tf32_matmul'] = True
                p.write_text(json.dumps(tmj))
                rc, log, err = run_recon(bad_rt, root, td, 'real_runtime')
                ok = rc == 2 and 'R2.1(5)' in log and 'tf32_matmul' in log
                record('tf32 on -> ABORT', ok, f'rc={rc} {err}')

                miss_rt = td / 'real_runtime_missing'
                shutil.copytree(root, miss_rt)
                (miss_rt / victim[0] / victim[1] / f's{victim[2]}'
                 / 'training_metrics.json').unlink()
                rc, log, err = run_recon(miss_rt, root, td, 'real_runtime_missing')
                ok = rc == 2 and 'no training_metrics.json' in log
                record('runtime attestation missing -> ABORT', ok, f'rc={rc} {err}')

                # the epoch-source fallback is exercised on the ANCHOR side: removing
                # training_metrics.json from segment A is now an R2.1(5) violation
                trim = td / 'real_trim'
                shutil.copytree(root, trim)
                (trim / victim[0] / victim[1] / f's{victim[2]}'
                 / 'training_metrics.json').unlink()
                # R2.1(7) makes the pruned anchor fail CLOSED: the epoch fallback still
                # works (the log shows history+FREEZE), but without the anchor's
                # n_train_windows the window count cannot be verified, so the run aborts.
                # A working-tree 7-file copy is therefore not a usable anchor.
                rc, log, err = run_recon(root, trim, td, 'real_trim')
                src = [ln for ln in log.splitlines() if 'epoch sources' in ln]
                ok = (rc == 2 and any('history+FREEZE|training_metrics' in ln for ln in src)
                      and 'R2.1(7)' in log and 'no n_train_windows' in log)
                record('anchor-side trim fallback (history+FREEZE)', ok, f'rc={rc} {src or err}')

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
