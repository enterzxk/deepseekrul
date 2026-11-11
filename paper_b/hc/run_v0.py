"""V0 runs: the frozen DSPT through the frozen trainer, same artifact schema.

Gate A needs V0 rows, and V0 must be trained by the frozen ``train_one`` (not by
``train_hc``) so the baseline is the published model rather than a
reimplementation. This module is a thin wrapper: it owns no training logic, only
the launch/close protocol and the artifact layout shared with ``run_hc``.

V0 triples (task book §3, X3 ruling)::

    V0C  full / basic / hf  + cycle on
    V0N  full / basic / hf  + cycle off
    V0-NoHI  == A3p, trained once by run_hc and referenced from both tables

Usage::

    python -m paper_b.hc.run_v0 --variant V0C --subset FD001 --seed 101 \
        --run-id HC-C-V0C-FD001-s101-... --hypothesis "..."
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from .config_hc import (FOLD, SPLIT_HASH_REF, SPLIT_SEED, VAL_CUTS, VAL_FRAC,
                        assert_protocol_matches_taskbook, split_hash)
from .run_hc import (REPO, code_hashes, code_lock_guard, env_info,
                     exclusive_write, now_utc, worktree_guard)

V0 = {
    'V0C': dict(use_cycle=True),
    'V0N': dict(use_cycle=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=sorted(V0))
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--hypothesis', required=True)
    ap.add_argument('--gate', default='A')
    ap.add_argument('--span', type=int, default=70)
    ap.add_argument('--baseline-version', default=None)
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--out-root', default='experiments/hc')
    ap.add_argument('--registry-root', default=str(REPO / 'docs' / 'paper_b_hc' / 'registry'))
    ap.add_argument('--code-lock', default=str(REPO / 'docs' / 'paper_b_hc' / 'registry' /
                                               'code_lock.json'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--smoke', action='store_true', help='1-epoch pipeline check; not evidence')
    a = ap.parse_args()

    assert_protocol_matches_taskbook()

    guard = worktree_guard()
    if guard is None:
        guard = code_lock_guard(a.code_lock)
    print(f'[v0] guard ok mode={guard["mode"]} sha={guard["git_sha"][:12]}', flush=True)

    from ..training.data import load_train, split_units
    import numpy as np
    units = np.sort(load_train(Path(a.data_dir), a.subset)['unit'].unique())
    _, val_units = split_units(units, SPLIT_SEED, FOLD, VAL_FRAC)
    got = split_hash(val_units)
    want = SPLIT_HASH_REF[a.subset]
    if got != want:
        raise SystemExit(f'SPLIT DRIFT on {a.subset}: {got} != {want}; refusing to run')
    print(f'[v0] split guard ok {a.subset} hash={got}', flush=True)

    lock = json.loads(Path(a.code_lock).read_text()) if Path(a.code_lock).exists() else {}
    baseline_version = a.baseline_version or lock.get('baseline_version', 'unregistered')
    git_sha_field = guard['git_sha'] if guard['mode'] == 'git' else 'WORKTREE'

    cfg = dict(
        model='dspt', subset=a.subset, seed=a.seed, span=a.span,
        epochs=1 if a.smoke else 40, patience=1 if a.smoke else 12,
        batch=512, lr=1e-3, weight_decay=1e-4, d_model=64, layers=2, dropout=0.2,
        mode='dual', ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='none',
        hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=V0[a.variant]['use_cycle'],
        use_pad_mask=True, input_mode='raw', aux_dual=True, instnorm=False,
        rul_cap=125.0, split_seed=SPLIT_SEED, fold=FOLD, val_frac=VAL_FRAC,
        n_cond=6, cuts=list(VAL_CUTS), legacy_rul=False, global_hi=False,
        legacy_ms_pad=False, legacy_ema=False,
        tag=f'V0_{a.variant}', data_dir=a.data_dir,
        run_id=a.run_id, stage='C', variant=a.variant,
        val_engine_ids_hash=got, val_units=sorted(int(u) for u in val_units),
        hypothesis=a.hypothesis, gate=a.gate, smoke=bool(a.smoke),
        budget_override=('smoke: 1 epoch, not evidence' if a.smoke else None),
    )

    out = Path(a.out_root) / a.run_id
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'REFUSED: {out} is not empty; a run_id is used once')

    reg = Path(a.registry_root)
    hashes = code_hashes()
    launch = dict(
        run_id=a.run_id, stage='C', variant=a.variant, subset=a.subset, seed=a.seed,
        span=a.span, hypothesis=a.hypothesis, gate=a.gate, host=env_info()['host'],
        smoke=bool(a.smoke), written_by='run_v0', process_start_utc=now_utc(),
        baseline_version=baseline_version, git_sha=git_sha_field, worktree_dirty=False,
        worktree=guard, code_lock_git_sha=(lock.get('git_sha') if guard['mode'] == 'lock' else None),
        code_md5=hashes,
        trainer='paper_b.training.train.train_one',
        data_split=dict(split_seed=SPLIT_SEED, fold=FOLD, val_frac=VAL_FRAC,
                        val_engine_ids_hash=got),
        budget=dict(epochs=cfg['epochs'], patience=cfg['patience'],
                    budget_override=cfg['budget_override'], est_gpu_min=25),
        launched_utc=now_utc())
    launch_path = reg / 'launch' / f'{a.run_id}.json'
    exclusive_write(launch_path, launch)
    back = json.loads(launch_path.read_text())
    if back.get('code_md5') != code_hashes() or back.get('written_by') != 'run_v0':
        raise SystemExit(f'SELF-CHECK FAILED: {launch_path}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'launch.json').write_text(json.dumps(back, indent=1))
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))

    logf = open(out / 'train.log', 'w', encoding='utf-8')

    def log(*args):
        msg = ' '.join(str(x) for x in args)
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    from ..training.train import train_one
    log(f'[v0] run_id={a.run_id} variant={a.variant} subset={a.subset} seed={a.seed}')
    try:
        m = train_one(cfg, device=a.device, out_root=out / 'frozen')
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        failed = dict(run_id=a.run_id, status='failed', closed_utc=now_utc(),
                      written_by='run_v0', main_metric='val_engine_mean_rmse',
                      metrics={}, notes=(f'FAILED {type(exc).__name__}: {exc}'[:400]))
        (out / 'close.json').write_text(json.dumps(failed, indent=1))
        (reg / 'close').mkdir(parents=True, exist_ok=True)
        (reg / 'close' / f'{a.run_id}.json').write_text(json.dumps(failed, indent=1))
        log(tb)
        logf.close()
        raise

    src = out / 'frozen' / cfg['tag'] / a.subset / f's{a.seed}'
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / 'model.pt', out / 'ckpt' / 'best.pt')
    shutil.copyfile(src / 'metrics.json', out / 'metrics.json')
    shutil.copyfile(src / 'history.json', out / 'history.json')
    shutil.copyfile(src / 'config.json', out / 'frozen_config.json')
    pred = pd.read_csv(src / 'val_pred.csv').rename(columns={'unit': 'engine_id'})
    pred['split'] = 'val'
    pred.to_csv(out / 'preds.csv', index=False)

    close = dict(
        run_id=a.run_id, status='smoke' if a.smoke else 'ok', closed_utc=now_utc(),
        written_by='run_v0', main_metric='val_engine_mean_rmse',
        epochs_run=m['epochs_run'], best_epoch=m['best_ep'],
        metrics=dict(val_engine_mean_rmse=m['val_engine_mean_rmse'],
                     val_pooled_rmse=m['val_pooled_rmse'],
                     val_nasa_mean=m.get('nasa_mean'), val_mae=m.get('mae'),
                     val_bias=m.get('bias')),
        trainer='frozen train_one',
        artifacts=['config.json', 'launch.json', 'metrics.json', 'preds.csv',
                   'train.log', 'history.json', 'ckpt/best.pt', 'frozen_config.json'],
        notes=('smoke run: not evidence' if a.smoke else
               'frozen trainer; init_from=none (independent init)'))
    (out / 'close.json').write_text(json.dumps(close, indent=1))
    (reg / 'close').mkdir(parents=True, exist_ok=True)
    (reg / 'close' / f'{a.run_id}.json').write_text(json.dumps(close, indent=1))

    man = dict(
        run_id=a.run_id, variant=a.variant, subset=a.subset, seed=a.seed,
        smoke=bool(a.smoke), baseline_version=baseline_version,
        git_sha=git_sha_field, worktree_dirty=False, val_engine_ids_hash=got,
        trainer='paper_b.training.train.train_one',
        code_md5=code_hashes(), split=dict(val_engine_ids_hash=got),
        env=env_info(),
        config_sha1=hashlib.sha1((out / 'config.json').read_bytes()).hexdigest(),
        artifacts=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()),
        notes='validation only; no test phase in Stage C',
    )
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    log('[v0] artifacts written to', str(out))
    logf.close()


if __name__ == '__main__':
    main()
