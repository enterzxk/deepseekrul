"""Run one Stage C experiment and leave a self-describing artifact directory.

Usage (on a40, after ``source /media/sangfor/vdb/user/zxk/env.sh``)::

    python -m paper_b.hc.run_hc \
        --variant A3a --subset FD001 --seed 101 \
        --run-id HC-C-A3a-FD001-s101-20261111T1402Z \
        --data-dir $HC_DATA --out-root $HC_OUT_ROOT

Writes, under ``<out-root>/<run_id>/``::

    launch.json     exclusive-created before any work starts (B2)
    config.json     effective config (not the template)
    metrics.json    validation metrics; no test_* before the freeze
    preds.csv       per-sample predictions
    history.json    per-epoch loss / engine-mean / pooled validation RMSE
    train.log       captured stdout
    ckpt/best.pt    best-by-validation checkpoint
    close.json      the registry close record
    manifest.json   code md5, split fingerprint, environment, artifact list

Guards, because this project has already paid for missing ones:
  * R6 split drift: the validation engine-id hash is recomputed and compared
    with the frozen reference before a single GPU second is spent.
  * B2 duplicate run: ``launch.json`` is created with O_EXCL, and an output
    directory that already holds artifacts is refused unless ``--force`` is
    given. A silent ``exist_ok=True`` overwrite is exactly how a batch gets
    re-run without anyone noticing.
  * C2 manifest completeness: the manifest is written last, so it lists the
    close record it depends on.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from .config_hc import (FOLD, SPLIT_HASH_REF, SPLIT_SEED, VAL_FRAC,
                        assert_protocol_matches_taskbook, hc_config, split_hash)

REPO = Path(__file__).resolve().parents[2]
FROZEN = ['paper_b/models/dspt.py', 'paper_b/training/data.py',
          'paper_b/training/train.py', 'paper_b/training/metrics.py',
          'paper_b/configs/protocol.json']
OWN = ['paper_b/hc/config_hc.py', 'paper_b/hc/data_hc.py', 'paper_b/hc/model_hc.py',
       'paper_b/hc/cycle_encoder.py', 'paper_b/hc/train_hc.py', 'paper_b/hc/run_hc.py',
       'paper_b/hc/verify_loop.py', 'paper_b/hc/snapshot_val.py',
       'paper_b/hc/compare_val_snapshot.py']
KEEP_DIRS = {'ckpt'}


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def code_hashes():
    """md5 of the files as they exist on THIS host (a40), not the local repo."""
    out = {}
    for rel in FROZEN + OWN:
        p = REPO / rel
        out[rel] = md5(p) if p.exists() else 'MISSING'
    return out


def env_info():
    import torch
    info = dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                host=platform.node())
    try:
        info['git'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        info['git'] = 'WORKTREE (no git on this host)'
    return info


def now_utc():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def exclusive_write(path, payload):
    """B2: O_EXCL create. An existing file means the run id was already used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=1).encode()
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(f'REFUSED: {path} already exists; a run_id is used once. '
                         f'Use a new timestamped run_id.')
    with os.fdopen(fd, 'wb') as fh:
        fh.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--span', type=int, default=None)
    ap.add_argument('--hypothesis', default='')
    ap.add_argument('--gate', default='A')
    ap.add_argument('--data-dir', default=os.environ.get('HC_DATA', 'data/CMAPSS'))
    ap.add_argument('--out-root', default=os.environ.get('HC_OUT_ROOT', 'experiments/hc'))
    ap.add_argument('--registry-root', default=str(REPO / 'docs' / 'paper_b_hc' / 'registry'),
                    help='launch/ and close/ records (task book §11)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--smoke', action='store_true', help='1-epoch pipeline check; not evidence')
    ap.add_argument('--force', action='store_true', help='overwrite an existing run dir')
    a = ap.parse_args()

    assert_protocol_matches_taskbook()

    # R6 drift guard: refuse to spend GPU on a split that is not the frozen one.
    from ..training.data import load_train, split_units
    units = np.sort(load_train(Path(a.data_dir), a.subset)['unit'].unique())
    _, val_units = split_units(units, SPLIT_SEED, FOLD, VAL_FRAC)
    got = split_hash(val_units)
    want = SPLIT_HASH_REF[a.subset]
    if got != want:
        raise SystemExit(f'SPLIT DRIFT on {a.subset}: val_engine_ids_hash={got} '
                         f'!= reference {want}; refusing to run')
    print(f'[hc] split guard ok {a.subset} val={len(val_units)} hash={got}', flush=True)

    cfg = hc_config(a.variant, a.subset, a.seed, span=a.span or 70)
    cfg['data_dir'] = a.data_dir
    cfg['run_id'] = a.run_id
    cfg['val_engine_ids_hash'] = got
    cfg['val_units'] = sorted(int(u) for u in val_units)
    cfg['hypothesis'] = a.hypothesis
    cfg['gate'] = a.gate
    cfg['smoke'] = bool(a.smoke)
    if a.smoke:
        cfg['epochs'], cfg['patience'] = 1, 1
        cfg['budget_override'] = 'smoke: 1 epoch, not evidence'

    out = Path(a.out_root) / a.run_id
    if out.exists() and any(p.name not in KEEP_DIRS for p in out.iterdir()) and not a.force:
        raise SystemExit(f'REFUSED: {out} is not empty; pass --force to overwrite '
                         f'or use a new run_id')

    # B2: the authoritative launch record goes to the registry (task book §11)
    # and is created with O_EXCL, so a run_id can never be launched twice.
    reg = Path(a.registry_root)
    launch = dict(
        run_id=a.run_id, stage=cfg['stage'], variant=a.variant, subset=a.subset,
        seed=a.seed, span=cfg['span'], hypothesis=a.hypothesis or '(not stated)',
        gate=a.gate, host=env_info()['host'], smoke=bool(a.smoke),
        code_md5=code_hashes(), git_sha=env_info()['git'],
        data_split=dict(split_seed=SPLIT_SEED, fold=FOLD, val_frac=VAL_FRAC,
                        val_engine_ids_hash=got),
        budget=dict(epochs=cfg['epochs'], patience=cfg['patience'],
                    budget_override=cfg.get('budget_override'), est_gpu_min=25),
        launched_utc=now_utc())
    exclusive_write(reg / 'launch' / f'{a.run_id}.json', launch)
    out.mkdir(parents=True, exist_ok=True)
    # A copy inside the run directory keeps the artifact directory self-describing.
    (out / 'launch.json').write_text(json.dumps(launch, indent=1))
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))

    logf = open(out / 'train.log', 'w', encoding='utf-8')

    def log(*args):
        msg = ' '.join(str(x) for x in args)
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    from . import data_hc as _d
    from .train_hc import train_hc

    log(f'[hc] run_id={a.run_id} variant={a.variant} subset={a.subset} seed={a.seed}')
    log(f'[hc] smoke={bool(a.smoke)} backbone={cfg["backbone"]}')
    if a.smoke:
        log('[hc] BUDGET OVERRIDE: 1 epoch; this run is not evidence')
    m, ds = train_hc(cfg, device=a.device, out_dir=out, log=log)

    close = dict(
        run_id=a.run_id, status='smoke' if a.smoke else 'ok', closed_utc=now_utc(),
        epochs_run=m['epochs_run'], best_epoch=m['best_ep'],
        metrics=dict(val_engine_mean_rmse=m['val_engine_mean_rmse'],
                     val_pooled_rmse=m['val_pooled_rmse'],
                     val_nasa_mean=m.get('nasa_mean'), val_mae=m.get('mae'),
                     val_bias=m.get('bias')),
        main_metric='val_engine_mean_rmse',
        artifacts=['config.json', 'manifest.json', 'metrics.json', 'preds.csv',
                   'train.log', 'history.json', 'ckpt/best.pt', 'launch.json'],
        notes=('smoke run: not evidence' if a.smoke else
               'intervention mean|d|=%.3f' % m['cycle_intervention']['mean_abs_delta']))
    (out / 'close.json').write_text(json.dumps(close, indent=1))
    (reg / 'close').mkdir(parents=True, exist_ok=True)
    (reg / 'close' / f'{a.run_id}.json').write_text(json.dumps(close, indent=1))

    # C2: manifest last, so it lists every artifact including close.json.
    man = dict(
        run_id=a.run_id, variant=a.variant, subset=a.subset, seed=a.seed,
        smoke=bool(a.smoke),
        val_engine_ids_hash=got,
        code_md5=code_hashes(),
        split=_d.split_fingerprint(ds),
        env=env_info(),
        config_sha1=hashlib.sha1((out / 'config.json').read_bytes()).hexdigest(),
        artifacts=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()),
        notes='validation only; no test phase in Stage C',
    )
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    log('[hc] artifacts written to', str(out))
    logf.close()


if __name__ == '__main__':
    main()
