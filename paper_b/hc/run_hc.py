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
       'paper_b/hc/compare_val_snapshot.py', 'paper_b/hc/code_lock.py',
       # v10: the G2 init-RMS calibration lives here, so it must be locked too
       # (it decides the scale of the cycle term for the calibrated arms).
       'paper_b/hc/scale_align.py']
KEEP_DIRS = {'ckpt'}


def md5(path):
    """md5 of the file as the repo stores it (LF), never the working-tree CRLF.

    Same definition as ``code_lock.check`` and as the lock's own ``definition``
    field. The two validation paths used to disagree: ``check`` normalised CRLF
    while this startup guard hashed raw bytes, so a CRLF sync to a git-less host
    passed ``--check`` and then refused to launch. That is now one definition.
    """
    return hashlib.md5(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


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


# Code a run actually imports. ``paper_b/experiments`` is deliberately absent:
# probe scripts are not imported by training, and that tree is shared with the
# parallel s23 line whose uncommitted artifacts must not block hc runs.
GUARD_PATHS = ['paper_b/hc', 'paper_b/training', 'paper_b/models', 'paper_b/configs']


def _git_status(paths, untracked):
    args = ['git', 'status', '--porcelain']
    if not untracked:
        args.append('--untracked-files=no')
    args += ['--'] + list(paths)
    return subprocess.check_output(args, cwd=REPO).decode().strip()


def worktree_guard():
    """Refuse to launch from an uncommitted tree (2026-11-11 ruling, step 1).

    Returns ``{'mode': 'git', 'git_sha': ...}`` on a host with git, or ``None``
    when git is absent (a40), in which case ``code_lock_guard`` takes over.
    ``paper_b/hc`` is checked with untracked files included (a new module is
    code); the other paths are tracked-only so result/artifact files written by
    parallel lines cannot block a run.
    """
    try:
        inside = subprocess.check_output(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None
    if inside != 'true':
        return None
    dirty = _git_status(['paper_b/hc'], True)
    dirty += ('\n' if dirty else '') + _git_status(
        ['paper_b/training', 'paper_b/models', 'paper_b/configs'], False)
    if dirty.strip():
        raise SystemExit('REFUSED: uncommitted code under paper_b/; commit first:\n' + dirty)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    return dict(mode='git', git_sha=sha, code_lock_sha1=None)


def code_lock_guard(path):
    """No-git equivalent: every hashed file must equal the committed lock."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f'REFUSED: no git on this host and no code lock at {path}; '
                         f'generate it on the repo with '
                         f'`python -m paper_b.hc.code_lock --write` and sync it')
    lock = json.loads(path.read_text())
    got = code_hashes()
    bad = {k: (got.get(k), v) for k, v in lock['code_md5'].items() if got.get(k) != v}
    if bad:
        raise SystemExit('REFUSED: code differs from the committed lock:\n'
                         + json.dumps(bad, indent=1))
    return dict(mode='lock', git_sha=(lock.get('git_sha') or 'WORKTREE'),
                code_lock_sha1=hashlib.sha1(path.read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--span', type=int, default=None)
    ap.add_argument('--hypothesis', required=True,
                    help='one-line falsifiable statement; required by the registry spec')
    ap.add_argument('--gate', default='A')
    ap.add_argument('--data-dir', default=os.environ.get('HC_DATA', 'data/CMAPSS'))
    ap.add_argument('--out-root', default=os.environ.get('HC_OUT_ROOT', 'experiments/hc'))
    ap.add_argument('--registry-root', default=str(REPO / 'docs' / 'paper_b_hc' / 'registry'),
                    help='launch/ and close/ records (task book §11)')
    ap.add_argument('--code-lock', default=str(REPO / 'docs' / 'paper_b_hc' / 'registry' /
                                               'code_lock.json'),
                    help='committed-code lock, used when the host has no git (a40)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--init-from', default=None,
                    help='state_dict (or dump payload) loaded before training')
    ap.add_argument('--dump-init', default=None,
                    help='write the freshly initialised weights here before training')
    ap.add_argument('--smoke', action='store_true', help='1-epoch pipeline check; not evidence')
    ap.add_argument('--baseline-version', default=None,
                    help='registered baseline label; defaults to the code lock value')
    ap.add_argument('--force', action='store_true', help='overwrite an existing run dir')
    a = ap.parse_args()

    assert_protocol_matches_taskbook()

    # The dump / load of the shared initialisation happens inside train_hc, so
    # the run that writes the dump and the run that consumes it are the same
    # kind of process and both record the same sha256.

    # Step 1 of the 2026-11-11 ruling: a run may only start from a committed
    # tree. Git hosts use `git status`; a40 has no git, so it verifies every
    # hashed file against the lock generated from `git show HEAD:` blobs.
    guard = worktree_guard()
    if guard is None:
        guard = code_lock_guard(a.code_lock)
        print(f'[hc] code lock ok git_sha={guard["git_sha"][:12]} '
              f'lock_sha1={guard["code_lock_sha1"][:12]}', flush=True)
    else:
        print(f'[hc] worktree clean at {guard["git_sha"][:12]}', flush=True)

    # Fingerprint triple required by the registry spec (2026-11-11): a run must
    # say which baseline it belongs to, and a dirty tree must never be mistaken
    # for evidence. Both guard branches above refuse to start when dirty, so the
    # recorded flag is False by construction -- and provably so, because the
    # process cannot reach this line otherwise.
    lock = {}
    if Path(a.code_lock).exists():
        lock = json.loads(Path(a.code_lock).read_text())
    baseline_version = a.baseline_version or lock.get('baseline_version', 'unregistered')
    git_sha_field = guard['git_sha'] if guard['mode'] == 'git' else 'WORKTREE'
    worktree_dirty = False

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
    cfg['init_from'] = a.init_from
    cfg['provenance'] = dict(code_md5=code_hashes(), git_sha=git_sha_field,
                             baseline_version=baseline_version)
    cfg['dump_init'] = a.dump_init
    if a.init_from:
        cfg['init_from_sha256'] = hashlib.sha256(Path(a.init_from).read_bytes()).hexdigest()

    if a.dump_init:
        # Export-only mode (2026-11-11 ruling): no launch, no close, no training.
        # The artefact is the only output; the training runs that consume it get
        # their own launch/close records.
        from .train_hc import dump_init as _dump
        _dump(cfg)
        print('[hc] init dump written; exiting without training', flush=True)
        return
    if a.smoke:
        cfg['epochs'], cfg['patience'] = 1, 1
        cfg['budget_override'] = 'smoke: 1 epoch, not evidence'

    out = Path(a.out_root) / a.run_id
    if out.exists() and any(p.name not in KEEP_DIRS for p in out.iterdir()) and not a.force:
        raise SystemExit(f'REFUSED: {out} is not empty; pass --force to overwrite '
                         f'or use a new run_id')

    # B2 + P0: the launch record is machine-generated at process start, created
    # with O_EXCL, then read back and compared. Hand-written or copied launch
    # files are what broke the evidence chain on the first two smoke runs, so the
    # record now carries written_by / process_start_utc and self-verifies.
    proc_start = now_utc()
    reg = Path(a.registry_root)
    hashes = code_hashes()
    launch = dict(
        run_id=a.run_id, stage=cfg['stage'], variant=a.variant, subset=a.subset,
        seed=a.seed, span=cfg['span'], hypothesis=a.hypothesis or '(not stated)',
        gate=a.gate, host=env_info()['host'], smoke=bool(a.smoke),
        written_by='run_hc', process_start_utc=proc_start,
        code_md5=hashes, git_sha=git_sha_field, worktree_dirty=worktree_dirty,
        baseline_version=baseline_version, worktree=guard,
        code_lock_git_sha=(lock.get('git_sha') if guard['mode'] == 'lock' else None),
        data_split=dict(split_seed=SPLIT_SEED, fold=FOLD, val_frac=VAL_FRAC,
                        val_engine_ids_hash=got),
        budget=dict(epochs=cfg['epochs'], patience=cfg['patience'],
                    budget_override=cfg.get('budget_override'), est_gpu_min=25),
        launched_utc=now_utc())
    launch_path = reg / 'launch' / f'{a.run_id}.json'
    exclusive_write(launch_path, launch)
    # Read-back self-check: what is on disk must equal what the process computed.
    back = json.loads(launch_path.read_text())
    if back.get('code_md5') != code_hashes() or back.get('written_by') != 'run_hc':
        raise SystemExit(f'SELF-CHECK FAILED: {launch_path} does not match the code '
                         f'this process is running')
    out.mkdir(parents=True, exist_ok=True)
    # A copy inside the run directory keeps the artifact directory self-describing.
    (out / 'launch.json').write_text(json.dumps(back, indent=1))
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

    # P2-1: a crash must still leave a close record. Without this, a batch of 24
    # runs leaves launch-without-close entries and O_EXCL then blocks those
    # run_ids forever (that is exactly what happened to T1431Z).
    try:
        m, ds = train_hc(cfg, device=a.device, out_dir=out, log=log)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        failed = dict(
            run_id=a.run_id, status='failed', closed_utc=now_utc(),
            epochs_run=0, best_epoch=None, metrics={},
            main_metric='val_engine_mean_rmse', written_by='run_hc',
            artifacts=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()),
            notes=(f'FAILED {type(exc).__name__}: {exc}'[:400]
                   + ' | last traceback line: ' + tb.strip().splitlines()[-1][:200]))
        (out / 'close.json').write_text(json.dumps(failed, indent=1))
        (reg / 'close').mkdir(parents=True, exist_ok=True)
        (reg / 'close' / f'{a.run_id}.json').write_text(json.dumps(failed, indent=1))
        log(f'[hc] FAILED {type(exc).__name__}: {exc}')
        log(tb)
        logf.close()
        raise

    if m is None:
        log('[hc] dump-only run: initial weights written, nothing trained')
        logf.close()
        return

    close = dict(
        run_id=a.run_id, status='smoke' if a.smoke else 'ok', closed_utc=now_utc(),
        epochs_run=m['epochs_run'], best_epoch=m['best_ep'],
        metrics=dict(val_engine_mean_rmse=m['val_engine_mean_rmse'],
                     val_pooled_rmse=m['val_pooled_rmse'],
                     val_nasa_mean=m.get('nasa_mean'), val_mae=m.get('mae'),
                     val_bias=m.get('bias')),
        main_metric='val_engine_mean_rmse', written_by='run_hc',
        artifacts=(['config.json', 'manifest.json', 'metrics.json', 'preds.csv',
                    'train.log', 'history.json', 'ckpt/best.pt', 'launch.json']
                   + (['calibration.json'] if (out / 'calibration.json').exists() else [])),
        notes=('smoke run: not evidence' if a.smoke else
               'intervention mean|d|=%.3f; init_from=%s; init_shared=%s'
               % (m['cycle_intervention']['mean_abs_delta'],
                  cfg.get('init_from') or 'none',
                  'yes' if cfg.get('init_from') else 'no (independent init)')))
    (out / 'close.json').write_text(json.dumps(close, indent=1))
    (reg / 'close').mkdir(parents=True, exist_ok=True)
    (reg / 'close' / f'{a.run_id}.json').write_text(json.dumps(close, indent=1))

    # C2: manifest last, so it lists every artifact including close.json.
    # provenance (task book §1.2 / test-lead): baseline_version pins the blob set
    # the run is compared against, worktree_dirty records whether the tree moved
    # away from that baseline at launch time.
    provenance = launch.get('provenance', {})
    man = dict(
        run_id=a.run_id, variant=a.variant, subset=a.subset, seed=a.seed,
        smoke=bool(a.smoke),
        baseline_version=baseline_version,
        worktree_dirty=worktree_dirty,
        git_sha=git_sha_field,
        code_lock_git_sha=(lock.get('git_sha') if guard['mode'] == 'lock' else None),
        val_engine_ids_hash=got,
        init_sha256=cfg.get('init_sha256') or cfg.get('init_from_sha256'),
        code_md5=code_hashes(),
        split=_d.split_fingerprint(ds),
        env=env_info(),
        config_sha1=hashlib.sha1((out / 'config.json').read_bytes()).hexdigest(),
        artifacts=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
        + ['manifest.json'],
        notes='validation only; no test phase in Stage C',
    )
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    log('[hc] artifacts written to', str(out))
    logf.close()


if __name__ == '__main__':
    main()
