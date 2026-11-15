#!/usr/bin/env python3
"""Queue driver for the Paper B sensor-generalization study (E0-E2).

Roles
  build   freeze the base config, the training-side noise statistics and the
          fixed missing-sensor patterns, then write the run manifests.
  worker  one process per physical GPU. It pulls the next unclaimed run from
          the shared manifest, waits until its card has no other training
          process, claims the run atomically, trains it, evaluates it on the
          clean / fixed-missing / fixed-noise development protocols and writes
          the run directory.
  status  print the per-run state.

No run is ever started twice: a run id is derived from the resolved config and
the code fingerprint, a COMPLETE status file makes it ineligible, and the claim
file is created with O_EXCL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
STUDY_ID = 'paper_b_sensor_generalization'
STUDY_DIR = REPO / 'experiments' / STUDY_ID
SHARED = STUDY_DIR / 'shared'
RUNS = STUDY_DIR / 'runs'
CLAIMS = STUDY_DIR / 'claims'
LOCKS = STUDY_DIR / 'locks'
MANIFESTS = STUDY_DIR / 'manifests'
LOGS = STUDY_DIR / 'logs'

BASE = dict(
    model='dspt', span=70, epochs=40, patience=12, batch=512, lr=1e-3,
    weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, mode='dual',
    ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125,
    split_seed=17, fold=0, val_frac=0.2, n_cond=6,
    cuts=[0.4, 0.55, 0.7, 0.85, 0.95],
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
)

# E0-E2 only. E3 (mask + noise combined) is removed in this study: both single
# channels were below their pre-registered thresholds in
# research/dspt-sensor-aug-20261114, so the combination has no hypothesis left.
# Old-study mapping kept for traceability: E1a/E1b == G1/G2 (mask, p=0.10/0.20),
# E2a/E2b == G3/G4 (noise, alpha=0.25/0.50). The old report's "+15.3% on FD001"
# noise figure is E1b evaluated UNDER the noise-stress protocol, not E2.
GROUPS = {
    'E0':  dict(mask_prob=0.0,  noise_alpha=0.0),
    'E1a': dict(mask_prob=0.10, noise_alpha=0.0),
    'E1b': dict(mask_prob=0.20, noise_alpha=0.0),
    'E2a': dict(mask_prob=0.0,  noise_alpha=0.25),
    'E2b': dict(mask_prob=0.0,  noise_alpha=0.50),
}

STAGE1_SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
STAGE1_SEEDS = [101, 202, 303]
STAGE2_SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
STAGE2_SEEDS = [101, 202, 303, 404, 505]


# --------------------------------------------------------------------- utils
def code_fingerprint():
    h = hashlib.sha256()
    for rel in ('paper_b/models/dspt.py', 'paper_b/training/train.py',
                'paper_b/training/data.py', 'paper_b/training/metrics.py',
                'paper_b/sensor_aug/augmentation.py', 'paper_b/sensor_aug/noise_stats.py',
                'paper_b/sensor_aug/evaluation.py'):
        h.update((REPO / rel).read_bytes())
    try:
        head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = ''
    return h.hexdigest()[:12], head


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def run_id_for(group, subset, seed, cfg):
    return f'{group}-{subset}-s{seed}-{config_hash(cfg)}'


def task_config(group, subset, seed, data_dir):
    g = GROUPS[group]
    if g['mask_prob'] > 0 and g['noise_alpha'] > 0:
        raise ValueError('E3 (mask + noise combined) is out of scope for this study')
    cfg = dict(BASE)
    cfg.update(subset=subset, seed=seed, data_dir=str(data_dir), tag='',
               study_id=STUDY_ID)
    if g['mask_prob'] > 0 or g['noise_alpha'] > 0:
        cfg['augment'] = dict(
            mask_prob=g['mask_prob'],
            noise_alpha=g['noise_alpha'],
            noise_stats_path=(str(SHARED / f'noise_stats_{subset}.json')
                              if g['noise_alpha'] > 0 else ''),
            mask_hash='',
        )
    else:
        cfg['augment'] = None
    return cfg


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def gpu_mem(idx):
    out = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-gpu=memory.used,memory.total',
         '--format=csv,noheader,nounits'], text=True)
    used, total = [int(x) for x in out.strip().split(',')]
    return used, total


def gpu_uuid_of(idx):
    out = subprocess.check_output(
        ['nvidia-smi', '-i', str(idx), '--query-gpu=uuid', '--format=csv,noheader'],
        text=True)
    return out.strip()


def foreign_gpu_mem(idx, own_pid):
    """Memory used on GPU idx by processes other than this worker.

    A finished run leaves several GiB in the PyTorch caching allocator, so the
    raw 'memory.used' would make a worker wait on itself forever. Only foreign
    processes are counted as occupancy.
    """
    try:
        uuid = gpu_uuid_of(idx)
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
             '--format=csv,noheader,nounits'], text=True)
    except Exception:
        return None
    total = 0
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) != 3:
            continue
        pid, gu, mem = parts
        if gu == uuid and pid != str(own_pid):
            total += int(float(mem))
    return total


def wait_for_gpu(idx, need_free_mib=20000, poll=60, log=print):
    while True:
        try:
            used, total = gpu_mem(idx)
            foreign = foreign_gpu_mem(idx, os.getpid())
        except Exception as exc:
            log(f'[gpu{idx}] nvidia-smi failed: {exc}; retrying', flush=True)
            time.sleep(poll)
            continue
        if foreign is None:
            foreign = used
        if total - foreign >= need_free_mib:
            return used, total
        log(f'[gpu{idx}] busy: foreign={foreign} total_used={used}/{total} MiB; '
            f'waiting {poll}s', flush=True)
        time.sleep(poll)


# --------------------------------------------------------------------- build
def build(args):
    from paper_b.training.data import _prepare_split
    from paper_b.sensor_aug.noise_stats import (compute_noise_stats, effective_scale,
                                                save_stats)
    from paper_b.sensor_aug.evaluation import (make_missing_patterns, patterns_hash)

    for d in (SHARED, RUNS, CLAIMS, LOCKS, MANIFESTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    code_hash, head = code_fingerprint()
    subsets = args.subsets.split(',')
    manifest = []
    shared_meta = {'study_id': STUDY_ID, 'code_fingerprint': code_hash,
                   'git_head': head, 'data_dir': str(data_dir),
                   'base': BASE, 'groups': GROUPS, 'built': time.time()}

    for subset in subsets:
        s = _prepare_split(subset, data_dir, BASE['split_seed'], BASE['fold'],
                           BASE['val_frac'], BASE['n_cond'])
        stats = compute_noise_stats(s['tr'], s['tr_s'], s['lt'], n_cond=BASE['n_cond'])
        stats.update(subset=subset, split_seed=BASE['split_seed'], fold=BASE['fold'],
                     val_frac=BASE['val_frac'], train_units=[int(x) for x in s['train_units']],
                     val_units=[int(x) for x in s['val_units']])
        stats['alpha_scales'] = {f'{a:g}': effective_scale(stats, a)
                                 for a in (0.25, 0.50)}
        save_stats(stats, SHARED / f'noise_stats_{subset}.json')
        # Development masks are frozen now. The official-test masks are drawn
        # only after the test evaluation is frozen (the test files stay sealed
        # until then); the same seed makes them reproducible.
        pat = make_missing_patterns(s['val_units'])
        (SHARED / f'missing_patterns_val_{subset}.json').write_text(json.dumps(
            {'subset': subset, 'split': 'val', 'seed': 20261114,
             'n_patterns': 10, 'hash': patterns_hash(pat), 'patterns': pat}, indent=1))
        shared_meta.setdefault('noise_stats', {})[subset] = {
            'hash': stats['hash'], 'n_global': stats['n_global'],
            'n_none': stats['n_none'],
            'scales': {k: {kk: vv for kk, vv in v.items() if kk != 's'}
                       for k, v in stats['alpha_scales'].items()},
        }
        print(f'[build] {subset}: noise sigma hash {stats["hash"]} '
              f'global={stats["n_global"]} none={stats["n_none"]} '
              f'cap25={stats["alpha_scales"]["0.25"]["cap_fraction"]:.3f} '
              f'cap50={stats["alpha_scales"]["0.5"]["cap_fraction"]:.3f}', flush=True)

    seeds = [int(x) for x in args.seeds.split(',')]
    groups = args.groups.split(',')
    for seed in seeds:
        for group in groups:
            for subset in subsets:
                cfg = task_config(group, subset, seed, data_dir)
                if cfg['augment'] is not None and cfg['augment']['noise_alpha'] > 0:
                    st = json.loads((SHARED / f'noise_stats_{subset}.json').read_text())
                    cfg['augment']['noise_stats_hash'] = st['hash']
                if cfg['augment'] is not None and cfg['augment']['mask_prob'] > 0:
                    mp = json.loads((SHARED / f'missing_patterns_val_{subset}.json').read_text())
                    cfg['augment']['mask_hash'] = mp['hash']
                rid = run_id_for(group, subset, seed, cfg)
                cfg['tag'] = rid
                cfg['run_id'] = rid
                manifest.append(dict(run_id=rid, group=group, subset=subset, seed=seed,
                                     tag=rid, config=cfg))
    # frozen config files (plan section 7.1)
    import yaml
    cfg_dir = REPO / 'paper_b' / 'sensor_aug' / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'frozen_base.yaml').write_text(yaml.safe_dump(BASE, sort_keys=False))
    for g, params in GROUPS.items():
        c = dict(BASE)
        c['group'] = g
        c['augment'] = dict(mask_prob=params['mask_prob'], noise_alpha=params['noise_alpha'],
                            estimation='same_condition_first_difference_mad',
                            minimum_pairs=200, minimum_engines=5,
                            max_std_relative_to_robust_scale=0.20,
                            location='before_features', baseline_policy='fixed_clean_prefix')
        (cfg_dir / f'{g}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    out = MANIFESTS / args.name
    out.write_text(json.dumps({'study_id': STUDY_ID, 'code_fingerprint': code_hash,
                               'git_head': head, 'runs': manifest}, indent=1))
    (SHARED / 'study_meta.json').write_text(json.dumps(shared_meta, indent=1))
    print(f'[build] wrote {out} with {len(manifest)} runs', flush=True)


# -------------------------------------------------------------------- worker
def run_dir_of(task):
    return RUNS / task['run_id'] / task['subset'] / f"s{task['seed']}"


def is_complete(task):
    st = run_dir_of(task) / 'status.json'
    if not st.exists():
        return False
    try:
        return json.loads(st.read_text()).get('status') == 'COMPLETE'
    except Exception:
        return False


def claim(task, gpu):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    p = CLAIMS / f"{task['run_id']}.claim"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            info = json.loads(p.read_text())
        except Exception:
            info = {}
        if not pid_alive(info.get('pid', -1)):
            p.unlink(missing_ok=True)
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
        else:
            return False
    os.write(fd, json.dumps({'pid': os.getpid(), 'gpu': gpu,
                             'time': time.time()}).encode())
    os.close(fd)
    return True


def next_task(manifest):
    for task in manifest:
        if is_complete(task):
            continue
        p = CLAIMS / f"{task['run_id']}.claim"
        if p.exists():
            try:
                info = json.loads(p.read_text())
            except Exception:
                info = {}
            if pid_alive(info.get('pid', -1)):
                continue
            p.unlink(missing_ok=True)
        return task
    return None


def evaluate_run(task, data_dir, device):
    import torch
    from paper_b.training.data import prepare_train
    from paper_b.training.train import _build
    from paper_b.sensor_aug.evaluation import (evaluate_clean, evaluate_robustness,
                                               make_noise_fields, engine_lengths,
                                               noise_fields_hash, benchmark_inference)

    cfg = task['config']
    rd = run_dir_of(task)
    dev = torch.device(device)
    ds = prepare_train(cfg['subset'], Path(data_dir), span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
    va = ds['val']
    net = _build(cfg).to(dev)
    net.load_state_dict(torch.load(rd / 'best.pt', map_location='cpu', weights_only=True))
    best_ep = json.loads((rd / 'metrics.json').read_text())['best_ep']
    mp = json.loads((SHARED / f"missing_patterns_val_{cfg['subset']}.json").read_text())
    st = json.loads((SHARED / f"noise_stats_{cfg['subset']}.json").read_text())
    lengths = engine_lengths(va)
    fields = make_noise_fields(va['unit'], lengths)
    meta = dict(study_id=STUDY_ID, run_id=task['run_id'], group=task['group'],
                dataset=cfg['subset'], split_seed=cfg['split_seed'],
                model_seed=cfg['seed'], mask_hash=mp['hash'],
                noise_draw_id=noise_fields_hash(fields), checkpoint_id='best.pt',
                best_epoch=best_ep, noise_fields=fields)
    m, _ = evaluate_clean(net, va, dev, cfg, meta, rd)
    rob = evaluate_robustness(net, va, dev, cfg, meta, mp['patterns'],
                              np.asarray(st['sigma'], np.float64), rd)
    out = dict(clean=m, missing=rob['missing'], noise=rob['noise'],
               inference=benchmark_inference(net, va, dev, cfg),
               missing_patterns_hash=mp['hash'], noise_fields_hash=meta['noise_draw_id'],
               noise_stats_hash=st['hash'])
    (rd / 'robustness_val.json').write_text(json.dumps(out, indent=1))
    return m, out


def worker(args):
    manifest = json.loads(Path(args.manifest).read_text())['runs']
    print(f'[worker gpu{args.gpu}] {len(manifest)} runs in manifest', flush=True)
    while True:
        task = next_task(manifest)
        if task is None:
            print(f'[worker gpu{args.gpu}] queue empty', flush=True)
            return
        wait_for_gpu(args.gpu, need_free_mib=args.need_free, poll=args.poll)
        # Re-select after the wait: another worker may have finished or claimed
        # the task we were holding while this card was busy.
        task = next_task(manifest)
        if task is None:
            print(f'[worker gpu{args.gpu}] queue empty after wait', flush=True)
            return
        if not claim(task, args.gpu):
            continue
        rd = run_dir_of(task)
        rd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f'[worker gpu{args.gpu}] START {task["run_id"]} ({task["group"]} '
              f'{task["subset"]} s{task["seed"]})', flush=True)
        trained = (rd / 'train_complete.json').exists() and (rd / 'best.pt').exists()
        try:
            from paper_b.training.train import train_one
            cfg = dict(task['config'])
            cfg['data_dir'] = str(args.data_dir)
            if trained:
                print(f'[worker gpu{args.gpu}] resume evaluation only for '
                      f'{task["run_id"]}', flush=True)
            else:
                train_one(cfg, device=f'cuda:{args.gpu}', out_root=str(RUNS))
                (rd / 'train_complete.json').write_text(json.dumps(
                    {'time': time.time(), 'elapsed_s': time.time() - t0}, indent=1))
            if not args.no_eval:
                evaluate_run(task, args.data_dir, f'cuda:{args.gpu}')
            print(f'[worker gpu{args.gpu}] DONE {task["run_id"]} in '
                  f'{time.time() - t0:.0f}s', flush=True)
        except Exception as exc:
            (rd / 'error.txt').write_text(traceback.format_exc())
            status = 'EVAL_FAILED' if trained else 'FAILED'
            (rd / 'status.json').write_text(json.dumps(
                {'status': status, 'error': repr(exc), 'time': time.time()}, indent=1))
            print(f'[worker gpu{args.gpu}] FAIL {task["run_id"]} ({status}): {exc}', flush=True)
        finally:
            (CLAIMS / f"{task['run_id']}.claim").unlink(missing_ok=True)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass



# -------------------------------------------------------------------- freeze
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze(args):
    man = json.loads(Path(args.manifest).read_text())
    screen = json.loads(Path(args.screen).read_text())
    cand = screen.get('clean_candidate') or screen.get('robust_candidate')
    if not cand:
        raise SystemExit('no candidate: nothing to freeze (NO_GAIN)')
    runs = [t for t in man['runs'] if t['group'] in ('E0', cand)]
    rows = []
    for t in runs:
        rd = run_dir_of(t)
        if not is_complete(t):
            raise SystemExit(f'run not COMPLETE: {t["run_id"]}')
        rows.append(dict(group=t['group'], subset=t['subset'], seed=int(t['seed']),
                         run_id=t['run_id'], best_pt_sha256=sha256_file(rd / 'best.pt')))
    code_hash, head = code_fingerprint()
    out = dict(study_id=STUDY_ID, frozen=True, time=time.time(),
               candidate=dict(group=cand,
                              mask_prob=GROUPS[cand]['mask_prob'],
                              noise_alpha=GROUPS[cand]['noise_alpha']),
               channel=screen.get('channel'),
               robustness_target=screen.get('robustness_target'),
               screen=screen, runs=rows, code_fingerprint=code_hash, git_head=head,
               test_rules=dict(metrics=['pooled_rmse', 'engine_equal_rmse', 'mae',
                                        'nasa_mean', 'nasa_sum'],
                               missing_ks=[0, 1, 2, 3, 4],
                               noise_betas=[0.0, 0.25, 0.50, 1.00],
                               single_sensor=True, opened_once=True))
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=1))
    print(f'[freeze] candidate={cand} channel={screen.get("channel")} '
          f'runs={len(rows)} -> {path}', flush=True)


# ---------------------------------------------------------------------- test
def evaluate_test_run(task, data_dir, device):
    import torch
    from paper_b.training.data import prepare_test
    from paper_b.training.train import _build
    from paper_b.sensor_aug.evaluation import (evaluate_clean, evaluate_robustness,
                                               make_noise_fields, engine_lengths,
                                               noise_fields_hash, make_missing_patterns,
                                               patterns_hash)

    cfg = task['config']
    rd = run_dir_of(task)
    dev = torch.device(device)
    ds = prepare_test(cfg['subset'], Path(data_dir), span=cfg['span'],
                      split_seed=cfg['split_seed'], fold=cfg['fold'],
                      val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'], n_cond=cfg['n_cond'])
    te = ds['test']
    net = _build(cfg).to(dev)
    net.load_state_dict(torch.load(rd / 'best.pt', map_location='cpu', weights_only=True))
    best_ep = json.loads((rd / 'metrics.json').read_text())['best_ep']
    pat = make_missing_patterns(te['unit'])
    st = json.loads((SHARED / f"noise_stats_{cfg['subset']}.json").read_text())
    fields = make_noise_fields(te['unit'], engine_lengths(te))
    meta = dict(study_id=STUDY_ID, run_id=task['run_id'], group=task['group'],
                dataset=cfg['subset'], split_seed=cfg['split_seed'],
                model_seed=cfg['seed'], mask_hash=patterns_hash(pat),
                noise_draw_id=noise_fields_hash(fields), checkpoint_id='best.pt',
                best_epoch=best_ep, noise_fields=fields)
    m, _ = evaluate_clean(net, te, dev, cfg, meta, rd, split='test')
    rob = evaluate_robustness(net, te, dev, cfg, meta, pat,
                              np.asarray(st['sigma'], np.float64), rd,
                              split='test', single_sensor=True)
    out = dict(clean=m, missing=rob['missing'], noise=rob['noise'],
               single_sensor=rob.get('single_sensor'),
               missing_patterns_hash=meta['mask_hash'],
               noise_fields_hash=meta['noise_draw_id'], noise_stats_hash=st['hash'])
    (rd / 'test_metrics.json').write_text(json.dumps(m, indent=1))
    (rd / 'robustness_test.json').write_text(json.dumps(out, indent=1))
    return m


def test_role(args):
    man = json.loads(Path(args.manifest).read_text())['runs']
    fz = json.loads(Path(args.freeze).read_text())
    if not fz.get('frozen'):
        raise SystemExit('freeze manifest is not frozen; test evaluation blocked')
    allowed = {(r['group'], r['subset'], int(r['seed'])) for r in fz['runs']}
    tasks = [t for t in man if (t['group'], t['subset'], t['seed']) in allowed]
    tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
    print(f'[test shard {args.shard}/{args.nshards}] {len(tasks)} runs on gpu{args.gpu}',
          flush=True)
    for t in tasks:
        rd = run_dir_of(t)
        if (rd / 'test_metrics.json').exists() and not args.force:
            print(f'[test] skip {t["run_id"]} (already evaluated)', flush=True)
            continue
        wait_for_gpu(args.gpu, need_free_mib=args.need_free, poll=args.poll)
        m = evaluate_test_run(t, args.data_dir, f'cuda:{args.gpu}')
        print(f'[test] {t["run_id"]} pooled={m["pooled_rmse"]:.4f} '
              f'nasa_sum={m["pooled_nasa_mean"] * m["n"]:.1f}', flush=True)

# -------------------------------------------------------------------- status
def status(args):
    manifest = json.loads(Path(args.manifest).read_text())['runs']
    counts = {}
    for task in manifest:
        rd = run_dir_of(task)
        st = 'PLANNED'
        if (rd / 'status.json').exists():
            try:
                st = json.loads((rd / 'status.json').read_text()).get('status', '?')
            except Exception:
                st = '?'
        counts[st] = counts.get(st, 0) + 1
        if args.verbose:
            print(f"{task['run_id']:42s} {st}")
    print(json.dumps(counts, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='role', required=True)
    b = sub.add_parser('build')
    b.add_argument('--data-dir', required=True)
    b.add_argument('--subsets', default='FD001,FD004')
    b.add_argument('--seeds', default='101,202,303')
    b.add_argument('--groups', default='E0,E1a,E1b,E2a,E2b')
    b.add_argument('--name', default='stage1.json')
    b.set_defaults(func=build)
    w = sub.add_parser('worker')
    w.add_argument('--manifest', required=True)
    w.add_argument('--gpu', type=int, required=True)
    w.add_argument('--data-dir', required=True)
    w.add_argument('--need-free', type=int, default=20000)
    w.add_argument('--poll', type=int, default=60)
    w.add_argument('--no-eval', action='store_true')
    w.set_defaults(func=worker)
    s = sub.add_parser('status')
    s.add_argument('--manifest', required=True)
    s.add_argument('--verbose', action='store_true')
    s.set_defaults(func=status)
    fz = sub.add_parser('freeze')
    fz.add_argument('--manifest', required=True)
    fz.add_argument('--screen', required=True)
    fz.add_argument('--out', required=True)
    fz.set_defaults(func=freeze)
    te = sub.add_parser('test')
    te.add_argument('--manifest', required=True)
    te.add_argument('--freeze', required=True)
    te.add_argument('--gpu', type=int, required=True)
    te.add_argument('--shard', type=int, default=0)
    te.add_argument('--nshards', type=int, default=1)
    te.add_argument('--data-dir', required=True)
    te.add_argument('--need-free', type=int, default=20000)
    te.add_argument('--poll', type=int, default=60)
    te.add_argument('--force', action='store_true')
    te.set_defaults(func=test_role)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
