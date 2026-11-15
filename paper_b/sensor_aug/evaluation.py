"""Evaluation: clean, fixed-mask missing, fixed-noise stress (plan sections 10-11).

Everything here is inference only. The clean development evaluation is the
primary comparison; the corruption evaluations reuse the SAME checkpoint and
never feed back into training or selection.

Fixed masks (plan 11.1): for every engine, 10 fixed sensor permutations are
drawn once from a study seed that is independent of the training seeds. For a
given k the first k entries of a permutation are hidden, so k=1..4 are nested
within an engine. Every model and every seed sees exactly the same patterns;
the patterns and their hash are written before any result is read.

Fixed noise fields (plan 11.3): five independent standard-normal fields per
engine, drawn on the engine timeline and then cut into windows, so the same
(engine, cycle) has the same perturbation in every overlapping window. The
training-side frozen sigma is used WITHOUT the 20% training cap, at
beta in {0, 0.25, 0.50, 1.00}; the baseline stays the clean prefix estimate.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..training.metrics import (engine_equal_mae, engine_equal_nasa_mean,
                                engine_equal_rmse, summarize)

N_SENSORS = 14
MASK_PATTERN_SEED = 20261114
NOISE_FIELD_SEED = 20261114
N_PATTERNS = 10
NOISE_FIELDS = 5
BETAS = (0.0, 0.25, 0.50, 1.00)
KS = (0, 1, 2, 3, 4)


def _predict(net, d, dev, cfg, keep=None, x=None, bs=512):
    net.eval()
    x = d['x'] if x is None else x
    n = len(x)
    outs = []
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            o = net(torch.from_numpy(np.ascontiguousarray(x[sl])).to(dev),
                    mask=torch.from_numpy(d['mask'][sl]).to(dev),
                    baseline=torch.from_numpy(d['baseline'][sl]).to(dev),
                    cycle=(torch.from_numpy(d['cycle'][sl]).to(dev) if net.use_cycle else None),
                    cond=torch.from_numpy(d['cond'][sl]).to(dev),
                    mode=cfg['mode'],
                    sensor_keep=(torch.from_numpy(keep[sl]).to(dev) if keep is not None else None))
            outs.append(o['prediction'].reshape(-1).float().cpu().numpy())
    return np.concatenate(outs)


def metric_block(ds, pred):
    u, y = ds['unit'], ds['y']
    s = summarize(pred, y)
    return dict(engine_equal_rmse=engine_equal_rmse(u, y, pred),
                engine_equal_mae=engine_equal_mae(u, y, pred),
                engine_equal_nasa_mean=engine_equal_nasa_mean(u, y, pred),
                pooled_rmse=s['rmse'], pooled_mae=s['mae'], pooled_nasa_mean=s['nasa_mean'],
                n=int(len(pred)))


# ---------------------------------------------------------------- fixed masks
def make_missing_patterns(units, n_sensors=N_SENSORS, n_patterns=N_PATTERNS,
                          seed=MASK_PATTERN_SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for u in sorted(int(x) for x in np.unique(units)):
        out[str(u)] = [rng.permutation(n_sensors).tolist() for _ in range(n_patterns)]
    return out


def patterns_hash(patterns):
    blob = json.dumps(patterns, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def missing_keep(units, patterns, pid, k, n_sensors=N_SENSORS):
    keep = np.ones((len(units), n_sensors), bool)
    for i, u in enumerate(units):
        hidden = patterns[str(int(u))][pid][:k]
        if k:
            keep[i, hidden] = False
    return keep


def eval_missing(net, ds, dev, cfg, patterns, ks=KS):
    per = []
    for pid in range(len(next(iter(patterns.values())))):
        for k in ks:
            keep = missing_keep(ds['unit'], patterns, pid, k)
            pred = _predict(net, ds, dev, cfg, keep=keep)
            m = metric_block(ds, pred)
            m.update(pattern_id=pid, k=int(k), n_sensors_kept=int(N_SENSORS - k))
            per.append(m)
    agg = {}
    for k in ks:
        rows = [r for r in per if r['k'] == k]
        agg[str(k)] = {key: float(np.mean([r[key] for r in rows])) for key in
                       ('engine_equal_rmse', 'engine_equal_mae', 'engine_equal_nasa_mean',
                        'pooled_rmse')}
        agg[str(k)]['std_engine_equal_rmse'] = float(np.std([r['engine_equal_rmse'] for r in rows]))
        agg[str(k)]['n_patterns'] = len(rows)
    return dict(per_pattern=per, aggregate=agg)


def eval_single_sensor(net, ds, dev, cfg):
    """Hide exactly one sensor at a time, all 14 listed (plan 11.2)."""
    rows = []
    for j in range(N_SENSORS):
        keep = np.ones((len(ds['unit']), N_SENSORS), bool)
        keep[:, j] = False
        pred = _predict(net, ds, dev, cfg, keep=keep)
        m = metric_block(ds, pred)
        m.update(sensor_index=int(j))
        rows.append(m)
    return rows


# --------------------------------------------------------------- fixed noise
def make_noise_fields(units, lengths, n_sensors=N_SENSORS, n_fields=NOISE_FIELDS,
                      seed=NOISE_FIELD_SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for u in sorted(int(x) for x in np.unique(units)):
        out[str(u)] = [rng.standard_normal((int(lengths[int(u)]), n_sensors)).astype(np.float32)
                       for _ in range(n_fields)]
    return out


def engine_lengths(ds):
    lengths = {}
    for u, c in zip(ds['unit'], ds['cut']):
        lengths[int(u)] = max(lengths.get(int(u), 0), int(c))
    return lengths


def noise_fields_hash(fields):
    h = hashlib.sha256()
    for u in sorted(fields):
        for arr in fields[u]:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def perturbed_x(ds, sigma, fields, fid, beta):
    """Perturbed copy of ds['x']; the baseline stays the clean prefix estimate."""
    x = np.array(ds['x'], copy=True)
    sigma = np.asarray(sigma, np.float64)
    for i in range(len(x)):
        u = int(ds['unit'][i])
        cut = int(ds['cut'][i])
        m = int(ds['mask'][i].sum())
        eps = fields[str(u)][fid][cut - m:cut]
        s = sigma[np.asarray(ds['cond_seq'][i, -m:], np.int64)]
        x[i, -m:, :] = x[i, -m:, :] + float(beta) * s.astype(np.float32) * eps
    return x


def eval_noise(net, ds, dev, cfg, sigma, fields, betas=BETAS):
    per = []
    for fid in range(len(next(iter(fields.values())))):
        for beta in betas:
            x = perturbed_x(ds, sigma, fields, fid, beta)
            pred = _predict(net, ds, dev, cfg, x=x)
            m = metric_block(ds, pred)
            m.update(noise_field_id=fid, beta=float(beta))
            per.append(m)
    agg = {}
    for beta in betas:
        rows = [r for r in per if r['beta'] == float(beta)]
        agg[str(beta)] = {key: float(np.mean([r[key] for r in rows])) for key in
                          ('engine_equal_rmse', 'engine_equal_mae', 'engine_equal_nasa_mean',
                           'pooled_rmse')}
        agg[str(beta)]['std_engine_equal_rmse'] = float(np.std([r['engine_equal_rmse'] for r in rows]))
    return dict(per_field=per, aggregate=agg)


# ---------------------------------------------------------------- prediction
def prediction_rows(ds, pred, meta, protocol, corruption_type='none', corruption_level=0,
                    pattern_id='', n_kept=N_SENSORS):
    units = np.asarray(ds['unit'])
    cuts = np.asarray(ds['cut'])
    cutoff_id = {}
    rows = []
    for i in range(len(pred)):
        u = int(units[i])
        cid = cutoff_id.get(u, 0)
        cutoff_id[u] = cid + 1
        rows.append(dict(
            study_id=meta['study_id'], run_id=meta['run_id'], group=meta['group'],
            dataset=meta['dataset'], split_seed=meta['split_seed'], model_seed=meta['model_seed'],
            engine_id=u, cutoff_id=cid, cutoff_cycle=int(cuts[i]),
            valid_length=int(ds['mask'][i].sum()),
            y_true=float(ds['y'][i]), y_pred_raw=float(pred[i]), y_pred_evaluated=float(pred[i]),
            evaluation_protocol=protocol, corruption_type=corruption_type,
            corruption_level=corruption_level, mask_pattern_id=pattern_id,
            mask_hash=meta.get('mask_hash', ''), noise_draw_id=meta.get('noise_draw_id', ''),
            checkpoint_id=meta.get('checkpoint_id', 'best.pt'),
            best_epoch=meta.get('best_epoch', -1), n_sensors_kept=int(n_kept),
        ))
    return rows


def benchmark_inference(net, ds, dev, cfg, batch=512, reps=20):
    """Latency and peak memory of a clean forward pass (plan Table D)."""
    import time
    net.eval()
    n = min(batch, len(ds['x']))
    xb = torch.from_numpy(np.ascontiguousarray(ds['x'][:n])).to(dev)
    mb = torch.from_numpy(ds['mask'][:n]).to(dev)
    bb = torch.from_numpy(ds['baseline'][:n]).to(dev)
    cb = torch.from_numpy(ds['cycle'][:n]).to(dev)
    nb = torch.from_numpy(ds['cond'][:n]).to(dev)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        for _ in range(3):
            net(xb, mask=mb, baseline=bb, cycle=cb, cond=nb, mode=cfg['mode'])
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            net(xb, mask=mb, baseline=bb, cycle=cb, cond=nb, mode=cfg['mode'])
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t0) / reps
    peak = (torch.cuda.max_memory_allocated(dev) / 2 ** 20) if dev.type == 'cuda' else 0.0
    return dict(ms_per_batch=round(dt * 1e3, 3), batch=n, reps=reps,
                rows_per_s=round(n / dt, 1), peak_mem_mib=round(peak, 1))


FIELDS = ['study_id', 'run_id', 'group', 'dataset', 'split_seed', 'model_seed',
          'engine_id', 'cutoff_id', 'cutoff_cycle', 'valid_length', 'y_true',
          'y_pred_raw', 'y_pred_evaluated', 'evaluation_protocol', 'corruption_type',
          'corruption_level', 'mask_pattern_id', 'mask_hash', 'noise_draw_id',
          'checkpoint_id', 'best_epoch', 'n_sensors_kept']


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def evaluate_clean(net, ds, dev, cfg, meta, out_dir, split='val'):
    out_dir = Path(out_dir)
    pred = _predict(net, ds, dev, cfg)
    m = metric_block(ds, pred)
    write_rows(out_dir / f'{split}_predictions.csv',
               prediction_rows(ds, pred, meta, f'clean_{split}'))
    return m, pred


def evaluate_robustness(net, ds, dev, cfg, meta, patterns, sigma, out_dir,
                        missing_ks=KS, betas=BETAS, single_sensor=False, split='val'):
    out_dir = Path(out_dir)
    miss = eval_missing(net, ds, dev, cfg, patterns, ks=missing_ks)
    rows = []
    for pid in range(len(next(iter(patterns.values())))):
        for k in missing_ks:
            keep = missing_keep(ds['unit'], patterns, pid, k)
            pred = _predict(net, ds, dev, cfg, keep=keep)
            rows += prediction_rows(ds, pred, {**meta, 'mask_hash': meta['mask_hash']},
                                    f'missing_{split}', 'missing', k, pid, N_SENSORS - k)
    write_rows(out_dir / f'{split}_missing_predictions.csv', rows)

    fields = meta['noise_fields']
    noise = eval_noise(net, ds, dev, cfg, sigma, fields, betas=betas)
    rows = []
    for fid in range(len(next(iter(fields.values())))):
        for beta in betas:
            x = perturbed_x(ds, sigma, fields, fid, beta)
            pred = _predict(net, ds, dev, cfg, x=x)
            rows += prediction_rows(ds, pred, {**meta, 'noise_draw_id': str(fid)},
                                    f'noise_{split}', 'noise', beta, '', N_SENSORS)
    write_rows(out_dir / f'{split}_noise_predictions.csv', rows)

    out = dict(missing=miss, noise=noise)
    if single_sensor:
        out['single_sensor'] = eval_single_sensor(net, ds, dev, cfg)
    return out
