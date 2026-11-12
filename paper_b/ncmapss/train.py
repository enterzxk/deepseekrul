"""N0-N3 training rungs for N-CMAPSS DS02.

All four rungs consume the frozen 51-column flight matrix from
``paper_b/ncmapss/features.py`` (protocol §4.1). N0 receives the age column only.
N1, N2 and N3 differ **only** in their input layer and encoder:

    N0  age-only MLP                     1 feature
    N1  Vanilla Transformer over flights 51 features, flat embedding
    N2  DSPT-Base ported to flights      51 features, 14 sensor tokens x 3 stats
    N3  DSPT with the C-MAPSS-frozen switches, read from the freeze manifest

Two rules are enforced rather than documented:

* the training schedule is read verbatim from ``paper_b/configs/protocol.json``
  (40 epochs / patience 12 / batch min(512, N) / lr 1e-3 / wd 1e-4 / d=64 /
  2 layers / dropout 0.2 / seeds 101,202,303). Only the input width changes.
  A rung that does not converge under this schedule is a finding, not a licence
  to tune.
* N3's switch setting is read from ``FREEZE_ABLATION_SEED101.json`` plus the A8
  grid entry it names. Nothing is hand-copied; the SHA256 of every source file
  is recorded in the run ledger.

The operating-condition embedding (``n_cond=6``) is a k-means on ``W`` fitted on
the current LOEO fold's training engines only. It carries **no information the
Vanilla rung lacks**: ``W``'s per-flight mean/std are already columns 43-50 of
the shared matrix, and the cluster id is a function of ``W``. Any change that
gives the DSPT rungs strictly more input is a protocol violation, not an
architecture choice.

    python -m paper_b.ncmapss.train --synthetic          # one epoch, all rungs
    python -m paper_b.ncmapss.train --h5 data/NCMAPSS/N-CMAPSS_DS02-006.h5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (N_FEATURES, N_SENSORS, N_STATS, SPAN, AgeScaler,
                       build_matrix, filter_low_confidence, flight_table,
                       make_windows)

N_CONTEXT = N_FEATURES - N_SENSORS * N_STATS      # 9 = 8 W stats + age
ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# frozen sources
# --------------------------------------------------------------------------
def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(root=ROOT):
    """Read the training schedule and N3 switches from the frozen files.

    ``protocol.json`` gives the schedule; ``FREEZE_ABLATION_SEED101.json`` names
    the runs and states in its note that the final model is MS+SA, which
    resolves to the ``A8_mssa_L70`` grid entry. The grid entry is where the
    switch *values* live, so it is read rather than transcribed.
    """
    root = Path(root)
    proto_p = root / 'paper_b/configs/protocol.json'
    freeze_p = root / 'paper_b/results/FREEZE_ABLATION_SEED101.json'
    grid_p = root / 'paper_b/experiments/grids/ablation8.json'
    for p in (proto_p, freeze_p, grid_p):
        if not p.exists():
            raise FileNotFoundError(
                f'{p} missing: the N3 freeze cannot be verified on this branch. '
                'Bring the C-MAPSS freeze files over rather than re-typing values.')
    proto = json.loads(proto_p.read_text())
    freeze = json.loads(freeze_p.read_text())
    if not freeze.get('frozen'):
        raise AssertionError('FREEZE_ABLATION_SEED101.json is not marked frozen')
    note = freeze.get('note', '')
    if 'MS+SA' not in note:
        raise AssertionError(f'freeze note does not name the final model: {note!r}')
    tag = 'A8_mssa_L70'
    if not any(r['tag'] == tag for r in freeze['runs']):
        raise AssertionError(f'{tag} not present in the freeze manifest runs')
    grid = json.loads(grid_p.read_text())
    entry = next((r for r in grid if r['tag'] == tag), None)
    if entry is None:
        raise AssertionError(f'{tag} not found in ablation8.json')
    return dict(
        epochs=int(proto['epochs']), batch=int(proto['batch']), lr=float(proto['lr']),
        weight_decay=float(proto['weight_decay']), d_model=int(proto['d_model']),
        layers=int(proto['layers']), dropout=float(proto['dropout']),
        seeds=[int(s) for s in proto['seeds']], n_cond=int(proto['n_cond']),
        heads=4,   # architecture constant of DSPT, not a training knob
        # D043 / protocol 4.2: no early stopping. The held-out unit must not pick
        # a stopping point, so the schedule is a fixed 40 epochs and the model at
        # the last epoch is the one reported.
        patience_ignored=int(proto['patience']),
        # Label-construction fields are NOT inherited from C-MAPSS: DS02 RUL is in
        # flight cycles, and val_cuts / val_frac describe an 80/20 split that LOEO
        # does not use. They are recorded here only so the non-inheritance is
        # auditable.
        label_fields_not_inherited={k: proto[k] for k in ('rul_cap', 'val_cuts', 'val_frac')},
        n3=dict(ms_mode=entry['ms_mode'], sa_mode=entry['sa_mode'],
                hi_source=entry['hi_source'], tag=tag),
        provenance={str(p.relative_to(root)): _sha256(p)
                    for p in (proto_p, freeze_p, grid_p)},
    )


class CondEncoder:
    """k-means(k=6) on the 1 Hz ``W`` rows, fitted on training engines only."""

    def __init__(self, n_cond=6, seed=7):
        self.n_cond = int(n_cond)
        self.seed = int(seed)
        self.km = None

    def fit(self, W_rows):
        from sklearn.cluster import KMeans
        self.km = KMeans(n_clusters=self.n_cond, random_state=self.seed,
                         n_init=10).fit(np.asarray(W_rows, np.float64))
        return self

    def labels(self, W_rows):
        if self.km is None:
            raise RuntimeError('CondEncoder.fit must be called first')
        return self.km.predict(np.asarray(W_rows, np.float64)).astype(np.int64)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class AgeOnly(nn.Module):
    """N0: deterministic map from age to RUL, no sequence."""

    dual_span = False

    def __init__(self, d=64, layers=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask=None, cond=None, **kw):
        age = x[:, -1, -1:]                       # windows are right-aligned
        y = self.net(age)
        return {'prediction': y, 'y_A': y, 'y_B': y, 'embedding': age}


class VanillaFlight(nn.Module):
    """N1: standard encoder over the 70-flight sequence of 51-dim vectors.

    ``n_cond`` is only set for the ``N1c`` diagnostic arm (vanilla + the same
    operating-condition embedding DSPT gets). N1 itself never reads ``cond``, so
    the N1-vs-N2 gap mixes the tokenisation with the condition embedding; N1c
    removes one of those two differences and is reported as a diagnostic, not as
    a main arm.
    """

    dual_span = False

    def __init__(self, d=64, layers=2, heads=4, dropout=0.2, span=SPAN, n_cond=None):
        super().__init__()
        self.embed = nn.Linear(N_FEATURES, d)
        pe = torch.zeros(span, d)
        pos = torch.arange(span, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, 'gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.cond_emb = nn.Embedding(n_cond, d) if n_cond else None
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, mask=None, cond=None, **kw):
        h = self.embed(x) + self.pe[:, :x.size(1)]
        pad = ~mask.bool()
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=pad)
        pooled = masked_mean(h, mask)
        if self.cond_emb is not None and cond is not None:
            pooled = pooled + self.cond_emb(cond)
        y = self.head(self.norm(pooled))
        return {'prediction': y, 'y_A': y, 'y_B': y, 'embedding': h}


class DSPTFlight(nn.Module):
    """N2/N3: DSPT structure on the flight matrix.

    The 42 sensor-statistic columns are read as 14 sensor tokens with 3 channels
    (mean/std/slope). The 9 context columns and the operating-condition
    embedding are added after sensor pooling, the way the cycle encoding is
    added in the C-MAPSS model.
    """

    dual_span = True

    def __init__(self, d=64, layers=2, heads=4, dropout=0.2, ms_mode='off',
                 sa_mode='off', n_cond=6, span=SPAN):
        super().__init__()
        self.ms_mode = ms_mode
        self.sa_mode = sa_mode
        self.sensor_in = nn.Linear(N_STATS, d)
        if ms_mode != 'linear_only':
            self.conv3 = nn.Conv1d(d, d, 3, padding=1, groups=d, padding_mode='replicate')
        if ms_mode in ('full', 'multi_noproj'):
            self.conv5 = nn.Conv1d(d, d, 5, padding=2, groups=d, padding_mode='replicate')
            self.conv7 = nn.Conv1d(d, d, 7, padding=3, groups=d, padding_mode='replicate')
        if ms_mode in ('full', 'single_proj', 'linear_only'):
            self.ms_proj = nn.Linear(d, d)
        self.ctx = nn.Linear(N_CONTEXT, d)
        self.cond_emb = nn.Embedding(n_cond, d)
        self.time_norm = nn.LayerNorm(d)
        self.temporal_pe = nn.Parameter(torch.zeros(1, span, d))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, 'gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        if sa_mode == 'basic':
            self.sensor_attn = nn.TransformerEncoderLayer(d, 2, 2 * d, dropout,
                                                          batch_first=True, norm_first=True)
        self.sensor_score = nn.Linear(d, 1)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def _ms(self, hh):
        if self.ms_mode == 'linear_only':
            return self.ms_proj(hh.transpose(1, 2)).transpose(1, 2)
        if self.ms_mode == 'off':
            return F.gelu(self.conv3(hh))
        if self.ms_mode == 'single_proj':
            return self.ms_proj(F.gelu(self.conv3(hh)).transpose(1, 2)).transpose(1, 2)
        ms = (F.gelu(self.conv3(hh)) + F.gelu(self.conv5(hh)) + F.gelu(self.conv7(hh))) / 3.0
        return self.ms_proj(ms.transpose(1, 2)).transpose(1, 2)

    def encode(self, x, mask, cond):
        B, L, _ = x.shape
        sensor = x[..., :N_SENSORS * N_STATS].reshape(B, L, N_SENSORS, N_STATS)
        ctx = x[..., N_SENSORS * N_STATS:]
        h = self.sensor_in(sensor)                        # (B,L,N,D)
        hh4 = h.permute(0, 2, 3, 1)                       # (B,N,D,L)
        valid = mask.bool()
        first = valid.float().argmax(1)
        fv = hh4[torch.arange(B, device=x.device), :, :, first]
        m4 = valid.unsqueeze(1).unsqueeze(1)
        hh = torch.where(m4, hh4, fv.unsqueeze(-1)).reshape(B * N_SENSORS, self.sensor_in.out_features, L)
        conv = self._ms(hh) * m4.expand(B, N_SENSORS, 1, L).reshape(B * N_SENSORS, 1, L)
        conv_out = conv.reshape(B, N_SENSORS, -1, L).permute(0, 3, 1, 2)
        h = self.time_norm(h + conv_out)                  # (B,L,N,D)

        tokens = h.permute(0, 2, 1, 3).reshape(B * N_SENSORS, L, -1)
        dist = torch.arange(L - 1, -1, -1, device=x.device)
        tokens = tokens + self.temporal_pe[:, dist]
        pad = ~mask.bool().unsqueeze(1).expand(B, N_SENSORS, L).reshape(B * N_SENSORS, L)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=pad)
        tokens = tokens.reshape(B, N_SENSORS, L, -1).permute(0, 2, 1, 3)

        if self.sa_mode == 'basic':
            st = tokens.reshape(B * L, N_SENSORS, -1)
            st = self.sensor_attn(st)
            tokens = st.reshape(B, L, N_SENSORS, -1)

        score = torch.softmax(self.sensor_score(tokens).squeeze(-1), dim=-1)
        pooled = (score.unsqueeze(-1) * tokens).sum(2)     # (B,L,D)
        pooled = pooled + self.ctx(ctx)
        if cond is not None:
            pooled = pooled + self.cond_emb(cond).unsqueeze(1)
        return self.norm(pooled)

    def forward(self, x, mask=None, cond=None, **kw):
        h = self.encode(x, mask, cond)
        L = h.size(1)
        dist = torch.arange(L - 1, -1, -1, device=x.device)
        short = mask.bool() & (dist.unsqueeze(0) < 50)
        hA = masked_mean(h, short.to(h.dtype))
        hB = masked_mean(h, mask)
        m = mask.sum(1, keepdim=True)
        g = torch.clamp((m - 50) / 50.0, 0.0, 1.0)
        hf = g * hB + (1 - g) * hA
        return {'prediction': self.head(hf), 'y_A': self.head(hA),
                'y_B': self.head(hB), 'embedding': hf}


def build(rung, cfg):
    if rung == 'N0':
        return AgeOnly(d=cfg['d_model'], layers=cfg['layers'])
    if rung == 'N1':
        return VanillaFlight(d=cfg['d_model'], layers=cfg['layers'],
                             heads=cfg['heads'], dropout=cfg['dropout'])
    if rung == 'N1c':
        # diagnostic only: vanilla + the operating-condition embedding, so the
        # N1-vs-N2 gap can be split into "tokenisation" and "condition id"
        return VanillaFlight(d=cfg['d_model'], layers=cfg['layers'],
                             heads=cfg['heads'], dropout=cfg['dropout'],
                             n_cond=cfg['n_cond'])
    if rung == 'N2':
        return DSPTFlight(d=cfg['d_model'], layers=cfg['layers'], heads=cfg['heads'],
                          dropout=cfg['dropout'], ms_mode='off', sa_mode='off',
                          n_cond=cfg['n_cond'])
    if rung == 'N3':
        f = cfg['n3']
        return DSPTFlight(d=cfg['d_model'], layers=cfg['layers'], heads=cfg['heads'],
                          dropout=cfg['dropout'], ms_mode=f['ms_mode'],
                          sa_mode=f['sa_mode'], n_cond=cfg['n_cond'])
    raise ValueError(rung)


# --------------------------------------------------------------------------
# data + training
# --------------------------------------------------------------------------
def asymmetric_loss(pred, y):
    e = pred.reshape(-1, 1) - y.reshape(-1, 1)
    return e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()


def load_subset(path, train_units, cfg, root=ROOT, subset=None):
    """Build features, operating-condition ids and windows for one N-CMAPSS subset.

    ``path`` may be any of DS01..DS08: the development split is read from the
    file's own ``A_dev`` array, never from a unit list stored in code, so the
    same call works for every subset. ``subset`` is only a label carried into
    the returned dict for provenance.

    Fitting happens on ``train_units`` only: the age scale and the k-means are
    both fold-local. Flights with too few 1 Hz samples are dropped explicitly
    and counted.
    """
    from .data import load_aux, load_features, load_labels
    f = load_features(path, 'dev')
    a = load_aux(path, 'dev')
    y = load_labels(path, 'dev')
    A = np.stack([a['unit'], a['cycle'], a['Fc'], a['hs']], axis=1)
    tr_row = np.isin(a['unit'], list(train_units))
    ce = CondEncoder(cfg['n_cond'], seed=cfg['seeds'][0]).fit(f['W'][tr_row])
    lab = ce.labels(f['W'])
    tbl = flight_table(f['W'], f['X_s'], A, cond_labels=lab, n_cond=cfg['n_cond'])
    y_flight = y[tbl['row_index']]          # one label per flight
    tbl, drop_info = filter_low_confidence(tbl)
    y_flight = y_flight[drop_info['keep']]
    tr = np.isin(tbl['unit'], list(train_units))
    scaler = AgeScaler().fit(tbl['cycle'][tr])
    X = build_matrix(tbl, scaler)
    d = make_windows(X, tbl['unit'], tbl['cycle'], y=y_flight)
    d['cond'] = tbl['cond']
    d['drop_info'] = drop_info
    d['cond_labels'] = lab
    d['is_train'] = np.isin(tbl['unit'], list(train_units))
    d['subset'] = subset or Path(path).stem
    from .data import unit_lifetime as _lifetime
    d['unit_lifetime'] = _lifetime(path, 'dev')
    return d


def _loss_on(net, x, mask, cond, y, bs, device, train):
    """Mean asymmetric loss over a tensor split; no gradient when ``train`` is False."""
    n = len(x)
    total, nb = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            with torch.autocast(device_type=device.split(':')[0],
                                dtype=torch.bfloat16, enabled=device != 'cpu'):
                out = net(x[sl], mask=mask[sl], cond=cond[sl])
                loss = asymmetric_loss(out['prediction'].float(), y[sl])
                if getattr(net, 'dual_span', False):
                    loss = loss + 0.25 * (asymmetric_loss(out['y_A'].float(), y[sl]) +
                                          asymmetric_loss(out['y_B'].float(), y[sl]))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            total += float(loss.detach())
            nb += 1
    return total / max(nb, 1)


def fit_rung(rung, data, cfg, seed, device='cpu', epochs=None, batch=None,
             normalize_target=True, early_stop=False, max_epochs=400,
             min_epochs=20, patience=12, snapshot_at=None):
    """Train one rung and return ``(net, report)``.

    ``early_stop=True`` is used only by the **inner** selection folds, whose
    validation unit is one of the outer fold's training engines. The outer
    evaluation trains for a fixed ``epoch*`` and reports the last epoch, so no
    outer held-out information selects anything (protocol 4.2).

    ``snapshot_at`` is an optional iterable of epoch numbers; the weights at the
    end of each such epoch are cloned into ``report['snapshots']``. It exists so
    that one training run can be scored at more than one epoch (the
    inner-selection ``epoch*`` and the fixed 200-pass endpoint) without
    retraining, which would let a reviewer attribute the difference to two
    different fits instead of two different stopping points.
    """
    torch.manual_seed(seed)
    net = build(rung, cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    x = torch.from_numpy(data['x']).to(device)
    mask = torch.from_numpy(data['mask']).to(device)
    y = torch.from_numpy(data['y']).to(device)
    cond = torch.from_numpy(data['cond']).to(device)
    if rung == 'N0':
        x = x[..., -1:]
    is_train = torch.from_numpy(np.asarray(data['is_train'], bool)).to(device)
    tr_idx = torch.nonzero(is_train, as_tuple=False).flatten()
    va_idx = torch.nonzero(~is_train, as_tuple=False).flatten()
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise ValueError('need at least one training and one held-out unit')
    # target standardisation, fitted on the training fold only
    if normalize_target:
        ytr = np.asarray(data['y'], np.float64)[np.asarray(data['is_train'], bool)]
        target_mu, target_sd = float(ytr.mean()), float(ytr.std() + 1e-8)
        y = (y - target_mu) / target_sd
    else:
        target_mu, target_sd = 0.0, 1.0
    n = len(tr_idx)
    bs = min(int(batch if batch is not None else cfg['batch']), n)
    epochs = int(max_epochs if early_stop else (cfg['epochs'] if epochs is None else epochs))
    hist = []
    snaps = {}
    want = {int(e) for e in snapshot_at} if snapshot_at else set()
    best_va, best_ep = float('inf'), -1
    for ep in range(1, epochs + 1):
        net.train()
        perm = tr_idx[torch.randperm(n, device=device)]
        total, nb = 0.0, 0
        for i in range(0, n, bs):
            sl = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(':')[0],
                                dtype=torch.bfloat16, enabled=device != 'cpu'):
                out = net(x[sl], mask=mask[sl], cond=cond[sl])
                loss = asymmetric_loss(out['prediction'].float(), y[sl])
                if getattr(net, 'dual_span', False):
                    loss = loss + 0.25 * (asymmetric_loss(out['y_A'].float(), y[sl]) +
                                          asymmetric_loss(out['y_B'].float(), y[sl]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            nb += 1
        tr_loss = total / max(nb, 1)
        # diagnostic on an outer fold; the selection signal only on an inner fold
        va_loss = _loss_on(net, x[va_idx], mask[va_idx], cond[va_idx], y[va_idx],
                           bs, device, train=False)
        hist.append(dict(epoch=ep, train_loss=tr_loss, holdout_loss=va_loss))
        if va_loss < best_va - 1e-4:
            best_va, best_ep = va_loss, ep
        if ep in want:
            # CPU copies: these are written to disk as checkpoints, and a GPU
            # tensor would make the file unloadable on a host without that card
            snaps[ep] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if early_stop and ep >= min_epochs and ep - best_ep >= patience:
            break
    net.eval()
    with torch.no_grad():
        out = net(x[:1], mask=mask[:1], cond=cond[:1])
    report = dict(rung=rung, seed=int(seed), n=int(n), batch=bs, epochs=int(epochs),
                  epochs_run=len(hist), best_epoch=int(best_ep),
                  params=int(sum(p.numel() for p in net.parameters())),
                  target_mu=float(target_mu), target_sd=float(target_sd),
                  final_train_loss=hist[-1]['train_loss'],
                  final_holdout_loss=hist[-1]['holdout_loss'],
                  out_shape=list(out['prediction'].shape), history=hist,
                  snapshots=snaps)
    return net, report


def run_rung(rung, data, cfg, seed, device='cpu', epochs=None):
    """Same as :func:`fit_rung` but drops the model, for the smoke-test CLI."""
    _, report = fit_rung(rung, data, cfg, seed, device=device, epochs=epochs)
    return report


def _self_check(data, cycles):
    x, mask, y, cyc = data['x'], data['mask'], data['y'], data['cycle']
    assert x.shape[2] == N_FEATURES, x.shape
    assert len(y) == len(x)
    assert np.allclose(mask.sum(1), np.minimum(cyc, SPAN))
    assert np.all(mask[:, -1] == 1.0)
    assert np.allclose(y, cycles - cyc), 'label must be the RUL at the window end'
    assert x[..., -1:].shape[2] == 1
    # short-flight guard: a 3-sample flight must be dropped and counted
    n1 = 12 * 10
    W = np.zeros((n1 + 3, 4)); X = np.zeros((n1 + 3, 14)); A = np.zeros((n1 + 3, 4))
    A[:n1, 0] = 1; A[:n1, 1] = np.repeat(np.arange(12), 10)
    A[n1:, 0] = 2; A[n1:, 1] = 1
    tbl = flight_table(W, X, A, min_samples=10)
    assert bool(tbl['low_confidence'][-1]) is True
    assert np.isnan(tbl['features'][-1, 28]), 'short flight must have NaN slope, not 0'
    kept, info = filter_low_confidence(tbl)
    assert info['n_dropped'] == 1 and info['n_kept'] == 12
    print(f'self-check PASS: {x.shape[0]} windows, {x.shape[2]} features, '
          f'mask==min(cycle,{SPAN}), label aligned, cond in [0,{data["cond"].max()}], '
          f'short-flight guard drops 1/13')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--h5', default='')
    ap.add_argument('--train-units', default='2,5,10,16,18')
    ap.add_argument('--epochs', type=int, default=0, help='override for smoke tests only')
    ap.add_argument('--out', default='experiments/ncmapss/rung_smoke.json')
    a = ap.parse_args()
    cfg = load_frozen()
    print('frozen schedule:', {k: cfg[k] for k in
                               ('epochs', 'batch', 'lr', 'weight_decay',
                                'd_model', 'layers', 'dropout', 'seeds')},
          '| patience ignored:', cfg['patience_ignored'])
    print('label fields not inherited:', cfg['label_fields_not_inherited'])
    print('N3 switches from freeze:', cfg['n3'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if a.synthetic:
        import tempfile
        from .test_contract import _write_fake
        td = tempfile.mkdtemp()
        p = Path(td) / 'fake_ds02.h5'
        _write_fake(p, dev_units=(2, 5, 10), test_units=(11,), cycles=80, rows_per_cycle=12)
        data = load_subset(p, [2, 10], cfg)
        _self_check(data, cycles=80)
    else:
        if not a.h5:
            raise SystemExit('--h5 is required unless --synthetic')
        data = load_subset(Path(a.h5), [int(u) for u in a.train_units.split(',')], cfg)

    print(f'windows {data["x"].shape}  mask mean {data["mask"].mean():.3f}  '
          f'y [{data["y"].min():.1f},{data["y"].max():.1f}]  device {device}')
    print('low-confidence drops:', data['drop_info']['n_dropped'], '/',
          data['drop_info']['n_total'])
    epochs = a.epochs or cfg['epochs']
    rows = [run_rung(r, data, cfg, seed=cfg['seeds'][0], device=device, epochs=epochs)
            for r in ('N0', 'N1', 'N2', 'N3')]
    for r in rows:
        print(f'{r["rung"]}: final_train={r["final_train_loss"]:.4f} '
              f'holdout={r["final_holdout_loss"]:.4f} params={r["params"]:,} '
              f'n={r["n"]} batch={r["batch"]} epochs={r["epochs"]} out={r["out_shape"]}')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    di = {k: v for k, v in data['drop_info'].items() if k != 'keep'}
    out.write_text(json.dumps(dict(device=device, windows=list(data['x'].shape),
                                   mask_mean=float(data['mask'].mean()),
                                   drop_info=di,
                                   provenance=cfg['provenance'], runs=rows), indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
