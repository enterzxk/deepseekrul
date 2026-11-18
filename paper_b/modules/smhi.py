"""SMHI-RF pilot: sensor-wise monotone HI bank with reliability fusion.

The frozen B0 backbone is used as a feature extractor. A forward hook on the
cross-sensor attention captures per-sensor tokens h_tj (B,L,N,D). The HI bank
builds a non-negative increment per sensor and a monotone cumulative HI per
sensor, learns a reliability weight per sensor, and fuses them into H_t. A
zero-initialised residual head maps [z, H_last] to a RUL correction, so at step
0 the prediction is exactly B0's.

Only the HI bank + residual head are trained. The backbone and reg_head are
frozen. Validation engines are never used for fitting or early stopping; the
early-stopping split is a held-out 20% of the TRAINING engines.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import rul_loss
from paper_b.modules.temporal_agg import build_model


class SensorHIBank(nn.Module):
    def __init__(self, d_model, n_sensors=14, hidden=32):
        super().__init__()
        self.inc = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        self.rel = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        self.res = nn.Sequential(nn.Linear(d_model + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.res[2].weight)
        nn.init.zeros_(self.res[2].bias)

    def forward(self, st, mask, z):
        B, L, N, D = st.shape
        delta = F.softplus(self.inc(st)).squeeze(-1) * mask.unsqueeze(-1)
        H = torch.cumsum(delta, dim=1)                 # (B,L,N)
        last = mask.sum(1).long() - 1
        ar = torch.arange(B, device=st.device)
        h_last = st[ar, last]                          # (B,N,D)
        ctx = h_last.mean(1, keepdim=True).expand(-1, N, -1)
        r = torch.softmax(self.rel(torch.cat([h_last, ctx], -1)).squeeze(-1),
                          dim=-1)                      # (B,N)
        H_last = H[ar, last]                           # (B,N)
        H_fused = (r * H_last).sum(-1)                 # (B,)
        res = self.res(torch.cat([z, H_fused.unsqueeze(-1)], -1)).reshape(-1)
        return res, H_fused, r


def make_samples(eb, cuts):
    out = []
    for i in range(len(eb['lengths'])):
        n = int(eb['lengths'][i])
        uid = int(eb['engine_id'][i])
        seen = set()
        for f in cuts:
            c = int(min(n, max(1, round(f * n))))
            if c in seen:
                continue
            seen.add(c)
            out.append(dict(i=i, t=c - 1, c=c, n=n, uid=uid,
                            y=float(min(n - c, 125.0))))
    return out


def batch_tensors(eb, samples, idx, dev):
    ss = [samples[k] for k in idx]
    i = np.array([s['i'] for s in ss], np.int64)
    t = np.array([s['t'] for s in ss], np.int64)
    x = torch.from_numpy(np.asarray(eb['x'][i, t], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][i, t], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][i, t], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][i, t], np.int64)).to(dev)
    y = torch.tensor([s['y'] for s in ss], dtype=torch.float32, device=dev)
    cy = torch.zeros(len(ss), device=dev)
    uid = np.array([s['uid'] for s in ss], np.int64)
    return x, m, b, cd, cy, y, uid


def predict(net, hi, eb, samples, dev, bs=256):
    net.eval(); hi.eval()
    preds = []
    for k in range(0, len(samples), bs):
        idx = list(range(k, min(k + bs, len(samples))))
        x, m, b, cd, cy, y, uid = batch_tensors(eb, samples, idx, dev)
        with torch.no_grad():
            z = net.pooled_feature(x, m, b, cy, cd)
            st = net._smhi_st.reshape(len(idx), x.shape[1], 14, -1)
            res, _, _ = hi(st, m, z)
            pred = net.reg_head(z).reshape(-1) + res
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def predict_b0(net, eb, samples, dev, bs=256):
    net.eval()
    out = []
    for k in range(0, len(samples), bs):
        idx = list(range(k, min(k + bs, len(samples))))
        x, m, b, cd, cy, _, _ = batch_tensors(eb, samples, idx, dev)
        with torch.no_grad():
            z = net.pooled_feature(x, m, b, cy, cd)
            out.append(net.reg_head(z).reshape(-1).cpu().numpy())
    return np.concatenate(out)


def engine_mean_rmse(uid, pred, y):
    d = pd.DataFrame(dict(uid=uid, se=(np.clip(pred, 0, 125) - y) ** 2))
    return float(np.sqrt(d.groupby('uid')['se'].mean()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--root', default=str(Path.home() / 'sprint24/experiments/hc/_gate'))
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save-preds', default=None,
                    help='write per-cut train/val predictions for stacking')
    a = ap.parse_args()

    run = 'HC-D-B0-%s-s%d-D1' % (a.subset, a.seed)
    ckpt = Path(a.root) / run / 'ckpt' / 'best.pt'
    if not ckpt.exists():
        raise SystemExit('missing checkpoint %s' % ckpt)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    net = build_model(a.subset, a.seed, ckpt, dev)
    for p in net.parameters():
        p.requires_grad_(False)
    # capture per-sensor tokens from the cross-sensor attention
    net._smhi_st = None

    def hook(_m, _i, out):
        net._smhi_st = out

    handle = net.sensor_attn.register_forward_hook(hook)
    d_model = net.d_model
    hi = SensorHIBank(d_model, 14, a.hidden).to(dev)
    opt = torch.optim.AdamW(hi.parameters(), lr=a.lr, weight_decay=a.wd)

    ebt = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='train')
    ebv = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='val')
    tr = make_samples(ebt, C.VAL_CUTS)
    va = make_samples(ebv, C.VAL_CUTS)

    # engine-disjoint early-stopping split of the TRAINING engines
    engines = np.unique(ebt['engine_id'])
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(engines)
    n_hold = max(1, int(round(0.2 * len(perm))))
    hold = set(int(x) for x in perm[:n_hold])
    tr_fit = [s for s in tr if s['uid'] not in hold]
    tr_hold = [s for s in tr if s['uid'] in hold]

    best, best_state, best_ep = float('inf'), None, -1
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(1, a.epochs + 1):
        hi.train()
        order = torch.randperm(len(tr_fit), generator=g).tolist()
        for k in range(0, len(order), a.batch):
            idx = order[k:k + a.batch]
            x, m, b, cd, cy, y, _ = batch_tensors(ebt, tr_fit, idx, dev)
            with torch.no_grad():
                z = net.pooled_feature(x, m, b, cy, cd)
                st = net._smhi_st.reshape(len(idx), x.shape[1], 14, -1)
            res, _, r = hi(st, m, z)
            pred = net.reg_head(z).reshape(-1) + res
            loss = rul_loss(pred, y, 125.0).mean()
            loss = loss + 1e-3 * (r * torch.log(r + 1e-8)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hi.parameters(), 1.0)
            opt.step()
        hp = predict(net, hi, ebt, tr_hold, dev)
        hy = np.array([s['y'] for s in tr_hold], float)
        hu = np.array([s['uid'] for s in tr_hold], np.int64)
        val = engine_mean_rmse(hu, hp, hy)
        if val < best - 1e-4:
            best, best_ep = val, ep
            best_state = {k: v.detach().clone() for k, v in hi.state_dict().items()}
        elif ep - best_ep >= a.patience:
            break
    if best_state is not None:
        hi.load_state_dict(best_state)

    # validation: B0 vs SMHI, engine-mean RMSE per engine (5 cuts)
    vy = np.array([s['y'] for s in va], float)
    vu = np.array([s['uid'] for s in va], np.int64)
    p_smhi = predict(net, hi, ebv, va, dev)
    p_b0 = predict_b0(net, ebv, va, dev)
    b0 = engine_mean_rmse(vu, p_b0, vy)
    sm = engine_mean_rmse(vu, p_smhi, vy)
    out = dict(subset=a.subset, seed=a.seed, n_train=len(tr_fit),
               n_hold=len(tr_hold), n_val=len(va), best_ep=best_ep,
               hold_rmse=best, b0=b0, smhi=sm, delta=b0 - sm,
               params=int(sum(p.numel() for p in hi.parameters())))
    if a.save_preds:
        ptr = predict(net, hi, ebt, tr, dev)
        pb0tr = predict_b0(net, ebt, tr, dev)
        rows = []
        for split, samples, ps, pb in [('train', tr, ptr, pb0tr),
                                       ('val', va, p_smhi, p_b0)]:
            for s, p1, p0 in zip(samples, ps, pb):
                rows.append(dict(subset=a.subset, seed=a.seed, split=split,
                                 engine=s['uid'], cut=s['c'], y_cap=s['y'],
                                 p_b0=float(p0), p_smhi=float(p1)))
        pd.DataFrame(rows).to_csv(a.save_preds, index=False)
    print(json.dumps(out, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    handle.remove()


if __name__ == '__main__':
    main()
