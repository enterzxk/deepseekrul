"""Build LTAG v3 prediction cache for B0 (cuts + Delta scan) and run parity A0.1."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.temporal_agg import build_model, predict_windows


def predict_fp32(net, eb, i, idxs, dev):
    x = torch.from_numpy(np.asarray(eb['x'][i, idxs], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][i, idxs], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][i, idxs], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][i, idxs], np.int64)).to(dev)
    cy = torch.zeros(len(idxs), device=dev)
    with torch.no_grad():
        z = net.pooled_feature(x, m, b, cy, cd)
    return net.reg_head(z.float()).detach().reshape(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='B0')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/_gate')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--out', default='/home/user/sprint24/cache/preds')
    ap.add_argument('--deltas', default='5,10,20,30')
    ap.add_argument('--max-j', type=int, default=5)
    ap.add_argument('--fp32', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    run = f'HC-D-B0-{a.subset}-s{a.seed}-D1'
    ckpt = Path(a.root) / run / 'ckpt' / 'best.pt'
    net = build_model(a.subset, a.seed, ckpt, dev)
    eb = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='val')
    rows = []
    deltas = [int(x) for x in a.deltas.split(',')]
    for i in range(len(eb['lengths'])):
        idxs = []
        meta = []
        for c in eb['cut_t'][i]:
            c = int(c)
            if c <= 0:
                continue
            idxs.append(c - 1)
            meta.append((c, c - 1, 0))
            for d in deltas:
                for j in range(a.max_j):
                    t = c - 1 - j * d
                    if t < 0 or eb['mask'][i, t].sum() < 10:
                        continue
                    idxs.append(t)
                    meta.append((c, t, j))
        if not idxs:
            continue
        for (c, t, j) in meta:
            p = float(predict_windows(net, eb, i, [t], dev)[0])
            rows.append(dict(engine=int(eb['engine_id'][i]), cut=c, t_end=t, j=j,
                             pred=p, y_cap=float(eb['y'][i, t]),
                             y_raw=float(eb['r'][i, t]), cond=int(eb['cond'][i, t]),
                             cycle=float(eb['cycle_raw'][i, t]),
                             valid_len=int(eb['mask'][i, t].sum())))
    out = Path(a.out) / a.backbone
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{a.subset}_s{a.seed}.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    # parity A0.1: cuts-only rows vs cata_gate preds
    ref = pd.read_csv(f'/home/user/sprint24/experiments/hc/cata_gate/preds/{a.subset}_s{a.seed}.csv')
    cur = pd.DataFrame(rows)
    cur = cur[cur.j == 0].merge(ref[['engine', 'cut', 'pred_k1']], on=['engine', 'cut'], how='inner')
    diff = float(np.max(np.abs(cur.pred - cur.pred_k1))) if len(cur) else float('nan')
    print(json.dumps(dict(path=str(path), rows=len(rows), parity_max_abs=diff,
                          parity_pass=bool(diff < 1e-4)), indent=1))


if __name__ == '__main__':
    main()
