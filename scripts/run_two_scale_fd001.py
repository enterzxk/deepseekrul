"""Decisive CPU test of candidate E on real frozen features (FD001, one fold).

Pipeline
  1. load the frozen E2ECQR body from experiments/punet_matrix
  2. capture the representation h via a forward hook on body.layers[-1], mean-pooled over time
  3. fit the two scale heads on the TRAINING engines (NLL + pinball)
  4. predict sigma_eng for the held-out VALIDATION engines
  5. report rho(sigma_eng, engine RMSE) and the Winkler comparison against the ablation

No training of the backbone, no gradient through it, no weight updates.

Run on a40:  python scripts/run_two_scale_fd001.py FD001 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_ida as R  # noqa: E402
from src.candidates.model import CandidateModel  # noqa: E402
from src.twoscale.train import Heads, fit_two_scale  # noqa: E402


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def load_body(ck_path: Path) -> CandidateModel:
    sd = torch.load(ck_path, map_location="cpu", weights_only=True)
    body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
    body = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    body.load_state_dict(body_sd, strict=True)
    body.eval()
    return body


def forward_capture(body, x, c, e):
    """Return (h_meanpooled, y_hat)."""
    grabbed = {}

    def hook(_m, _inp, out):
        grabbed["h"] = out[0] if isinstance(out, (tuple, list)) else out

    layer = body.norm
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            try:
                y = body(x, c, e)
            except TypeError:
                try:
                    y = body(x, e)
                except TypeError:
                    y = body(x)
    finally:
        handle.remove()
    h = grabbed["h"]
    print("captured h:", tuple(h.shape))
    if h.ndim == 3:
        h = h.mean(dim=1)
    if isinstance(y, dict):
        print('  forward keys:', list(y.keys()))
        for k in ('y', 'pred', 'y_hat', 'rul', 'mu', 'out', 'y_pred'):
            if k in y:
                y = y[k]
                break
        else:
            y = next(v for v in y.values() if torch.is_tensor(v))
    if isinstance(y, (tuple, list)):
        y = y[0]
    return h.detach().float(), torch.as_tensor(y).reshape(-1).float()


def main() -> int:
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    body = load_body(ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42" / "model.pt")

    splits, frame = R.outer_folds(subset, 5)
    tr_u, va_u = splits[fold]
    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(
        frame, subset, fold, tr_u, va_u, window=50, max_rul=125.0)

    xt = torch.as_tensor(np.asarray(xtr), dtype=torch.float32)
    et = torch.as_tensor(np.asarray(etr), dtype=torch.float32)
    ct = None if ctr is None else torch.as_tensor(np.asarray(ctr), dtype=torch.float32)
    xv = torch.as_tensor(np.asarray(xva), dtype=torch.float32)
    ev = torch.as_tensor(np.asarray(eva), dtype=torch.float32)
    cv = None if cva is None else torch.as_tensor(np.asarray(cva), dtype=torch.float32)

    h_tr, y_tr = forward_capture(body, xt, ct, et)
    h_va, y_va = forward_capture(body, xv, cv, ev)
    print(f"h_tr {tuple(h_tr.shape)}  h_va {tuple(h_va.shape)}")

    # engine ids from the meta array (columns: unit, cutoff)
    m_tr = np.asarray(mtr)
    m_va = np.asarray(mva)
    u_tr = m_tr[:, 0].astype(np.int64)
    u_va = m_va[:, 0].astype(np.int64)
    ids_tr = np.unique(u_tr, return_inverse=True)[1].astype(np.int64)
    ids_va = np.unique(u_va, return_inverse=True)[1].astype(np.int64)
    n_tr, n_va = ids_tr.max() + 1, ids_va.max() + 1
    print(f"train engines {n_tr}  val engines {n_va}")

    yt = np.asarray(ytr, float)
    yv = np.asarray(yva, float)
    ctr_np = y_tr.numpy()
    half = np.ones_like(ctr_np)

    results = {}
    for use_eng in (True, False):
        model, scale_tr, eng_tr = fit_two_scale(
            h_tr.numpy(), ids_tr, ctr_np, half, yt, n_tr, use_engine=use_eng, epochs=300)
        with torch.no_grad():
            win_v, eng_v = model(h_va, torch.as_tensor(ids_va), n_va)
            scale_va = (win_v * eng_v).numpy()
        ctr_va = y_va.numpy()
        rmse_eng = np.array([np.sqrt(np.mean((yv[ids_va == e] - ctr_va[ids_va == e]) ** 2))
                             for e in range(n_va)])
        if use_eng:
            pred_eng = np.array([eng_v.numpy()[ids_va == e][0] for e in range(n_va)])
            rho = spearman(pred_eng, rmse_eng)
        else:
            rho = float("nan")
        lo = ctr_va - scale_va
        hi = ctr_va + scale_va
        w = float(np.mean((hi - lo) + 20 * (np.clip(lo - yv, 0, None) + np.clip(yv - hi, 0, None))))
        results[use_eng] = (rho, w)
        print(f"use_engine={use_eng}: rho={rho:+.3f}  Winkler={w:.3f}")

    rho, w1 = results[True]
    _, w0 = results[False]
    print(f"\nGATE rho>=0.60: {'PASS' if rho >= 0.60 else 'FAIL'} ({rho:+.3f})")
    print(f"GATE beats ablation: {'PASS' if w1 < w0 else 'FAIL'} ({w1:.3f} vs {w0:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
