"""Candidate E: full CPU evaluation over 4 subsets x 5 folds on frozen features.

For each (subset, fold):
  * frozen body -> h (hook on body.norm, mean-pooled) and the point prediction
  * fit the two scale heads on the TRAINING engines
  * split the validation engines into calibration (30%) and test (70%) with seed 42
  * calibrate the conformal quantile per arm and evaluate on the test engines
  * arms: global (no scale), window-only (ablation), engine x window (E), oracle per-engine q

Gates: rho(sigma_eng, test engine RMSE) >= 0.60 and captured >= 1/3 of the oracle gain.

Run on a40:  python scripts/run_two_scale_all.py
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
from src.twoscale.conformal import conformal_quantile, cqr_scores, intervals, winkler  # noqa: E402
from src.twoscale.train import fit_two_scale  # noqa: E402

FOLDS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(5))
SUBSETS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["FD001", "FD002", "FD003", "FD004"]


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def load_body(subset: str, fold: int) -> CandidateModel:
    ck = ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42" / "model.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
    body = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    body.load_state_dict(body_sd, strict=True)
    body.eval()
    return body


def capture(body, x, c, e, chunk=2048):
    hs, ys = [], []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        eb = e[i:i + chunk]
        cb = None if c is None else c[i:i + chunk]
        grabbed = {}

        def hook(_m, _i, out):
            grabbed["h"] = out[0] if isinstance(out, (tuple, list)) else out

        handle = body.norm.register_forward_hook(hook)
        try:
            with torch.no_grad():
                try:
                    out = body(xb, cb, eb)
                except TypeError:
                    out = body(xb)
        finally:
            handle.remove()
        y = out["prediction"] if isinstance(out, dict) else out
        h = grabbed["h"]
        if h.ndim == 3:
            h = h.mean(dim=1)
        hs.append(h.detach().float())
        ys.append(torch.as_tensor(y).reshape(-1).float())
    return torch.cat(hs, 0), torch.cat(ys, 0)


def main() -> int:
    rows = []
    for subset in SUBSETS:
        for fold in FOLDS:
            body = load_body(subset, fold)
            splits, frame = R.outer_folds(subset, 5)
            tr_u, va_u = splits[fold]
            (xtr, ytr, etr, mtr, _), (xva, yva, eva, mva, _) = R.prep_fold(
                frame, subset, fold, tr_u, va_u, window=50, max_rul=125.0)
            xt = torch.as_tensor(np.asarray(xtr), dtype=torch.float32)
            xv = torch.as_tensor(np.asarray(xva), dtype=torch.float32)
            et = torch.as_tensor(np.asarray(etr), dtype=torch.float32)
            ev = torch.as_tensor(np.asarray(eva), dtype=torch.float32)

            h_tr, y_tr = capture(body, xt, None, et)
            h_va, y_va = capture(body, xv, None, ev)
            m_tr, m_va = np.asarray(mtr), np.asarray(mva)
            ids_tr = np.unique(m_tr[:, 0], return_inverse=True)[1].astype(np.int64)
            ids_va = np.unique(m_va[:, 0], return_inverse=True)[1].astype(np.int64)
            n_tr, n_va = ids_tr.max() + 1, ids_va.max() + 1
            yt, yv = np.asarray(ytr, float), np.asarray(yva, float)
            ctr_tr, ctr_va = y_tr.numpy(), y_va.numpy()

            # calibration / test engine split inside the validation engines
            rng = np.random.default_rng(42)
            order = rng.permutation(n_va)
            n_cal = max(1, int(round(n_va * 0.3)))
            cal_e, tst_e = order[:n_cal], order[n_cal:]
            cal_mask = np.isin(ids_va, cal_e)
            tst_mask = np.isin(ids_va, tst_e)

            arms = {}
            for use_eng, name in ((False, "window"), (True, "engine")):
                model, _, _ = fit_two_scale(h_tr.numpy(), ids_tr, ctr_tr, np.ones_like(ctr_tr),
                                            yt, n_tr, use_engine=use_eng, epochs=120)
                with torch.no_grad():
                    w, g = model(h_va, torch.as_tensor(ids_va), n_va)
                    scale = (w * g).numpy()
                q = conformal_quantile(cqr_scores(yv[cal_mask], ctr_va[cal_mask], ctr_va[cal_mask],
                                                 scale=scale[cal_mask]), 0.1)
                lo, hi = intervals(ctr_va, ctr_va, q, scale=scale)
                arms[name] = (scale, q, winkler(yv[tst_mask], lo[tst_mask], hi[tst_mask]))

            # global arm: no per-window scale at all
            q_g = conformal_quantile(cqr_scores(yv[cal_mask], ctr_va[cal_mask], ctr_va[cal_mask]), 0.1)
            lo_g, hi_g = intervals(ctr_va, ctr_va, q_g)
            w_global = winkler(yv[tst_mask], lo_g[tst_mask], hi_g[tst_mask])

            # oracle: one q per test engine (upper bound on any allocation)
            w_or = 0.0
            n_t = 0
            scale_eng = arms["engine"][0]
            q_eng = arms["engine"][1]
            for e in tst_e:
                m = ids_va == e
                grid = np.linspace(0.2 * q_eng, 3.0 * q_eng, 57)
                best = min(winkler(yv[m], ctr_va[m] - qq * scale_eng[m],
                                   ctr_va[m] + qq * scale_eng[m]) for qq in grid)
                w_or += best * m.sum()
                n_t += m.sum()
            w_oracle = w_or / max(n_t, 1)

            rmse_eng = np.array([np.sqrt(np.mean((yv[ids_va == e] - ctr_va[ids_va == e]) ** 2))
                                 for e in tst_e])
            eng_vals = arms["engine"][0]
            pred_eng = np.array([eng_vals[ids_va == e][0] for e in tst_e])
            rho = spearman(pred_eng, rmse_eng) if len(tst_e) > 3 else float("nan")

            captured = (arms["engine"][2] - w_global) / (w_oracle - w_global) if abs(w_oracle - w_global) > 1e-9 else float("nan")
            rows.append((subset, fold, rho, w_global, arms["window"][2], arms["engine"][2], w_oracle, captured))
            print(f"{subset} f{fold}: rho={rho:+.3f}  W glob {w_global:.1f} | win {arms['window'][2]:.1f} | "
                  f"eng {arms['engine'][2]:.1f} | oracle {w_oracle:.1f}  captured {captured*100:+.1f}%")

    a = np.array([[r[2], r[7]] for r in rows], float)
    print(f"\nmean rho {np.nanmean(a[:,0]):+.3f}   mean captured {np.nanmean(a[:,1])*100:+.1f}%")
    print("GATE rho>=0.60:", "PASS" if np.nanmean(a[:, 0]) >= 0.60 else "FAIL")
    print("GATE captured>=1/3:", "PASS" if np.nanmean(a[:, 1]) >= 1 / 3 else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
