"""Same-baseline comparison: E's scale applied to the frozen CQR base interval.

preds_cqr.csv stores the FINAL intervals lo = lo0 - q, hi = hi0 + q.  The run JSON
stores q_hat, so the base interval is recovered as lo0 = lo + q_hat, hi0 = hi - q_hat.
E's arm then expands that same base by q_E * scale instead of q_hat.

Arms
  frozen   : lo0 - q_hat,        hi0 + q_hat          (the paper's baseline)
  window   : lo0 - q_w * s_win,  hi0 + q_w * s_win    (ablation, no engine head)
  engine   : lo0 - q_e * s,      hi0 + q_e * s        (E)

q for each arm is calibrated on the 30% calibration engines with the engine_split(42)
convention, and evaluated on the remaining engines.

Run on a40: python scripts/run_same_baseline.py FD001,FD002
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_ida as R  # noqa: E402
from src.candidates.model import CandidateModel  # noqa: E402
from src.punet.conformal import engine_calib_split
from src.twoscale.conformal import conformal_quantile, winkler  # noqa: E402
from src.twoscale.train import fit_two_scale  # noqa: E402


def load_body(subset, fold):
    ck = ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42" / "model.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
    body = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    body.load_state_dict(body_sd, strict=True)
    body.eval()
    return body


def capture(body, x, e, chunk=2048):
    hs, ys = [], []
    for i in range(0, x.shape[0], chunk):
        xb, eb = x[i:i + chunk], e[i:i + chunk]
        grabbed = {}

        def hook(_m, _i, out):
            grabbed["h"] = out[0] if isinstance(out, (tuple, list)) else out

        handle = body.norm.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = body(xb, None, eb)
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
    subs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["FD001", "FD002"]
    folds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(5))
    rows = []
    for subset in subs:
        for fold in folds:
            d = ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42"
            preds = pd.read_csv(d / "preds_cqr.csv")
            q_hat = json.loads((d / "e2e_cqr_none.json").read_text())["q_hat"]
            body = load_body(subset, fold)
            splits, frame = R.outer_folds(subset, 5)
            tr_u, va_u = splits[fold]
            (xtr, ytr, etr, mtr, _), (xva, yva, eva, mva, _) = R.prep_fold(
                frame, subset, fold, tr_u, va_u, window=50, max_rul=125.0)
            xt = torch.as_tensor(np.asarray(xtr), dtype=torch.float32)
            xv = torch.as_tensor(np.asarray(xva), dtype=torch.float32)
            et = torch.as_tensor(np.asarray(etr), dtype=torch.float32)
            ev = torch.as_tensor(np.asarray(eva), dtype=torch.float32)
            h_tr, y_tr = capture(body, xt, et)
            h_va, y_va = capture(body, xv, ev)

            m_tr, m_va = np.asarray(mtr), np.asarray(mva)
            ids_tr = np.unique(m_tr[:, 0], return_inverse=True)[1].astype(np.int64)
            ids_va = np.unique(m_va[:, 0], return_inverse=True)[1].astype(np.int64)
            n_tr, n_va = ids_tr.max() + 1, ids_va.max() + 1
            y = preds.y_true.to_numpy()
            lo_f = preds.y_lo.to_numpy()
            hi_f = preds.y_hi.to_numpy()
            lo0, hi0 = lo_f + q_hat, hi_f - q_hat

            cal_idx, tst_idx = engine_calib_split(m_va[:, 0], calib_frac=0.3, seed=42)
            cal = np.zeros(len(y), bool)
            cal[cal_idx] = True
            tst = np.zeros(len(y), bool)
            tst[tst_idx] = True
            w_frozen = winkler(y[tst], lo_f[tst], hi_f[tst])
            out = {"frozen": w_frozen}
            for use_eng, name in ((False, "window"), (True, "engine")):
                model, _, _ = fit_two_scale(h_tr.numpy(), ids_tr, y_tr.numpy(),
                                            np.ones_like(y_tr.numpy()), np.asarray(ytr, float),
                                            n_tr, use_engine=use_eng, epochs=120)
                with torch.no_grad():
                    w, g = model(h_va, torch.as_tensor(ids_va), n_va)
                    s = (w * g).numpy()
                score = np.maximum(lo0 - y, y - hi0) / np.clip(s, 1e-6, None)
                q = conformal_quantile(score[cal], 0.1)
                lo, hi = lo0 - q * s, hi0 + q * s
                out[name] = winkler(y[tst], lo[tst], hi[tst])
                print(f"  PICP {name} {np.mean((y[tst] >= lo[tst]) & (y[tst] <= hi[tst])):.4f}  q={q:.3f}")
            def picp(a, b, c):
                return float(np.mean((y[tst] >= a[tst]) & (y[tst] <= c[tst])))
            pc_f = picp(lo_f, lo_f, hi_f)
            print(f"  PICP frozen {pc_f:.4f}")
            delta = (out["engine"] - out["frozen"]) / out["frozen"] * 100
            rows.append((subset, fold, out["frozen"], out["window"], out["engine"], delta))
            print(f"{subset} f{fold}: frozen {out['frozen']:.2f} | win {out['window']:.2f} | "
                  f"eng {out['engine']:.2f}  vs frozen {delta:+.2f}%")

    a = np.array([[r[2], r[4], r[5]] for r in rows], float)
    print(f"\nmean frozen {a[:,0].mean():.2f}  mean eng {a[:,1].mean():.2f}  mean delta {a[:,2].mean():+.2f}%")
    print("E beats frozen on Winkler:", "PASS" if a[:, 2].mean() < 0 else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
