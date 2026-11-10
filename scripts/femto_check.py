"""Independent re-check of the FEMTO cross-domain claim (ledger C11: 5/6, -12.21%).

Two protocols are computed, because scripts/evaluate_femto.py scores each point on
its frontier with alpha = delta (the Winkler penalty drifts with the quantile
level), while the main C-MAPSS analysis was corrected under D005 to hold
alpha = 0.1 fixed. If the FEMTO number only survives the drifting-alpha version,
C11 is not comparable with the main results.

  protocol A (as in evaluate_femto.py): winkler(..., alpha=delta)
  protocol B (D005-corrected)          : winkler(..., alpha=0.1)

Each method is interpolated to realised coverage 0.90 on its own frontier,
calibrated on the bearing's `cal` rows and scored on its `test` rows.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
TARGET = 0.90


def winkler(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q_at(vals, dl):
    return float(np.quantile(vals, min(1.0, (1 - dl) * (1 + 1 / vals.size)), method="higher"))


def frontier_cqr(y, lo, hi, cal, tst, fixed_alpha):
    out = []
    s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    for dl in DELTAS:
        q = q_at(s, dl)
        L, H = lo[tst] - q, hi[tst] + q
        a = 0.1 if fixed_alpha else dl
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, a)))
    return sorted(out)


def frontier_const(y, mu, cal, tst, fixed_alpha):
    out = []
    s = np.abs(y[cal] - mu[cal])
    for dl in DELTAS:
        q = q_at(s, dl)
        L, H = mu[tst] - q, mu[tst] + q
        a = 0.1 if fixed_alpha else dl
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, a)))
    return sorted(out)


def at(pts, target=TARGET):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0), True
    return min(pts, key=lambda p: abs(p[0] - target))[1], False


def main() -> int:
    rows = []
    for d in sorted(glob.glob("experiments/femto/Bearing*")):
        pq, pp = os.path.join(d, "preds_q.csv"), os.path.join(d, "preds_p.csv")
        if not (os.path.exists(pq) and os.path.exists(pp)):
            continue
        q, p = pd.read_csv(pq), pd.read_csv(pp)
        cq, tq = (q["split"] == "cal").to_numpy(), (q["split"] == "test").to_numpy()
        cp, tp = (p["split"] == "cal").to_numpy(), (p["split"] == "test").to_numpy()
        yq, lo, hi = q["y"].to_numpy(), q["lo"].to_numpy(), q["hi"].to_numpy()
        yp, mu = p["y"].to_numpy(), p["mu"].to_numpy()
        rec = dict(bearing=os.path.basename(d), n_test=int(tq.sum()))
        for tag, fixed in [("A", False), ("B", True)]:
            c, ok1 = at(frontier_cqr(yq, lo, hi, cq, tq, fixed))
            k, ok2 = at(frontier_const(yp, mu, cp, tp, fixed))
            rec[f"cqr_{tag}"], rec[f"const_{tag}"] = c, k
            rec[f"rel_{tag}"] = (c - k) / k * 100
            rec[f"ok_{tag}"] = bool(ok1 and ok2)
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    print()
    for tag, label in [("A", "alpha=delta (evaluate_femto.py)"),
                       ("B", "alpha fixed 0.1 (D005-corrected)")]:
        rel = df[f"rel_{tag}"]
        print(f"protocol {tag} [{label}]: wins {int((rel<0).sum())}/{len(rel)} "
              f"mean {rel.mean():+.2f}% median {rel.median():+.2f}% "
              f"unbracketed {int((~df[f'ok_{tag}']).sum())}")
    df.to_csv("docs/tables/femto_check.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
