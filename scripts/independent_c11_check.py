"""Independent recomputation of C11 (FEMTO cross-domain) plus a protocol check.

C11 as recorded: CQR beats constant width on 5/6 bearings, mean -12.21%.

The recorded numbers come from scripts/evaluate_femto.py, which passes only the
`test` rows to pts_cqr / pts_scaled -- those functions both *calibrate and
evaluate on the same array*.  The `cal` rows exist in the dumps but are unused.
So the published number is an in-sample conformal comparison, not a split-conformal
one.  Because both arms are treated identically the comparison remains paired, but
the absolute coverage is optimistic for both.

This script reports both:
  (A) reproduce  -- the recorded protocol (calibrate+evaluate on test rows)
  (B) proper     -- calibrate the quantile on `cal` rows, evaluate on `test` rows

If (A) reproduces and (B) keeps the same sign/rough size, the cross-domain claim is
robust to the protocol.  If (B) changes the conclusion, C11 must be re-scoped in the
manuscript.

Usage: python scripts/independent_c11_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def winkler(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q_at(vals, dl, n=None):
    n = vals.size if n is None else n
    return float(np.quantile(vals, min(1.0, (1 - dl) * (1 + 1 / n)), method="higher"))


def frontier_in_sample(y, lo, hi):
    """Recorded protocol: calibrate and evaluate on the same rows."""
    out = []
    for dl in DELTAS:
        q = q_at(np.maximum(lo - y, y - hi), dl)
        out.append((float(np.mean((y >= lo - q) & (y <= hi + q))), winkler(y, lo - q, hi + q)))
    return sorted(out)


def frontier_split(y_cal, lo_cal, hi_cal, y_tst, lo_tst, hi_tst):
    """Proper protocol: calibrate on cal rows, evaluate on test rows."""
    out = []
    s_cal = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    for dl in DELTAS:
        q = q_at(s_cal, dl)
        L, H = lo_tst - q, hi_tst + q
        out.append((float(np.mean((y_tst >= L) & (y_tst <= H))), winkler(y_tst, L, H)))
    return sorted(out)


def const_in_sample(y, mu):
    out = []
    for dl in DELTAS:
        q = q_at(np.abs(y - mu), dl)
        out.append((float(np.mean((y >= mu - q) & (y <= mu + q))), winkler(y, mu - q, mu + q)))
    return sorted(out)


def const_split(y_cal, mu_cal, y_tst, mu_tst):
    out = []
    for dl in DELTAS:
        q = q_at(np.abs(y_cal - mu_cal), dl)
        out.append((float(np.mean((y_tst >= mu_tst - q) & (y_tst <= mu_tst + q))),
                    winkler(y_tst, mu_tst - q, mu_tst + q)))
    return sorted(out)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return min(pts, key=lambda p: abs(p[0] - target))[1]


def main() -> int:
    rows = []
    for d in sorted(Path("experiments/femto").glob("Bearing*")):
        qf, pf = d / "preds_q.csv", d / "preds_p.csv"
        if not (qf.exists() and pf.exists()):
            continue
        q, p = pd.read_csv(qf), pd.read_csv(pf)
        mq_cal, mq_tst = q["split"] == "cal", q["split"] == "test"
        mp_cal, mp_tst = p["split"] == "cal", p["split"] == "test"
        yq, lo, hi = q["y"].to_numpy(), q["lo"].to_numpy(), q["hi"].to_numpy()
        yp, mu = p["y"].to_numpy(), p["mu"].to_numpy()
        rows.append(dict(
            bearing=d.name,
            # (A) recorded protocol
            A_cqr=at(frontier_in_sample(yq[mq_tst], lo[mq_tst], hi[mq_tst])),
            A_const=at(const_in_sample(yp[mp_tst], mu[mp_tst])),
            # (B) proper split
            B_cqr=at(frontier_split(yq[mq_cal], lo[mq_cal], hi[mq_cal],
                                    yq[mq_tst], lo[mq_tst], hi[mq_tst])),
            B_const=at(const_split(yp[mp_cal], mu[mp_cal], yp[mp_tst], mu[mp_tst])),
            n_cal=int(mq_cal.sum()), n_tst=int(mq_tst.sum()),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no bearings found")
        return 1
    df["A_rel"] = (df.A_cqr - df.A_const) / df.A_const * 100
    df["B_rel"] = (df.B_cqr - df.B_const) / df.B_const * 100
    print(df.round(2).to_string(index=False))
    print()
    print(f"(A) recorded protocol : wins {int((df.A_rel<0).sum())}/{len(df)}  "
          f"mean {df.A_rel.mean():+.2f}%")
    print(f"(B) proper split      : wins {int((df.B_rel<0).sum())}/{len(df)}  "
          f"mean {df.B_rel.mean():+.2f}%")
    ok = int((df.A_rel < 0).sum()) == 5 and abs(df.A_rel.mean() + 12.21) < 0.1
    print("\nC11 reproduction (A):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
