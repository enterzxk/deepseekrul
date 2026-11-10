"""Frozen-baseline audit: re-derive every number in the frozen evidence set.

D016 item 2 (BASELINE_FREEZE.md) requires that each published number be
traceable to code + config + split + seed + prediction file + metric
implementation, and that its recomputability class be stated honestly.
This script is the executable half of that contract: it reads only local
artefacts and recomputes each headline number from source.

Nothing here selects a model or touches the official test labels of a new
method; it only re-scores saved predictions with the frozen metric code.

Usage:
    python scripts/freeze_audit.py [--root .] [--json out.json]

Exit code 0 iff every assertion passes.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]
FOLDS = list(range(5))

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


# --------------------------------------------------------------------------
# Frozen metric / protocol implementations (copied verbatim from
# scripts/table_all_methods.py, the code that produced docs/tables/cells_all.csv)
# --------------------------------------------------------------------------
def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def pts_scaled(y, mu, w, cal, tst):
    out = []
    for dl in DELTAS:
        q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                              min(1.0, (1 - dl) * (1 + 1 / cal.sum())), method="higher"))
        L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def pts_cqr(y, lo, hi, cal, tst):
    out = []
    for dl in DELTAS:
        s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        q = float(np.quantile(s, min(1.0, (1 - dl) * (1 + 1 / s.size)), method="higher"))
        L, H = lo[tst] - q, hi[tst] + q
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return min(pts, key=lambda p: abs(p[0] - target))[1]


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    matrix = root / "experiments/punet_matrix"
    e2e = root / "experiments/punet_e2e"
    femto = root / "experiments/femto"

    # ---- A. inventory -----------------------------------------------------
    n_cqr = len(list(matrix.glob("*/f*_s*/preds_cqr.csv")))
    n_pt = len(list(matrix.glob("*/f*_s*/preds_point.csv")))
    n_mcd = len(list(matrix.glob("*/f*_s*/preds_mcd.csv")))
    n_e2e = len(list(e2e.glob("*/f*_s42/*.json")))
    n_femto = len(list(femto.glob("*/preds_q.csv")))
    check("A1 80 matrix CQR preds", n_cqr == 80, str(n_cqr))
    check("A2 80 matrix point preds", n_pt == 80, str(n_pt))
    check("A3 10 MC-Dropout preds (partial)", n_mcd == 10, str(n_mcd))
    check("A4 80 e2e JSONs", n_e2e == 80, str(n_e2e))
    check("A5 6 FEMTO bearings", n_femto == 6, str(n_femto))

    # ---- B. 80-cell table: CQR vs constant width, coverage-matched -------
    rows = []
    for sub in SUBSETS:
        for seed in SEEDS:
            for fold in FOLDS:
                d = matrix / sub / f"f{fold}_s{seed}"
                c = np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True)
                p = np.genfromtxt(d / "preds_point.csv", delimiter=",", names=True)
                cal, tst = engine_split(c["unit"])
                mu = (p["y_lo"] + p["y_hi"]) / 2.0
                rows.append(dict(
                    subset=sub, seed=seed, fold=fold,
                    const=at(pts_scaled(p["y_true"], mu, np.ones_like(mu), cal, tst)),
                    cqr=at(pts_cqr(c["y_true"], c["y_lo"], c["y_hi"], cal, tst)),
                ))
    df = pd.DataFrame(rows)
    rel = (df["cqr"] - df["const"]) / df["const"] * 100
    check("B1 CQR wins 75/80", int((rel < 0).sum()) == 75, f"{int((rel<0).sum())}/{len(rel)}")
    check("B2 mean -9.09%", abs(rel.mean() + 9.09) < 0.05, f"{rel.mean():+.3f}%")
    check("B3 median -10.75%", abs(rel.median() + 10.75) < 0.05, f"{rel.median():+.3f}%")
    for sub, want in {"FD001": -7.57, "FD002": -14.07, "FD003": -6.49, "FD004": -8.22}.items():
        got = rel[df.subset == sub].mean()
        check(f"B4 {sub} {want:+.2f}%", abs(got - want) < 0.05, f"{got:+.3f}%")

    # compare against the committed evidence table
    cells = pd.read_csv(root / "docs/tables/cells_all.csv")
    m = cells.merge(df, on=["subset", "seed", "fold"], suffixes=("_ev", "_re"))
    d_cqr = float(np.max(np.abs(m["cqr_ev"] - m["cqr_re"])))
    d_con = float(np.max(np.abs(m["const_ev"] - m["const_re"])))
    check("B5 cells_all.csv cqr column reproduced", d_cqr < 1e-6, f"max|diff|={d_cqr:.2e}")
    check("B6 cells_all.csv const column reproduced", d_con < 1e-6, f"max|diff|={d_con:.2e}")

    # ---- C. fold level (seed-averaged, n=20) -----------------------------
    df["rel"] = rel
    fold = df.groupby(["subset", "fold"])["rel"].mean()
    check("C1 fold-level 19/20", int((fold < 0).sum()) == 19, f"{int((fold<0).sum())}/{len(fold)}")
    check("C2 fold mean -9.09%", abs(fold.mean() + 9.09) < 0.05, f"{fold.mean():+.3f}%")
    check("C3 fold median -9.69%", abs(fold.median() + 9.69) < 0.05, f"{fold.median():+.3f}%")

    # seed variance decomposition (one-way ANOVA on (subset,fold) groups)
    grand = df["rel"].mean()
    ss_tot = float(((df["rel"] - grand) ** 2).sum())
    ss_within = float(df.groupby(["subset", "fold"])["rel"].transform(lambda x: (x - x.mean()) ** 2).sum())
    share_within = ss_within / ss_tot * 100
    check("C4 seed share of variance 46.8%", abs(share_within - 46.8) < 0.5, f"{share_within:.2f}%")
    rng_per_fold = df.groupby(["subset", "fold"])["rel"].agg(lambda x: x.max() - x.min())
    check("C5 within-fold seed range median 6.13pp", abs(rng_per_fold.median() - 6.13) < 0.05,
          f"{rng_per_fold.median():.2f}pp")
    check("C6 within-fold seed range max 36.63pp", abs(rng_per_fold.max() - 36.63) < 0.05,
          f"{rng_per_fold.max():.2f}pp")
    same = df.groupby(["subset", "fold"])["rel"].agg(lambda x: bool((x < 0).all() or (x > 0).all()))
    check("C7 16/20 folds all four seeds same sign", int(same.sum()) == 16, f"{int(same.sum())}/20")

    # ---- D. e2e nominal level (single seed 42, 20 cells) ----------------
    def load(s, f, name):
        p = e2e / s / f"f{f}_s42" / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    e2e_rel, a4_rel, eng = [], [], []
    for s in SUBSETS:
        for f in FOLDS:
            a, b = load(s, f, "e2e_cqr_none"), load(s, f, "e2e_point_none")
            if a and b:
                e2e_rel.append((s, f, (a["winkler"] - b["winkler"]) / b["winkler"] * 100))
            p1 = load(s, f, "e2e_cqr_A4")
            if p1 and a:
                a4_rel.append((s, f, (p1["winkler"] - a["winkler"]) / a["winkler"] * 100))
            if a:
                eng.append((s, f, a["picp"], a["engine_min"], a["rmse"]))
    ev = np.array([x[2] for x in e2e_rel])
    check("D1 e2e 19/20", int((ev < 0).sum()) == 19, f"{int((ev<0).sum())}/{len(ev)}")
    check("D2 e2e median -8.68%", abs(st.median(ev) + 8.68) < 0.05, f"{st.median(ev):+.3f}%")
    # Per-subset e2e numbers in docs/audit_closeout.md are RATIO OF MEANS
    # (subset-aggregate Winkler), not the mean of per-cell ratios.  The two
    # differ by up to 0.5 pp; both are recorded here so the 口径 is explicit.
    for sub, want in {"FD001": -5.25, "FD002": -16.21, "FD003": -8.83, "FD004": -6.99}.items():
        c = [load(sub, f, "e2e_cqr_none")["winkler"] for f in FOLDS]
        p = [load(sub, f, "e2e_point_none")["winkler"] for f in FOLDS]
        got = (np.mean(c) - np.mean(p)) / np.mean(p) * 100
        check(f"D3 {sub} e2e ratio-of-means {want:+.2f}%", abs(got - want) < 0.05, f"{got:+.3f}%")
    check("D4 e2e mean-of-ratios mean -9.31%",
          abs(float(np.mean(ev)) + 9.31) < 0.05, f"{float(np.mean(ev)):+.3f}%")

    # ---- E. A4 permutation -------------------------------------------------
    a4 = np.array([x[2] for x in a4_rel])
    check("E1 A4 mean +0.25%", abs(a4.mean() - 0.25) < 0.05, f"{a4.mean():+.3f}%")
    check("E2 A4 11/20 folds worse", int((a4 > 0).sum()) == 11, f"{int((a4>0).sum())}/{len(a4)}")
    for sub, want in {"FD001": 2.54, "FD002": 0.24, "FD003": 1.22, "FD004": -3.02}.items():
        got = float(np.mean([x[2] for x in a4_rel if x[0] == sub]))
        check(f"E3 {sub} A4 {want:+.2f}%", abs(got - want) < 0.05, f"{got:+.3f}%")

    # ---- F. engine-level failure ------------------------------------------
    emin = np.array([x[3] for x in eng])
    check("F1 18/20 folds engine_min<0.80", int((emin < 0.80).sum()) == 18, f"{int((emin<0.80).sum())}/20")
    # TRUE per-subset minimum single-engine coverage (min over folds of the
    # per-fold engine minimum).  This is worse than the D018 table suggests.
    for sub, want in {"FD001": 0.4214, "FD002": 0.4500, "FD003": 0.5521, "FD004": 0.2814}.items():
        got = float(np.min([x[3] for x in eng if x[0] == sub]))
        check(f"F2 {sub} TRUE engine_min {want:.4f}", abs(got - want) < 0.0015, f"{got:.4f}")
    # D018 / failure_inventory.csv column "engine_min" is the MEAN over folds of
    # the per-fold engine minimum, not the minimum.  Verify that reading so the
    # mislabel is pinned down rather than silently repeated.
    for sub, want in {"FD001": 0.690, "FD002": 0.529, "FD003": 0.714, "FD004": 0.501}.items():
        got = float(np.mean([x[3] for x in eng if x[0] == sub]))
        check(f"F2b {sub} fold-mean engine_min {want:.3f}", abs(got - want) < 0.0015, f"{got:.4f}")
    worst = min(eng, key=lambda x: x[3])
    check("F3 worst = FD004 f2 0.2814", worst[0] == "FD004" and worst[1] == 2 and abs(worst[3] - 0.28137) < 1e-4,
          f"{worst[0]} f{worst[1]} {worst[3]:.4f}")
    picp = np.array([x[2] for x in eng])
    rmse = np.array([x[4] for x in eng])
    check("F4 rho(picp, engine_min)=+0.725", abs(np.corrcoef(picp, emin)[0, 1] - 0.725) < 0.01,
          f"{np.corrcoef(picp, emin)[0, 1]:+.3f}")
    check("F5 rho(rmse, engine_min)=-0.783", abs(np.corrcoef(rmse, emin)[0, 1] + 0.783) < 0.01,
          f"{np.corrcoef(rmse, emin)[0, 1]:+.3f}")

    # ---- G. FEMTO ----------------------------------------------------------
    # 2026-11-10 correction: the previous version passed only split=="test" rows
    # into pts_cqr/pts_scaled, which derive q_hat from the rows they are given --
    # i.e. it calibrated on the same rows it scored (self-calibration leak), and
    # the two assertions locked in the leaked numbers (5/6, -12.2%).
    # Now: calibrate on split=="cal", score on split=="test", alpha fixed at 0.1
    # (D005), and report how many bearings can actually reach 0.90 coverage.
    sys.path.insert(0, str(root / "scripts"))
    from femto_check import at as f_at, frontier_cqr as f_cqr, frontier_const as f_const  # noqa: E402

    frel, f_ok = [], 0
    for d0 in sorted(femto.glob("Bearing*")):
        q, p = d0 / "preds_q.csv", d0 / "preds_p.csv"
        if not (q.exists() and p.exists()):
            continue
        qq, pp = pd.read_csv(q), pd.read_csv(p)
        cq, tq = (qq.split == "cal").to_numpy(), (qq.split == "test").to_numpy()
        cp, tp = (pp.split == "cal").to_numpy(), (pp.split == "test").to_numpy()
        c, ok1 = f_at(f_cqr(qq.y.values, qq.lo.values, qq.hi.values, cq, tq, fixed_alpha=True))
        k, ok2 = f_at(f_const(pp.y.values, pp.mu.values, cp, tp, fixed_alpha=True))
        frel.append((c - k) / k * 100)
        f_ok += int(ok1 and ok2)
    check("G1 FEMTO 6/6 better (held-out calibration)", int(sum(1 for x in frel if x < 0)) == 6,
          f"{int(sum(1 for x in frel if x < 0))}/{len(frel)}")
    check("G2 FEMTO mean -16.3%", abs(np.mean(frel) + 16.3) < 0.2,
          f"{np.mean(frel):+.2f}% | 0.90 reachable on {f_ok}/6 bearings")

    # ---- H. split identity: analysis-time split reproduces run-time q_hat ---
    # run_e2e_cqr.py calibrates with engine_calib_split(seed=args.seed); the
    # analysis (table_all_methods.py) re-splits with engine_split(seed=42).
    # For the seed-42 e2e cells the two must coincide, which is testable: the
    # recomputed q_hat must equal the one stored in the run-time JSON.
    for sub, fold in [("FD001", 0), ("FD002", 0)]:
        j = load(sub, fold, "e2e_cqr_none")
        p = pd.read_csv(e2e / sub / f"f{fold}_s42" / "preds.csv")
        cal, _ = engine_split(p["unit"].to_numpy())
        s = np.maximum(p["y_lo"].to_numpy()[cal] - p["y_true"].to_numpy()[cal],
                       p["y_true"].to_numpy()[cal] - p["y_hi"].to_numpy()[cal])
        lvl = min(1.0, (1.0 - 0.1) * (1.0 + 1.0 / s.size))
        q = float(np.quantile(s, lvl, method="higher"))
        check(f"H1 {sub} f{fold} run-time q_hat reproduced", abs(q - j["q_hat"]) < 1e-6,
              f"recomputed {q:.6f} vs json {j['q_hat']:.6f}")

    # ---- report ------------------------------------------------------------
    ok = sum(1 for _, v, _ in results if v)
    print(f"{'CHECK':<46} RESULT   DETAIL")
    for name, v, detail in results:
        print(f"{name:<46} {'PASS' if v else 'FAIL':<8} {detail}")
    print(f"\n{ok}/{len(results)} checks passed")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"checks": [{"name": n, "pass": v, "detail": d} for n, v, d in results],
             "passed": ok, "total": len(results)}, indent=1), encoding="utf-8")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
