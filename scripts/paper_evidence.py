"""Paper-level evidence for the IJPHM application/evaluation manuscript.

Computes the claims that are NOT already covered by scripts/freeze_audit.py
(which covers the frozen baseline) -- specifically the 80-cell coverage match
and the fold-level separation of the conditional-scale effect.

Every number here is derived from frozen artefacts only:
  docs/tables/mcd80_cells.csv        (80 cells; produced by scripts/analyze_mcd80.py)
  experiments/punet_mcd80/verify_mcd80.csv  (integrity gate, must be VERIFY_OK)

No training, no model selection, no official test set.

Usage:  python scripts/paper_evidence.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

MULTI = ["FD002", "FD004"]
SINGLE = ["FD001", "FD003"]


def _engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Fisher exact test for [[a,b],[c,d]] without scipy (hypergeometric)."""
    n1, n2 = a + b, c + d
    k, N = a + c, a + b + c + d
    if N == 0:
        return float("nan")

    def p(x: int) -> float:
        return comb(n1, x) * comb(n2, k - x) / comb(N, k)

    obs = p(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return float(sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="docs/tables/mcd80_cells.csv")
    ap.add_argument("--verify", default="experiments/punet_mcd80/verify_mcd80.csv")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    d = pd.read_csv(args.cells)
    print(f"cells: {len(d)}  seeds: {sorted(d.seed.unique())}")

    out: dict = {}

    # ---- integrity gate must have passed before any number is used ----------
    vp = Path(args.verify)
    if vp.exists():
        v = pd.read_csv(vp)
        ok = bool(pd.to_numeric(v["max_d_y_lo"], errors="coerce").fillna(0).abs().max() <= 1e-3)
        print(f"integrity gate: {len(v)} rows, max|d_y_lo| = "
              f"{pd.to_numeric(v['max_d_y_lo'], errors='coerce').abs().max():.3e} -> "
              f"{'OK' if ok else 'FAIL'}")
        out["integrity_ok"] = ok
        if not ok:
            print("REFUSING to compute claims: integrity gate failed.")
            return 1
    else:
        print(f"WARNING: {vp} not found; run scripts/verify_mcd80.py first")
        out["integrity_ok"] = None

    # ---- C1: CQR vs constant width at the nominal level --------------------
    rel = (d.cqr_nom - d.const_nom) / d.const_nom * 100
    fold_rel = rel.groupby([d.subset, d.fold]).mean()
    print("\n[C1] CQR vs constant width, NOMINAL level (delta=0.1)")
    print(f"  cells: wins {int((rel<0).sum())}/{len(rel)}  mean {rel.mean():+.2f}%  median {rel.median():+.2f}%")
    print(f"  folds: wins {int((fold_rel<0).sum())}/{len(fold_rel)}  mean {fold_rel.mean():+.2f}%  "
          f"median {fold_rel.median():+.2f}%")
    out["C1"] = dict(cells_wins=int((rel < 0).sum()), cells_n=int(len(rel)),
                     cells_mean=float(rel.mean()), cells_median=float(rel.median()),
                     folds_wins=int((fold_rel < 0).sum()), folds_n=int(len(fold_rel)),
                     folds_mean=float(fold_rel.mean()), folds_median=float(fold_rel.median()))

    # ---- C2: MC-Dropout vs CQR (the scale vs a learned conditional width) --
    rq = (d.mcd_nom - d.cqr_nom) / d.cqr_nom * 100
    fold_q = rq.groupby([d.subset, d.fold]).mean()
    print("\n[C2] MC-Dropout vs CQR, nominal level")
    print(f"  cells: wins {int((rq<0).sum())}/{len(rq)}  mean {rq.mean():+.2f}%  median {rq.median():+.2f}%")
    print(f"  folds: wins {int((fold_q<0).sum())}/{len(fold_q)}  mean {fold_q.mean():+.2f}%")
    out["C2"] = dict(cells_wins=int((rq < 0).sum()), cells_n=int(len(rq)),
                     cells_mean=float(rq.mean()), cells_median=float(rq.median()),
                     folds_wins=int((fold_q < 0).sum()), folds_n=int(len(fold_q)),
                     folds_mean=float(fold_q.mean()))

    # ---- C4b: per-engine coverage minimum over the 80 matrix cells ---------
    # NOTE the scope difference: freeze_audit F2 reports the minimum over the
    # 20 e2e cells (single seed 42, a separate run); here we take the minimum
    # over all 80 matrix cells (4 seeds x 5 folds).  Both are correct; the paper
    # must name which one it quotes.
    def _per_engine(root: Path):
        out = []
        for sub in ["FD001", "FD002", "FD003", "FD004"]:
            for fold in range(5):
                for seed in [42, 7, 2024, 13]:
                    p = root / "experiments/punet_matrix" / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                    if not p.exists():
                        continue
                    c = np.genfromtxt(p, delimiter=",", names=True)
                    y, lo, hi, u = c["y_true"], c["y_lo"], c["y_hi"], c["unit"]
                    cal, tst = _engine_split(u)
                    s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
                    q = float(np.quantile(s, min(1.0, 0.9 * (1 + 1 / s.size)), method="higher"))
                    cov = (y[tst] >= lo[tst] - q) & (y[tst] <= hi[tst] + q)
                    for uu in np.unique(u[tst]):
                        out.append((sub, float(cov[u[tst] == uu].mean())))
        return pd.DataFrame(out, columns=["subset", "coverage"])

    pe = _per_engine(Path("."))
    print("\n[C4b] per-engine coverage minimum over the 80 MATRIX cells (4 seeds)")
    mins = {s: float(pe[pe.subset == s].coverage.min()) for s in ["FD001", "FD002", "FD003", "FD004"]}
    for s, v in mins.items():
        print(f"  {s}: min {v:.4f}   p01 {float(pe[pe.subset==s].coverage.quantile(0.01)):.4f}")
    out["C4b"] = mins

    # ---- C3: coverage match at 80 cells ------------------------------------
    diff = (d.cqr_nom_cov - d.const_nom_cov) * 100
    print("\n[C3] realised coverage at the nominal level, CQR minus constant width")
    rows = {}
    for s, g in d.groupby("subset"):
        dd = (g.cqr_nom_cov - g.const_nom_cov) * 100
        rows[s] = dict(cqr=float(g.cqr_nom_cov.mean()), const=float(g.const_nom_cov.mean()),
                       diff_pp=float(dd.mean()), max_abs_pp=float(dd.abs().max()))
        print(f"  {s}: cqr {rows[s]['cqr']:.4f}  const {rows[s]['const']:.4f}  "
              f"diff {rows[s]['diff_pp']:+.2f}pp  |max| {rows[s]['max_abs_pp']:.2f}pp")
    print(f"  ALL: cqr {d.cqr_nom_cov.mean():.4f}  const {d.const_nom_cov.mean():.4f}  "
          f"diff {diff.mean():+.2f}pp  |max| {diff.abs().max():.2f}pp")
    out["C3"] = dict(per_subset=rows, cqr=float(d.cqr_nom_cov.mean()),
                     const=float(d.const_nom_cov.mean()), diff_pp=float(diff.mean()),
                     max_abs_pp=float(diff.abs().max()))

    # ---- C6/C7: conditional-scale effect -----------------------------------
    r = (d.mcd_nom - d.const_mc_nom) / d.const_mc_nom * 100
    d = d.assign(r_scale=r)
    print("\n[C6] MC-Dropout scale vs SAME-CENTRE constant width (isolates the scale)")
    print(f"  cells: wins {int((r<0).sum())}/{len(r)}  mean {r.mean():+.2f}%  median {r.median():+.2f}%")
    per_sub = {}
    for s, g in d.groupby("subset"):
        per_sub[s] = dict(wins=int((g.r_scale < 0).sum()), n=int(len(g)),
                          mean=float(g.r_scale.mean()), median=float(g.r_scale.median()))
        print(f"  {s}: wins {per_sub[s]['wins']}/{per_sub[s]['n']}  mean {per_sub[s]['mean']:+.2f}%  "
              f"median {per_sub[s]['median']:+.2f}%")
    out["C6"] = dict(cells_wins=int((r < 0).sum()), cells_n=int(len(r)),
                     mean=float(r.mean()), median=float(r.median()), per_subset=per_sub)

    # ---- C8: fold-level separation by operating-condition structure --------
    fold = d.groupby(["subset", "fold"]).r_scale.mean().reset_index()
    multi = fold[fold.subset.isin(MULTI)].r_scale
    single = fold[fold.subset.isin(SINGLE)].r_scale
    a, b = int((multi < 0).sum()), int((multi >= 0).sum())
    c, dd = int((single < 0).sum()), int((single >= 0).sum())
    p = exact_two_sided(a, b, c, dd)
    print("\n[C8] fold-level separation by operating-condition structure (DESCRIPTIVE)")
    print(f"  multi-condition folds  ({'+'.join(MULTI)}): {a}/{a+b} better, mean {multi.mean():+.2f}%")
    print(f"  single-condition folds ({'+'.join(SINGLE)}): {c}/{c+dd} better, mean {single.mean():+.2f}%")
    print(f"  Fisher exact two-sided p = {p:.4f}   (folds share training data; not an independent-sample test)")
    out["C8"] = dict(multi_better=a, multi_n=a + b, multi_mean=float(multi.mean()),
                     single_better=c, single_n=c + dd, single_mean=float(single.mean()),
                     fisher_p=p, note="descriptive; folds share training data")

    # ---- C12: prior-uncertainty (sigma_T) permutation ----------------------
    print("\n[C12] see scripts/freeze_audit.py E1-E3 (A4 permutation: +0.246%, 11/20 worse)")

    print("\nAll numbers above are derived from frozen artefacts; see docs/VENUE_FIT.md for claim mapping.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- assertions: the manuscript numbers must not drift -----------------
    checks = [
        ("C1 cells 75/80", out["C1"]["cells_wins"] == 75),
        ("C1 folds 20/20", out["C1"]["folds_wins"] == 20),
        ("C1 mean -10.22%", abs(out["C1"]["cells_mean"] + 10.22) < 0.05),
        ("C2 folds 0/20", out["C2"]["folds_wins"] == 0),
        ("C4b FD001 0.4214", abs(out["C4b"]["FD001"] - 0.4214) < 0.002),
        ("C4b FD002 0.2375", abs(out["C4b"]["FD002"] - 0.2375) < 0.002),
        ("C4b FD003 0.4933", abs(out["C4b"]["FD003"] - 0.4933) < 0.002),
        ("C4b FD004 0.1217", abs(out["C4b"]["FD004"] - 0.1217) < 0.002),
        ("C3 diff -0.42pp", abs(out["C3"]["diff_pp"] + 0.42) < 0.05),
        ("C3 max|cell| 8.84pp", abs(out["C3"]["max_abs_pp"] - 8.84) < 0.05),
        ("C6 43/80", out["C6"]["cells_wins"] == 43),        ("C6 mean +9.49%", abs(out["C6"]["mean"] - 9.49) < 0.10),
        ("C6 FD002 19/20", out["C6"]["per_subset"]["FD002"]["wins"] == 19),
        ("C6 FD003 1/20", out["C6"]["per_subset"]["FD003"]["wins"] == 1),
        ("C8 multi 9/10", out["C8"]["multi_better"] == 9),
        ("C8 single 2/10", out["C8"]["single_better"] == 2),
        ("C8 fisher p<0.01", out["C8"]["fisher_p"] < 0.01),
    ]
    bad = [n for n, ok in checks if not ok]
    print("\n" + " | ".join(f"{n}:{'PASS' if ok else 'FAIL'}" for n, ok in checks))
    print(f"{len(checks)-len(bad)}/{len(checks)} assertions passed")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
