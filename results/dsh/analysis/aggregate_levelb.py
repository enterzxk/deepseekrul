"""Level B pre-registered replication aggregation (frozen runner 8fce61ff)."""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_order03 import ROOT, OUT, basic, paired_bootstrap  # noqa: E402

SUBSETS = ["FD001", "FD002"]
TAGS = ["M0_tf", "M4_racdr"]
KINDS = {
    "matched": ("matched_cutoff_predictions.csv", None),
    "protocol_c": ("protocol_c_predictions.csv", "abs_cut"),
    "protocol_b": ("pseudo_test_predictions.csv", "cutoff"),
}


def load_all() -> dict:
    frames = {k: [] for k in KINDS}
    provenance = []
    for tag in TAGS:
        for subset in SUBSETS:
            for kind, (fname, _) in KINDS.items():
                for p in sorted(glob.glob(os.path.join(ROOT, "experiments", "levelb", tag, subset, "*", fname))):
                    cell = os.path.basename(os.path.dirname(p))
                    seed = int(cell.rsplit("_s", 1)[1])
                    df = pd.read_csv(p)
                    df["tag"] = tag
                    df["dataset"] = subset
                    df["seed"] = seed
                    frames[kind].append(df)
            for m in sorted(glob.glob(os.path.join(ROOT, "experiments", "levelb", tag, subset, "*", "metrics.json"))):
                d = json.load(open(m))
                provenance.append(dict(tag=tag, subset=subset, seed=d.get("seed"),
                                       code_sha8=str(d.get("code_sha256", ""))[:8],
                                       split_seed=d.get("split_seed"),
                                       val_units=len(d.get("val_units", [])),
                                       best_epoch=d.get("best_epoch"),
                                       params=d.get("params")))
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in frames.items()}, provenance


def per_cell(frames: dict) -> pd.DataFrame:
    rows = []
    keys = []
    for subset in SUBSETS:
        for tag in TAGS:
            for seed in (2029, 2030, 2031):
                row = dict(subset=subset, tag=tag, seed=seed)
                found = False
                for kind, (_, cutcol) in KINDS.items():
                    d = frames[kind]
                    if d.empty:
                        continue
                    g = d[(d.tag == tag) & (d.dataset == subset) & (d.seed == seed)]
                    if g.empty:
                        continue
                    found = True
                    b = basic(g.signed_error.to_numpy())
                    for k, v in b.items():
                        row[f"{kind}_{k}"] = v
                    if cutcol:
                        per = g.groupby(cutcol).signed_error.apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
                        row[f"{kind}_macro_rmse"] = float(per.mean())
                if found:
                    rows.append(row)
    return pd.DataFrame(rows)


def band_severe(df: pd.DataFrame, lo: float = 0.0, hi: float = 25.0) -> dict:
    g = df[(df.y_true >= lo) & (df.y_true < hi)]
    if g.empty:
        return dict(n=0, rmse=float("nan"), severe_late=float("nan"))
    e = g.signed_error.to_numpy()
    return dict(n=int(len(g)), rmse=float(np.sqrt(np.mean(e ** 2))),
                severe_late=float((e > 10).mean()))


def main() -> int:
    frames, prov = load_all()
    cells = per_cell(frames)
    cells.to_csv(os.path.join(OUT, "levelb_cells.csv"), index=False)

    group = []
    for subset in SUBSETS:
        for tag in TAGS:
            g = cells[(cells.subset == subset) & (cells.tag == tag)]
            if g.empty:
                continue
            row = dict(subset=subset, tag=tag, n=len(g))
            for col in ("matched_rmse", "matched_late_fraction", "matched_severe_late_gt10",
                        "protocol_c_rmse", "protocol_c_macro_rmse", "protocol_c_late_fraction",
                        "protocol_c_severe_late_gt10", "protocol_b_rmse"):
                xs = g[col].dropna().tolist()
                if xs:
                    row[col + "_mean"] = float(np.mean(xs))
                    row[col + "_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
                    row[col + "_vals"] = [round(float(x), 4) for x in xs]
            group.append(row)
    pd.DataFrame(group).to_csv(os.path.join(OUT, "levelb_group.csv"), index=False)

    paired = []
    for subset in SUBSETS:
        for kind in ("matched", "protocol_c", "protocol_b"):
            d = frames[kind]
            if d.empty:
                continue
            a = d[(d.tag == "M4_racdr") & (d.dataset == subset)]
            b = d[(d.tag == "M0_tf") & (d.dataset == subset)]
            if a.empty or b.empty:
                continue
            r = paired_bootstrap(a, b, kind)
            r.update(subset=subset, a_tag="M4_racdr", b_tag="M0_tf")
            paired.append(r)

    # pre-registered gates
    def g(subset, tag):
        return cells[(cells.subset == subset) & (cells.tag == tag)]

    gates = {}
    for subset in SUBSETS:
        m4, m0 = g(subset, "M4_racdr"), g(subset, "M0_tf")
        if not m4.empty and not m0.empty:
            gates[f"{subset}_matched_rmse"] = dict(
                m4=float(m4.matched_rmse.mean()), m0=float(m0.matched_rmse.mean()),
                pass_=bool(m4.matched_rmse.mean() < m0.matched_rmse.mean()))
    # gate 2: FD001 [0,25) severe-late <= 1.1x tf, on matched and on protocol_c (sensitivity)
    for kind in ("matched", "protocol_b", "protocol_c"):
        d = frames[kind]
        if d.empty:
            continue
        b4 = band_severe(d[(d.tag == "M4_racdr") & (d.dataset == "FD001")])
        b0 = band_severe(d[(d.tag == "M0_tf") & (d.dataset == "FD001")])
        gates[f"FD001_band_0_25_{kind}"] = dict(
            m4=b4, tf=b0,
            ratio=(b4["severe_late"] / b0["severe_late"]) if b0["severe_late"] == b0["severe_late"] and b0["severe_late"] > 0 else None,
            pass_=bool(b4["severe_late"] <= 1.1 * b0["severe_late"]) if b0["severe_late"] == b0["severe_late"] else None)

    out = dict(level="B", subsets=SUBSETS, tags=TAGS, cells=cells.to_dict(orient="records"),
               group=group, paired=paired, gates=gates, provenance=prov,
               frozen_runner="scripts/frozen/run_candidate_v3_8fce61ff.py",
               split_seed=2027, seeds=[2029, 2030, 2031])
    with open(os.path.join(OUT, "levelb_aggregate.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== cells ==")
    for _, r in cells.sort_values(["subset", "tag", "seed"]).iterrows():
        print(f"{r.subset} {r.tag:9s} s{int(r.seed)} matched={r.matched_rmse:.4f} "
              f"late={r.matched_late_fraction:.3f} sev={r.matched_severe_late_gt10:.3f} "
              f"protoC={r.protocol_c_rmse:.4f} protoB={r.protocol_b_rmse:.4f}")
    print("\n== group ==")
    for r in group:
        print("{subset} {tag:9s} n={n} matched={matched_rmse_mean:.4f}+-{matched_rmse_sd:.4f} "
              "late={matched_late_fraction_mean:.3f} sev={matched_severe_late_gt10_mean:.3f} "
              "protoC={protocol_c_rmse_mean:.4f}".format(**r))
    print("\n== paired M4 - tf ==")
    for r in paired:
        print("{subset} {kind:11s} diff={rmse_diff:+.4f} CI=[{ci_low:+.4f},{ci_high:+.4f}] "
              "excl0={excludes_zero} p={prob_a_better:.3f}".format(**r))
    print("\n== gates ==")
    print(json.dumps(gates, indent=2))
    print("\n== provenance ==")
    for p in prov:
        print(p)
    print(f"\nwrote {OUT}/levelb_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
