"""FD001 aggregation built on the verified ORDER_03 helpers (no training)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_order03 import ROOT, OUT, cell_metrics, load_frames, paired_bootstrap  # noqa: E402

SUBSET = "FD001"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    tags = ["M0_raw_tf", "M2_pop", "M3_cond", "M4_racdr", "M4b_popgate", "M4c_noid"]
    cells = []
    for tag in tags:
        base = os.path.join(ROOT, "experiments", "order03", tag, SUBSET)
        if not os.path.isdir(base):
            continue
        for cell in sorted(os.listdir(base)):
            if not os.path.isdir(os.path.join(base, cell)):
                continue
            try:
                seed = int(cell.rsplit("_s", 1)[1])
            except (IndexError, ValueError):
                continue
            cells.append(cell_metrics(tag, SUBSET, cell, seed))

    rows = []
    for c in cells:
        if "matched" not in c:
            continue
        r = dict(tag=c["tag"], cell=c["cell"], seed=c["seed"], variant=c.get("variant"),
                 params=c.get("params"), best_epoch=c.get("best_epoch"), epochs=c.get("epochs_ran"),
                 val=c.get("best_val_rmse"))
        for kind in ("matched", "protocol_c", "protocol_b"):
            for k in ("rmse", "macro_rmse", "late_fraction", "severe_late_gt10", "nasa", "n"):
                if kind in c and k in c[kind]:
                    r[f"{kind}_{k}"] = c[kind][k]
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["tag", "seed"])
    df.to_csv(os.path.join(OUT, "fd001_cells.csv"), index=False)

    group = []
    for tag in sorted(df.tag.unique()):
        g = df[df.tag == tag]
        row = dict(tag=tag, n_seeds=int(len(g)))
        for col in ("matched_rmse", "matched_late_fraction", "matched_severe_late_gt10",
                    "protocol_b_rmse", "protocol_b_macro_rmse",
                    "protocol_c_rmse", "protocol_c_macro_rmse"):
            xs = g[col].dropna().tolist()
            if xs:
                row[col + "_mean"] = float(np.mean(xs))
                row[col + "_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
                row[col + "_vals"] = [round(float(x), 4) for x in xs]
        group.append(row)
    pd.DataFrame(group).to_csv(os.path.join(OUT, "fd001_group.csv"), index=False)

    frames = load_frames([t for t in tags if t in set(df.tag)], SUBSET)
    paired = []
    for kind in ("matched", "protocol_c", "protocol_b"):
        if kind not in frames:
            continue
        d = frames[kind]
        for a_tag, b_tag in (("M4_racdr", "M0_raw_tf"), ("M4_racdr", "M3_cond"),
                             ("M3_cond", "M0_raw_tf"), ("M4b_popgate", "M4_racdr"),
                             ("M4c_noid", "M4_racdr")):
            if a_tag not in set(d.tag) or b_tag not in set(d.tag):
                continue
            r = paired_bootstrap(d[d.tag == a_tag], d[d.tag == b_tag], kind)
            r.update(a_tag=a_tag, b_tag=b_tag, subset=SUBSET)
            paired.append(r)

    # duplicate check: M4 vs M4c per seed
    dup = []
    mm = df[df.tag == "M4_racdr"][["seed", "matched_rmse", "val"]].rename(columns={"matched_rmse": "m4", "val": "v4"})
    cc = df[df.tag == "M4c_noid"][["seed", "matched_rmse", "val"]].rename(columns={"matched_rmse": "m4c", "val": "v4c"})
    j = mm.merge(cc, on="seed", how="inner")
    for _, r in j.iterrows():
        dup.append(dict(seed=int(r.seed), m4=float(r.m4), m4c=float(r.m4c),
                        equal=bool(abs(r.m4 - r.m4c) < 1e-12 and abs(r.v4 - r.v4c) < 1e-12)))

    out = dict(subset=SUBSET, cells=cells, group=group, paired=paired,
               duplicate_check=dup, n_val_units=int(df.matched_n.max()) if "matched_n" in df else None,
               generated_by="results/dsh/analysis/aggregate_fd001.py")
    with open(os.path.join(OUT, "fd001_aggregate.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("== FD001 cells ==")
    for r in rows:
        print(f"{r['tag']:12s} s{r['seed']} matched={r.get('matched_rmse', float('nan')):.4f} "
              f"late={r.get('matched_late_fraction', float('nan')):.3f} "
              f"sev={r.get('matched_severe_late_gt10', float('nan')):.3f} "
              f"protoB={r.get('protocol_b_rmse', float('nan')):.4f} "
              f"protoC={r.get('protocol_c_rmse', float('nan')):.4f}")
    print("\n== FD001 group ==")
    for r in group:
        print(f"{r['tag']:12s} n={r['n_seeds']} matched={r['matched_rmse_mean']:.4f}+-{r['matched_rmse_sd']:.4f} "
              f"late={r['matched_late_fraction_mean']:.3f} sev={r['matched_severe_late_gt10_mean']:.3f} "
              f"protoB={r['protocol_b_rmse_mean']:.4f} protoC={r['protocol_c_rmse_mean']:.4f}")
    print("\n== FD001 paired ==")
    for r in paired:
        print(f"{r['kind']:11s} {r['a_tag']:10s}-{r['b_tag']:10s} diff={r['rmse_diff']:+.4f} "
              f"CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] excl0={r['excludes_zero']} "
              f"p={r['prob_a_better']:.3f} n_eng={r['n_engines']}")
    print("\n== duplicate check ==")
    for d in dup:
        print(d)
    print(f"wrote {OUT}/fd001_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
