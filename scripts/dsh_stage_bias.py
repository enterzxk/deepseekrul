"""Is prediction bias stage-dependent, using only observable stage proxies?

Signed error = y_pred - y_true (>0 means late, i.e. optimistic). If bias varies
systematically with the observation stage (current cycle / window length), a
stage-conditional correction that uses ONLY observable inputs could reduce both
RMSE and late risk. This measures that structure with engine-clustered intervals.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = "experiments/order03"
CELLS = ["M1_global", "M6_tfA", "M4_racdr", "M0_raw_tf"]
OUT = Path("results/dsh/review_r1")

def load(cell, subset="FD002"):
    fs = glob.glob(f"{ROOT}/{cell}/{subset}/*/protocol_c_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["cell"] = cell
    return d

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in CELLS:
        d = load(cell)
        if d is None:
            continue
        for cut, g in d.groupby("abs_cut"):
            # engine-clustered mean signed error
            per_eng = g.groupby("unit_id").signed_error.mean()
            rng = np.random.default_rng(20261022)
            v = per_eng.to_numpy()
            boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)])
            rows.append({"cell": cell, "abs_cut": int(cut), "engines": len(per_eng),
                         "bias": float(v.mean()), "ci_low": float(np.percentile(boot, 2.5)),
                         "ci_high": float(np.percentile(boot, 97.5)),
                         "late_frac": float((g.signed_error > 0).mean()),
                         "rmse": float(np.sqrt((g.signed_error ** 2).mean())),
                         "mae": float(g.signed_error.abs().mean())})
    df = pd.DataFrame(rows).sort_values(["cell", "abs_cut"])
    df.to_csv(OUT / "stage_bias.csv", index=False)
    print(df.round(3).to_string(index=False))
    print()
    piv = df.pivot(index="abs_cut", columns="cell", values="bias").round(2)
    print("bias by cutoff (rows=abs_cut):")
    print(piv.to_string())

if __name__ == "__main__":
    main()
