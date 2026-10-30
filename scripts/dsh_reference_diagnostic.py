"""Why does a per-condition healthy reference hurt (FD002, same-seed gap 3.77)?

Two candidate explanations, both testable without training:
  H-noise: the per-condition baseline is estimated from very few early samples
           (n <= 5 per condition), so it is a noisy estimate of the healthy level.
  H-signal: subtracting a per-condition reference removes degradation signal that
           the condition-level offset itself carries.

Diagnostics per engine:
  * n_samples(m): number of early samples available for each condition cluster
  * trend_strength: Spearman correlation between ||residual_t|| and cycle
  * residual variance explained by a monotone trend
Compares global-reference residual vs condition-reference residual, paired by engine.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

ROOT = Path("data/CMAPSS")
SENSORS = [f"s{i}" for i in range(1, 22)]
USE = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
OPS = ["op_1","op_2","op_3"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--out", default="results/dsh/review_r1")
    args = ap.parse_args()
    df = pd.read_csv(ROOT / f"train_{args.subset}.txt", sep=r"\s+", header=None,
                     names=["unit","cycle"] + OPS + SENSORS)
    km = KMeans(n_clusters=6, random_state=7, n_init=10).fit(df[OPS].to_numpy(float))
    rows = []
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        x = g[USE].to_numpy(float)
        cyc = g.cycle.to_numpy(int)
        cl = km.predict(g[OPS].to_numpy(float))
        early = cyc <= 30
        g_ref = x[early].mean(0)
        res_g = np.linalg.norm(x - g_ref, axis=1)
        # condition reference: per-condition mean of early samples, fallback to global
        refs, counts = {}, []
        res_c = np.empty(len(x))
        for i, m in enumerate(cl):
            if m not in refs:
                sel = early & (cl == m)
                counts.append(int(sel.sum()))
                refs[m] = x[sel].mean(0) if sel.any() else g_ref
        res_c = np.linalg.norm(x - np.stack([refs[m] for m in cl]), axis=1)
        tg = spearmanr(cyc, res_g).correlation
        tc = spearmanr(cyc, res_c).correlation
        rows.append({"unit": int(u), "n_early": int(early.sum()),
                     "min_cond_samples": int(min(counts)) if counts else 0,
                     "trend_global": float(tg), "trend_cond": float(tc),
                     "var_global": float(np.var(res_g)), "var_cond": float(np.var(res_c))})
    d = pd.DataFrame(rows)
    out = {
        "engines": len(d),
        "median_min_cond_samples": float(d.min_cond_samples.median()),
        "frac_engines_with_cond_lt3": float((d.min_cond_samples < 3).mean()),
        "trend_global_mean": float(d.trend_global.mean()),
        "trend_cond_mean": float(d.trend_cond.mean()),
        "trend_diff_mean": float((d.trend_global - d.trend_cond).mean()),
        "trend_diff_sd": float((d.trend_global - d.trend_cond).std(ddof=1)),
        "frac_global_steeper": float((d.trend_global > d.trend_cond).mean()),
        "var_cond_over_global": float((d.var_cond / d.var_global).mean()),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    d.to_csv(f"{args.out}/reference_diagnostic.csv", index=False)
    json.dump(out, open(f"{args.out}/reference_diagnostic.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
