"""Compare only where both models estimate the same quantity.

After onset (t >= t_cp) the change-point label equals the true RUL, so post-onset
engines are the fair sub-population. Engines whose matched target is below the cap
have a recoverable cutoff (cutoff = T - y_true), which lets us classify them.
"""
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")
import dataset
from dataset import load_cmapss_train

OUT = Path("results/dsh/review_r1")
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")

# per-engine onset cycle, same rule as the labelling code
onset = {}
for u, g in df.groupby("unit"):
    g = g.sort_values("cycle")
    t = g.cycle.to_numpy(float)
    cols = [c for c in g.columns if c.startswith("s") and c[1:].isdigit()]
    x = g[cols].to_numpy(float); n = len(g)
    w = max(5, n // 20)
    mu = x[:w].mean(0); sd = x[:w].std(0) + 1e-9
    hi = ((x - mu) / sd).mean(1); k = max(5, n // 20)
    sm = np.convolve(hi, np.ones(k) / k, mode="same")
    base = sm[: max(5, n // 10)]
    ab = np.where(sm > base.mean() + 3.0 * (base.std() + 1e-9))[0]
    cp = int(ab[0]) if ab.size else max(0, int(0.6 * n))
    onset[u] = float(t[min(max(cp, 10), n - 1)])
    # store last_cycle too
    onset[(u, "T")] = float(t[-1])

def load(tag):
    fs = glob.glob(f"experiments/{tag}/FD002/*/matched_cutoff_predictions.csv")
    return pd.concat([pd.read_csv(f) for f in fs]).groupby("unit_id").agg(
        y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))

b = load("order03/M1_global").rename(columns={"y_pred": "m1", "y_true": "truth"})
c = load("cplab2").rename(columns={"y_pred": "cp"})
j = b.join(c[["cp"]], how="inner")
j["life"] = [onset[(u, "T")] for u in j.index]
j["t_cp"] = [onset[u] for u in j.index]
j["uncapped"] = j.truth < 124.9
j["cutoff"] = np.where(j.uncapped, j["life"] - j.truth, j["life"] - 125.0)
j["post_onset"] = j.cutoff >= j.t_cp

sub = j[j.uncapped & j.post_onset]
print(f"engines total={len(j)}  uncapped={int(j.uncapped.sum())}  post-onset subset={len(sub)}")
rows = []
for name, col in (("M1", "m1"), ("cplab2", "cp")):
    e = (sub[col] - sub.truth).to_numpy()
    rows.append({"model": name, "n": len(sub), "rmse": float(np.sqrt((e**2).mean())),
                 "mae": float(np.abs(e).mean()), "late": float((e > 0).mean()),
                 "severe": int((e > 10).sum()), "bias": float(e.mean())})
out = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "cp_post_onset.csv", index=False)
print(out.round(3).to_string(index=False))
