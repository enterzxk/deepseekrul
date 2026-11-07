"""Generate the manuscript figures from the saved artefacts."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

ROOT = Path("experiments/punet_matrix")
OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def wq(v, w, lv):
    o = np.argsort(v); vv, ww = v[o], w[o]
    cw = np.cumsum(ww) / ww.sum()
    return float(vv[min(int(np.searchsorted(cw, lv)), vv.size - 1)])


def curve_scaled(y, mu, w, cal, tst):
    xs, ys = [], []
    for dl in DELTAS:
        q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                              min(1.0, (1 - dl) * (1 + 1 / cal.sum())), method="higher"))
        L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
        xs.append(float(np.mean((y[tst] >= L) & (y[tst] <= H)))); ys.append(wink(y[tst], L, H, dl))
    return xs, ys


def curve_cqr(y, lo, hi, cal, tst):
    xs, ys = [], []
    for dl in DELTAS:
        s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        q = float(np.quantile(s, min(1.0, (1 - dl) * (1 + 1 / s.size)), method="higher"))
        L, H = lo[tst] - q, hi[tst] + q
        xs.append(float(np.mean((y[tst] >= L) & (y[tst] <= H)))); ys.append(wink(y[tst], L, H, dl))
    return xs, ys


def curve_loc(y, lo, hi, cal, tst):
    s_c = (hi[cal] - lo[cal]) / 2.0
    E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    s_t = (hi[tst] - lo[tst]) / 2.0
    bw = max(float(np.median(np.abs(s_c - np.median(s_c)))), 1e-3)
    xs, ys = [], []
    for dl in DELTAS:
        lv = min(1.0, (1 - dl) * (1 + 1 / E.size))
        L = np.empty(tst.sum()); H = np.empty(tst.sum())
        for i, q in enumerate(s_t):
            z = wq(E, np.exp(-np.abs(s_c - q) / bw), lv)
            L[i], H[i] = lo[tst][i] - z, hi[tst][i] + z
        xs.append(float(np.mean((y[tst] >= L) & (y[tst] <= H)))); ys.append(wink(y[tst], L, H, dl))
    return xs, ys


# ---- Figure 1: frontiers ----
fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=False)
for ax, sub in zip(axes, ["FD001", "FD002", "FD003", "FD004"]):
    d = ROOT / sub / "f0_s42"
    if not (d / "preds_cqr.csv").exists():
        continue
    c = np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True)
    p = np.genfromtxt(d / "preds_point.csv", delimiter=",", names=True)
    cal, tst = split(c["unit"])
    y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
    mu = (p["y_lo"] + p["y_hi"]) / 2.0
    for (xs, ys), lab, st in [
        (curve_scaled(p["y_true"], mu, np.ones_like(mu), cal, tst), "constant width", ":"),
        (curve_cqr(y, lo, hi, cal, tst), "CQR", "--"),
        (curve_loc(y, lo, hi, cal, tst), "localized CQR", "-"),
    ]:
        ax.plot(xs, ys, st, label=lab, lw=1.6 if lab.startswith("loc") else 1.2)
    ax.axvline(0.90, color="grey", lw=0.7, alpha=0.6)
    ax.set_title(sub, fontsize=9)
    ax.set_xlabel("realised coverage")
axes[0].set_ylabel("Winkler score")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig1_frontiers.pdf")
fig.savefig(OUT / "fig1_frontiers.png", dpi=200)
print("fig1 ok")

# ---- Figure 2: stage-wise coverage ----
subs = ["FD001", "FD002", "FD003", "FD004"]
stage = {s: [] for s in subs}
for s in subs:
    vals = []
    for f in range(5):
        p = Path("experiments/punet_e2e") / s / f"f{f}_s42" / "e2e_cqr_none.json"
        if p.exists():
            bins = json.loads(p.read_text()).get("diagnostics", {}).get("stage_bins", [])
            if len(bins) == 5:
                vals.append([b["coverage"] for b in bins])
    if vals:
        stage[s] = np.mean(np.array(vals), axis=0)
if any(len(v) for v in stage.values()):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    w = 0.2
    for i, s in enumerate(subs):
        if len(stage[s]):
            ax.bar(np.arange(5) + (i - 1.5) * w, stage[s], w, label=s)
    ax.axhline(0.90, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.set_xlabel("degradation stage (early to end-of-life)")
    ax.set_ylabel("coverage")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_ylim(0.75, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_stages.pdf"); fig.savefig(OUT / "fig2_stages.png", dpi=200)
    print("fig2 ok")

# ---- Figure 3: per-engine coverage distribution ----
mins = {s: [] for s in subs}
for s in subs:
    for f in range(5):
        p = Path("experiments/punet_e2e") / s / f"f{f}_s42" / "e2e_cqr_none.json"
        if p.exists():
            j = json.loads(p.read_text())
            if j.get("engine_min") is not None:
                mins[s].append(j["engine_min"])
if any(mins.values()):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    data = [mins[s] for s in subs if mins[s]]
    ax.boxplot(data, labels=[s for s in subs if mins[s]], widths=0.5)
    ax.axhline(0.80, color="r", ls="--", lw=0.8)
    ax.axhline(0.90, color="k", ls=":", lw=0.8)
    ax.set_ylabel("per-engine minimum coverage")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_engine.pdf"); fig.savefig(OUT / "fig3_engine.png", dpi=200)
    print("fig3 ok")
