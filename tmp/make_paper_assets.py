"""Build paper tables and figures from saved per-window predictions (read-only).

Every number is read from CSV/JSON; nothing is typed by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_T = HERE / "paper_tables"; OUT_F = HERE / "paper_figures"
OUT_T.mkdir(exist_ok=True); OUT_F.mkdir(exist_ok=True)
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
REPS = 5000


def nasa(e):
    e = np.asarray(e, float)
    return np.where(e > 0, np.expm1(e / 10.0), np.expm1(-e / 13.0))


def load(base, subset, arm, seed, folds=range(5)):
    blocks = []
    for k in folds:
        p = Path(base) / subset / str(k) / f"{arm}_{seed}.csv"
        if not p.is_file():
            return None
        blocks.append(pd.read_csv(p))
    d = pd.concat(blocks, ignore_index=True)
    d["err"] = d.y_pred - d.y_true
    return d


def eng(d):
    g = d.groupby("unit")
    return pd.DataFrame({"mse": g.apply(lambda x: float((x.err ** 2).mean()), include_groups=False),
                         "mae": g.apply(lambda x: float(x.err.abs().mean()), include_groups=False),
                         "nasa": g.apply(lambda x: float(nasa(x.err).mean()), include_groups=False),
                         "bias": g.err.mean(), "n": g.size()})


def pooled(t):
    return dict(rmse=float(np.sqrt(t.mse.mean())), mae=float(t.mae.mean()), nasa=float(t.nasa.mean()),
                bias=float(t.bias.mean()), engines=int(len(t)), windows=int(t.n.sum()))


def paired(a, b):
    idx = a.index.intersection(b.index)
    av, bv = a.loc[idx, "mse"].to_numpy(), b.loc[idx, "mse"].to_numpy()
    rng = np.random.default_rng(99)
    dr = rng.integers(0, len(idx), (REPS, len(idx)))
    dif = np.sqrt(av[dr].mean(1)) - np.sqrt(bv[dr].mean(1))
    return float(np.sqrt(av.mean()) - np.sqrt(bv.mean())), float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))


def main():
    lock = json.loads((HERE / "REFERENCE_LOCK.json").read_text()) if (HERE / "REFERENCE_LOCK.json").is_file() else {"subsets": {}}
    rows = []
    for subset in SUBSETS:
        ref = lock["subsets"].get(subset, {}).get("reference")
        for arm in ["M0", "OPS", "FULL", "NORM", "PHASE", "DIFF"]:
            per_seed = {}
            for seed in [101, 202, 303]:
                d = load("runs", subset, arm, seed)
                if d is None:
                    continue
                per_seed[seed] = eng(d)
            if not per_seed:
                continue
            rmses = [pooled(t)["rmse"] for t in per_seed.values()]
            first = pooled(next(iter(per_seed.values())))
            row = dict(subset=subset, arm=arm, n_seeds=len(per_seed),
                       rmse_mean=float(np.mean(rmses)), rmse_sd=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                       mae=first["mae"], nasa=first["nasa"], bias=first["bias"],
                       engines=first["engines"], windows=first["windows"])
            if ref and 101 in per_seed:
                rd = load("runs", subset, ref, 101)
                if rd is not None:
                    dlt, lo, hi = paired(per_seed[101], eng(rd))
                    row.update(reference=ref, delta_rmse=dlt, ci_lo=lo, ci_hi=hi,
                               gain=1.0 - pooled(per_seed[101])["rmse"] / pooled(eng(rd))["rmse"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(HERE / "RESULTS.csv", index=False)
        Path(HERE / "RESULTS.json").write_text(frame.to_json(orient="records", indent=2))
        md = frame.round(4).to_markdown(index=False)
        (OUT_T / "main_table.md").write_text(md)
        print(md)
        # figure: RMSE by arm and subset
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, subset in enumerate([s for s in SUBSETS if s in set(frame.subset)]):
            sub = frame[frame.subset == subset]
            ax.bar(np.arange(len(sub)) + i * 0.25, sub.rmse_mean, width=0.22, label=subset, yerr=sub.rmse_sd, capsize=2)
        ax.set_xticks(np.arange(len(frame.arm.unique())))
        ax.set_xticklabels(list(frame.arm.unique()))
        ax.set_ylabel("pooled RMSE (cycles)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(OUT_F / "fig_rmse_by_arm.png", dpi=200)
        fig.savefig(OUT_F / "fig_rmse_by_arm.pdf")
    obs = HERE / "observability.json"
    if obs.is_file():
        o = json.loads(obs.read_text())
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for subset, vals in o.items():
            cs = sorted(int(c) for c in vals)
            ax.plot(cs, [vals[str(c)]["corr"] for c in cs], marker="o", label=subset)
        ax.set_xlabel("observed cycles at cutoff"); ax.set_ylabel(r"corr($\mu_T$, true life)")
        ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT_F / "fig_observability.png", dpi=200)
        fig.savefig(OUT_F / "fig_observability.pdf")
        rows = []
        for subset, vals in o.items():
            for c, v in sorted(vals.items(), key=lambda kv: int(kv[0])):
                rows.append(dict(subset=subset, cutoff=int(c), **v))
        pd.DataFrame(rows).to_csv(OUT_T / "observability.csv", index=False)
        (OUT_T / "observability.md").write_text(pd.DataFrame(rows).round(3).to_markdown(index=False))
    print("assets written to", OUT_T, OUT_F)


if __name__ == "__main__":
    main()
