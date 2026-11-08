"""Ablation table: each variant vs M0 on the folds where both exist."""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REPS = 5000
VARIANTS = [
    ("DIFF", "runs3_w50", "M0", "window minus individual early reference"),
    ("PHASE", "runs3_w50", "M0", "RUL = phi_hat * mu_T"),
    ("M0_ep200", "runs3_w50_ep200", "M0", "200 epochs instead of 80"),
    ("M0_eqbal", "runs3_w50_eqbal", "M0", "engine-balanced sample weights"),
    ("NORM", "runs", "M0", "cycle coordinate replaced by c/mu_T"),
]


def load(base, subset, arm, seed=101, folds=range(5)):
    blocks = []
    for k in folds:
        p = Path(base) / subset / str(k) / f"{arm}_{seed}.csv"
        if not p.is_file():
            return None
        blocks.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(blocks, ignore_index=True)


def pooled(d):
    e = d.y_pred - d.y_true
    g = pd.DataFrame({"u": d.unit, "e2": e ** 2}).groupby("u").e2.mean()
    return float(np.sqrt(g.mean())), int(len(g))


def paired(a, b):
    ea = a.y_pred - a.y_true; eb = b.y_pred - b.y_true
    ga = pd.DataFrame({"u": a.unit, "e2": ea ** 2}).groupby("u").e2.mean()
    gb = pd.DataFrame({"u": b.unit, "e2": eb ** 2}).groupby("u").e2.mean()
    idx = ga.index.intersection(gb.index)
    av, bv = ga.loc[idx].to_numpy(), gb.loc[idx].to_numpy()
    rng = np.random.default_rng(99)
    dr = rng.integers(0, len(idx), (REPS, len(idx)))
    dif = np.sqrt(av[dr].mean(1)) - np.sqrt(bv[dr].mean(1))
    return (float(np.sqrt(av.mean()) - np.sqrt(bv.mean())),
            float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5)))


rows = []
for subset in ["FD002", "FD004"]:
    for name, base, ref, note in VARIANTS:
        folds = [k for k in range(5)
                 if (Path(base) / subset / str(k) / f"{name if name in ('NORM','DIFF','PHASE') else 'M0'}_101.csv").is_file()]
        if not folds:
            continue
        arm = name if name in ("NORM", "DIFF", "PHASE") else "M0"
        v = load(base, subset, arm, 101, folds)
        r = load("runs", subset, ref, 101, folds)
        if v is None or r is None:
            continue
        d, lo, hi = paired(v, r)
        rows.append(dict(subset=subset, variant=name, folds=len(folds), note=note,
                         rmse_variant=pooled(v)[0], rmse_M0=pooled(r)[0],
                         delta=d, ci_lo=lo, ci_hi=hi, engines=pooled(v)[1]))
frame = pd.DataFrame(rows)
if len(frame):
    frame.to_csv(HERE / "ABLATIONS.csv", index=False)
    Path(HERE / "ABLATIONS.json").write_text(frame.to_json(orient="records", indent=2))
    (HERE / "paper_tables").mkdir(exist_ok=True)
    (HERE / "paper_tables" / "ablations.md").write_text(frame.round(3).to_markdown(index=False))
    print(frame.round(3).to_string(index=False))
else:
    print("no ablation runs available yet")
