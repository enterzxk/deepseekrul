"""Independent recheck of S1-A first half. Reimplements the frozen rule from
the prereg text; does not import analyse_s1a.py."""
import csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path("paper_b/results/s1a_collected")
TAGS = {"none": "S1A_none", "pre_norm": "S1A_pre",
        "post_norm": "S1A_post", "head_concat": "S1A_head"}
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [101, 202, 303, 42, 43]
POSITIONS = ["none", "pre_norm", "post_norm", "head_concat"]


def per_engine(path):
    acc = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            u = int(r["unit"])
            t, p = float(r["y_true"]), float(r["y_pred"])
            acc.setdefault(u, []).append((t, p))
    out = {}
    for u, pts in acc.items():
        out[u] = math.sqrt(sum((p - t) ** 2 for t, p in pts) / len(pts))
    return out


cells = {}
for pos in POSITIONS:
    for sub in SUBSETS:
        for seed in SEEDS:
            path = ROOT / TAGS[pos] / sub / f"s{seed}" / "val_pred.csv"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            cells[(pos, sub, seed)] = per_engine(path)

# ---- 7.1 absolute table -------------------------------------------------
doc_abs = {
    ("FD001", "none"): 11.7198, ("FD001", "pre_norm"): 10.5248,
    ("FD001", "post_norm"): 11.1654, ("FD001", "head_concat"): 11.7711,
    ("FD002", "none"): 11.3998, ("FD002", "pre_norm"): 10.6992,
    ("FD002", "post_norm"): 10.7296, ("FD002", "head_concat"): 11.0795,
    ("FD003", "none"): 11.9353, ("FD003", "pre_norm"): 10.2740,
    ("FD003", "post_norm"): 10.5964, ("FD003", "head_concat"): 10.3683,
    ("FD004", "none"): 13.5791, ("FD004", "pre_norm"): 11.8872,
    ("FD004", "post_norm"): 12.0917, ("FD004", "head_concat"): 12.4120,
}
print("## 7.1 absolute table")
worst = 0.0
for sub in SUBSETS:
    for pos in POSITIONS:
        per_seed = [np.mean(list(cells[(pos, sub, s)].values())) for s in SEEDS]
        mean, sd = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
        diff = mean - doc_abs[(sub, pos)]
        worst = max(worst, abs(diff))
        print(f"{sub} {pos:12s} mine {mean:.4f} +/- {sd:.4f}  doc {doc_abs[(sub,pos)]:.4f}  d={diff:+.5f}")
print(f"max |mine-doc| over 16 cells = {worst:.5f}")


def delta_matrix(left, right, sub):
    units = sorted(cells[(left, sub, 101)])
    if units != sorted(cells[(right, sub, 101)]):
        raise SystemExit(f"engine sets differ {left}/{right}/{sub}")
    rows = []
    for seed in SEEDS:
        a, b = cells[(left, sub, seed)], cells[(right, sub, seed)]
        rows.append([a[u] - b[u] for u in units])
    return np.asarray(rows, float)


def boot(matrix, rng, resamples=10000):
    n = matrix.shape[1]
    picks = rng.integers(0, n, size=(resamples, n))
    return matrix[:, picks].mean(axis=0).mean(axis=1)


pairs = [(p, "none") for p in POSITIONS[1:]]
pairs += [(POSITIONS[j], POSITIONS[i])
          for i in range(1, len(POSITIONS)) for j in range(i + 1, len(POSITIONS))]

doc_pairs = {
    ("pre_norm", "none"): (-1.3122, (-1.8898, -0.7583), (-2.0230, -0.6441), 4),
    ("post_norm", "none"): (-1.0127, (-1.4042, -0.6244), (-1.4878, -0.5426), 4),
    ("head_concat", "none"): (-0.7508, (-1.1284, -0.4012), (-1.2058, -0.3265), 3),
    ("post_norm", "pre_norm"): (0.2994, (-0.1409, 0.7303), (-0.2494, 0.8134), 0),
    ("head_concat", "pre_norm"): (0.5614, (0.1420, 0.9817), (0.0504, 1.0857), 0),
    ("head_concat", "post_norm"): (0.2620, (0.0146, 0.5059), (-0.0365, 0.5574), 1),
}

rng = np.random.default_rng(20261114)
print("\n## 7.2/7.3 pairs (replicated draw order)")
print(f"{'pair':34s} {'mean':>9s} {'95% CI':>22s} {'98.33% CI':>22s} dir  weighted")
for left, right in pairs:
    samples, per_sub = [], {}
    for sub in SUBSETS:
        m = delta_matrix(left, right, sub)
        per_sub[sub] = float(m.mean())
        samples.append(boot(m, rng))
    comb = np.mean(np.vstack(samples), axis=0)
    lo, hi = np.percentile(comb, [2.5, 97.5])
    lo2, hi2 = np.percentile(comb, [0.835, 99.165])
    mean = float(np.mean(list(per_sub.values())))
    direction = sum(1 for v in per_sub.values() if v < 0)
    n_eng = {sub: len(cells[(left, sub, 101)]) for sub in SUBSETS}
    weighted = float(sum(n_eng[s] * per_sub[s] for s in SUBSETS) / sum(n_eng.values()))
    d = doc_pairs[(left, right)]
    print(f"{left+' - '+right:34s} {mean:+9.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"[{lo2:+.4f},{hi2:+.4f}]  {direction}/4  {weighted:+.4f}")
    print(f"{'':34s} doc {d[0]:+.4f} [{d[1][0]:+.4f},{d[1][1]:+.4f}]  "
          f"[{d[2][0]:+.4f},{d[2][1]:+.4f}]  {d[3]}/4")
    print(f"{'':34s} dmean={mean-d[0]:+.5f} dlo={lo-d[1][0]:+.5f} dhi={hi-d[1][1]:+.5f} "
          f"dlo2={lo2-d[2][0]:+.5f} dhi2={hi2-d[2][1]:+.5f}")
    print(f"{'':34s} per-subset " + " / ".join(f"{per_sub[s]:+.4f}" for s in SUBSETS))
