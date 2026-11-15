"""Independent combine + paired bootstrap for the S1-A phase-2 audit.

Reads the two per-cell JSONs produced by scripts/audit_s1a2_remote.py.
Implements the prereg algorithm from scratch: engine-level RMSE, same-engine
same-cut pairing, seed mean first, four-subset equal weight, 10,000 engine
resamples, RNG 20261114.
"""
import json
from pathlib import Path

import numpy as np

METHODS = ["aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn"]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [101, 202, 303]
EXP_ROWS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 250}


def load(path):
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return json.loads(text)["cells"]


cells, dups = {}, []
for path in ["audit_a40.json", "audit_3090.json"]:
    for c in load(path):
        key = (c["arm"], c["method"], c["subset"], c["seed"])
        if key in cells:
            dups.append((key, cells[key]["engine_rmse"] == c["engine_rmse"]))
        else:
            cells[key] = c

expected = [("v0", m, s, k) for m in METHODS for s in SUBSETS for k in SEEDS]
expected += [("v1", m, s, k) for m in METHODS for s in SUBSETS for k in SEEDS]
expected += [("sham", m, s, k) for m in ["aidgn", "two_phase", "dual_mixer"]
             for s in SUBSETS for k in SEEDS]
missing = [e for e in expected if e not in cells]
extra = [k for k in cells if k not in expected]

bad_rows = [(k, c["n_rows"]) for k, c in cells.items() if c["n_rows"] != EXP_ROWS[c["subset"]]]
bad_cuts = [k for k, c in cells.items() if not c["cuts_ok"]]
bad_cfg = [(k, c["problems"]) for k, c in cells.items() if c["problems"]]

print(f"cells={len(cells)} expected={len(expected)} missing={len(missing)} extra={len(extra)} dups={len(dups)}")
print(f"bad_rows={len(bad_rows)} bad_cuts={len(bad_cuts)} bad_cfg={len(bad_cfg)}")
for label, items in (("missing", missing), ("extra", extra), ("bad_rows", bad_rows),
                     ("bad_cuts", bad_cuts), ("bad_cfg", bad_cfg)):
    if items:
        print(f"  {label}: {items[:8]}")


def pair(left, right, method, subset):
    rows = []
    for k in SEEDS:
        l, r = cells.get((left, method, subset, k)), cells.get((right, method, subset, k))
        if not l or not r:
            return None
        lu = sorted(int(u) for u in l["engine_rmse"])
        ru = sorted(int(u) for u in r["engine_rmse"])
        if lu != ru:
            raise SystemExit(f"unit mismatch {left}/{right}/{method}/{subset}/s{k}")
        rows.append([l["engine_rmse"][str(u)] - r["engine_rmse"][str(u)] for u in lu])
    return np.asarray(rows, float)


def analyse(left, right, method, rng, resamples=10000):
    per_sub, samples = {}, []
    for s in SUBSETS:
        m = pair(left, right, method, s)
        if m is None:
            return None
        per_sub[s] = float(m.mean())
        n = m.shape[1]
        picks = rng.integers(0, n, size=(resamples, n))
        samples.append(m[:, picks].mean(axis=0).mean(axis=1))
    comb = np.mean(np.vstack(samples), axis=0)
    lo, hi = np.percentile(comb, [2.5, 97.5])
    lo2, hi2 = np.percentile(comb, [0.835, 99.165])
    delta = float(np.mean(list(per_sub.values())))
    direction = sum(1 for v in per_sub.values() if v < 0)
    weights = {s: cells[(left, method, s, 101)]["n_units"] for s in SUBSETS}
    weighted = float(sum(weights[s] * per_sub[s] for s in SUBSETS) / sum(weights.values()))
    return dict(delta=delta, lo=float(lo), hi=float(hi), lo2=float(lo2), hi2=float(hi2),
                direction=direction, weighted=weighted, per_sub=per_sub)


report = {
    ("aidgn", "v1", "sham"): (-11.7130, -12.8280, -10.5461, -13.0689, -10.2620, 3, -15.8277),
    ("two_phase", "v1", "sham"): (-0.7129, -1.0542, -0.3729, -1.1281, -0.3094, 4, -0.6716),
    ("dual_mixer", "v1", "sham"): (-0.1615, -0.4745, 0.1516, -0.5473, 0.2188, 3, -0.1998),
    ("ffve", "v1", "v0"): (-0.2371, -0.5745, 0.1099, -0.6609, 0.1964, 3, -0.1723),
    ("fcstgnn", "v1", "v0"): (-0.3091, -0.6887, 0.0811, -0.7709, 0.1631, 3, -0.2711),
}

rng = np.random.default_rng(20261114)
order = [("aidgn", "v1", "sham"), ("two_phase", "v1", "sham"),
         ("dual_mixer", "v1", "sham"), ("ffve", "v1", "v0"),
         ("fcstgnn", "v1", "v0"), ("aidgn", "v1", "v0"),
         ("aidgn", "sham", "v0"), ("two_phase", "v1", "v0"),
         ("two_phase", "sham", "v0"), ("dual_mixer", "v1", "v0"),
         ("dual_mixer", "sham", "v0")]
results = {}
print("\n=== independent recompute ===")
for method, left, right in order:
    r = analyse(left, right, method, rng)
    results[(method, left, right)] = r
    subs = " / ".join(f"{r['per_sub'][s]:+.4f}" for s in SUBSETS)
    print(f"{method:10s} {left}-{right:5s} d={r['delta']:+.4f} "
          f"ci=[{r['lo']:+.4f},{r['hi']:+.4f}] ci98=[{r['lo2']:+.4f},{r['hi2']:+.4f}] "
          f"dir={r['direction']}/4 w={r['weighted']:+.4f} | {subs}")

print("\n=== vs report (§2.1) ===")
for key, exp in report.items():
    r = results[key]
    dd = r["delta"] - exp[0]
    dlo = r["lo"] - exp[1]
    dhi = r["hi"] - exp[2]
    dlo2 = r["lo2"] - exp[3]
    dhi2 = r["hi2"] - exp[4]
    print(f"{key[0]:10s} dmean={dd:+.5f} dlo={dlo:+.5f} dhi={dhi:+.5f} "
          f"dlo98={dlo2:+.5f} dhi98={dhi2:+.5f} dir {r['direction']}/{exp[5]} "
          f"w {r['weighted']-exp[6]:+.5f}")

print("\n=== decomposition V1-sham == (V1-V0) - (sham-V0) ===")
for method in ["aidgn", "two_phase", "dual_mixer"]:
    a = results[(method, "v1", "sham")]["delta"]
    b = results[(method, "v1", "v0")]["delta"]
    c = results[(method, "sham", "v0")]["delta"]
    print(f"{method:10s} {b:+.4f} - {c:+.4f} = {b-c:+.4f}  vs {a:+.4f}  diff {b-c-a:+.6f}")

print("\n=== two_phase V0 vs V1 config ===")
for s in SUBSETS:
    for k in SEEDS:
        v0 = cells[("v0", "two_phase", s, k)]
        v1 = cells[("v1", "two_phase", s, k)]
        diffs = []
        for key in ["epochs", "patience", "batch", "lr", "window", "precision"]:
            if v0[key] != v1[key]:
                diffs.append(f"{key}:{v0[key]}->{v1[key]}")
        common = set(v0["code_sha256"] or {}) & set(v1["code_sha256"] or {})
        code_diff = [n for n in sorted(common)
                     if v0["code_sha256"][n] != v1["code_sha256"][n]]
        print(f"{s}/s{k}: budget_diff={diffs or 'none'} code_diff={code_diff or 'none'}")

print("\n=== aidgn sham DEAD candidates (sham engine RMSE) ===")
for s, k in [("FD004", 101), ("FD004", 303), ("FD002", 303), ("FD004", 202), ("FD001", 303)]:
    sham = cells.get(("sham", "aidgn", s, k))
    v0 = cells.get(("v0", "aidgn", s, k))
    if sham and v0:
        sm = float(np.mean(list(sham["engine_rmse"].values())))
        vm = float(np.mean(list(v0["engine_rmse"].values())))
        print(f"sham/aidgn/{s}/s{k}: sham_engine={sm:.4f} v0_engine={vm:.4f} gap={sm-vm:+.4f}")
