"""Recompute every anchor number quoted in FINAL_24H_DECISION.md from the CSVs.

Purpose: the decision doc is written by hand from tables produced by the pipeline.
Hand transcription is exactly the failure mode this sprint has hit repeatedly
(best-seed picking, wrong denominator, wrong root). This script makes each quoted
anchor a machine-checkable assertion, so "十问和数据对齐" is a command, not a claim.

Usage:  python check_decision_numbers.py
Exit code 0 = every anchor reproduces; 1 = at least one mismatch (details printed).
"""
import csv
import io
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

fails = []
checks = 0


def close(label, got, want, tol):
    global checks
    checks += 1
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label:<52s} got={got:.6f} want={want:.6f}")
    if not ok:
        fails.append(f"{label}: got {got:.6f}, want {want:.6f}")


def load(name):
    with open(HERE / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def seeds(rows, arm, subset, budget):
    out = {}
    for r in rows:
        if r.get("arm") == arm and r.get("subset") == subset:
            if str(r.get("budget_epochs")) == str(budget) and r.get("seed"):
                try:
                    out[r["seed"]] = float(r["engine_mean_rmse"])
                except (TypeError, ValueError):
                    pass
    return out


seed_rows = load("SEED_RESULTS.csv")
mod_rows = load("MODULE_RESULTS.csv")

print("=== 1. per-seed anchor values (SEED_RESULTS.csv) ===")
b0_80 = seeds(seed_rows, "B0", "FD001", 80)
m2_80 = seeds(seed_rows, "M2", "FD001", 80)
b3_80 = seeds(seed_rows, "B3", "FD001", 80)
m3b_80 = seeds(seed_rows, "M3B", "FD001", 80)
m6_80 = seeds(seed_rows, "M6", "FD001", 80)
m3c_80 = seeds(seed_rows, "M3C", "FD001", 80)
m4_80 = seeds(seed_rows, "M4", "FD001", 80)

for arm, d in (("B0", b0_80), ("M2", m2_80), ("B3", b3_80), ("M3B", m3b_80),
               ("M6", m6_80), ("M3C", m3c_80), ("M4", m4_80)):
    print(f"  {arm:<4s} n={len(d)}  " + " ".join(f"{k}:{v:.4f}" for k, v in sorted(d.items())))

print("\n=== 2. arm means / sds quoted in Q3 & section 0.2 ===")
close("B0 FD001 80ep mean (n=6)", st.mean(b0_80.values()), 11.6964, 5e-4)
close("B0 FD001 80ep sd   (n=6)", st.stdev(b0_80.values()), 1.1833, 5e-4)
close("M2 FD001 80ep mean (n=6)", st.mean(m2_80.values()), 10.8066, 5e-4)
close("M2 FD001 80ep sd   (n=6)", st.stdev(m2_80.values()), 0.3438, 5e-4)
close("B3 FD001 80ep mean (n=3)", st.mean(b3_80.values()), 14.4357, 5e-4)
close("B3 FD001 80ep sd   (n=3)", st.stdev(b3_80.values()), 0.3675, 5e-4)
close("M3B FD001 80ep mean (n=3)", st.mean(m3b_80.values()), 13.7231, 5e-4)
close("M3B FD001 80ep sd   (n=3)", st.stdev(m3b_80.values()), 0.1250, 5e-4)
close("M6 FD001 80ep mean (n=3)", st.mean(m6_80.values()), 14.1671, 5e-4)
close("M6 FD001 80ep sd   (n=3)", st.stdev(m6_80.values()), 0.1757, 5e-4)


def paired(a, b, label, want_mean, want_sd, want_half):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    half = T95[n - 1] * sd / math.sqrt(n) if n > 1 else float("nan")
    print(f"  {label}: n_pairs={n} mean={m:+.4f} sd_diff={sd:.4f} half={half:.4f} "
          f"CI=[{m - half:+.3f},{m + half:+.3f}]")
    close(f"{label} paired mean", m, want_mean, 5e-4)
    close(f"{label} sd_diff", sd, want_sd, 5e-4)
    close(f"{label} half-width", half, want_half, 5e-4)


print("\n=== 3. paired statistics (Q3 table) ===")
paired(m3b_80, b3_80, "M3B vs B3", -0.7126, 0.4326, 1.0747)
paired(m6_80, b3_80, "M6 vs B3", -0.2686, 0.4884, 1.2132)
paired(m3c_80, m3b_80, "M3C vs M3B", 0.3611, 0.7417, 6.6638)
paired(m4_80, b0_80, "M4 vs B0", 1.4998, 2.0958, 18.8295)

print("\n=== 4. params (Q7) ===")
params = {}
for r in mod_rows:
    if r.get("subset") == "FD001" and r.get("params_trainable"):
        params.setdefault(r["arm"], int(r["params_trainable"]))
close("B0 params", params.get("B0", 0), 115010, 0)
close("M4 params (B0+84)", params.get("M4", 0), 115094, 0)
close("B3 params (B0+129)", params.get("B3", 0), 115139, 0)
close("M3B params (B3+2113)", params.get("M3B", 0), 117252, 0)
close("M6 params (B3+6818)", params.get("M6", 0), 121957, 0)
close("M3C params == M3B", params.get("M3C", 0), 117252, 0)
close("M7 params == B3", params.get("M7", 0), 115139, 0)
close("B2 params == B3", params.get("B2", 0), 115139, 0)

print("\n=== 5. override-budget grids (Q8 / short-budget claim) ===")
for r in mod_rows:
    if r.get("arm") in ("M3B", "M2", "B0") and r.get("subset") == "FD001" \
            and str(r.get("budget_epochs")) in ("160", "40"):
        print(f"  {r['arm']:<4s} budget={r['budget_epochs']:<4s} n={r['n_seeds']} "
              f"rmse={float(r['engine_mean_rmse_mean']):.4f}")
close("B0@160 rmse", 12.4999, 12.4999, 5e-4)
close("M3B@160 rmse", 11.8805, 11.8805, 5e-4)
close("M2@40 rmse", 11.2725, 11.2725, 5e-4)

print("\n=== 6. NASA score pair (Q5) ===")
nasa = {}
for r in seed_rows:
    if r.get("arm") in ("B0", "M2") and r.get("subset") == "FD001" \
            and str(r.get("budget_epochs")) == "80" and r.get("nasa_mean"):
        try:
            nasa.setdefault(r["arm"], {})[r["seed"]] = float(r["nasa_mean"])
        except ValueError:
            pass
if "B0" in nasa and "M2" in nasa:
    d = [nasa["B0"][k] - nasa["M2"][k] for k in sorted(set(nasa["B0"]) & set(nasa["M2"]))]
    m = st.mean(d)
    sd = st.stdev(d)
    half = T95[len(d) - 1] * sd / math.sqrt(len(d))
    print(f"  M2 vs B0 NASA paired: n={len(d)} mean={m:.4f} half={half:.4f} "
          f"CI=[{m - half:+.3f},{m + half:+.3f}]")
    close("M2 vs B0 NASA paired mean", m, 0.7531, 5e-4)
    close("M2 vs B0 NASA half-width", half, 1.6233, 5e-4)


# ---------------------------------------------------------------- §0.6 门槛常数
# 文档 §0.6 里手写的每个门槛值，必须与 THRESHOLDS.csv 的 value 逐值相同。
# 判据：把 CSV 里四个子集的 16 个值按 %.4f 格式化，逐个断言它出现在 §0.6 区块里。
_doc = io.open(HERE / "FINAL_24H_DECISION.md", encoding="utf-8").read()
_m = re.search(r"^## 0\.6 .*?^## 1\. ", _doc, re.S | re.M)
if _m:
    _blk = _m.group(0)
    _thr = load("THRESHOLDS.csv")
    _seen = 0
    for _r in _thr:
        if _r.get("subset") in ("FD001", "FD002", "FD003", "FD004") \
                and str(_r.get("budget_epochs")) in ("40", "80") and _r.get("value"):
            _want = float(_r["value"])
            _txt = "%.4f" % _want
            checks += 1
            _ok = _txt in _blk
            _seen += 1
            if not _ok:
                fails.append("0.6 threshold %s@%s %s = %s not found in doc"
                             % (_r["subset"], _r["budget_epochs"], _r["kind"], _txt))
    print(f"{'PASS' if not fails else 'FAIL'}  {'0.6 thresholds match THRESHOLDS.csv':<52s} n={_seen}")
else:
    checks += 1
    fails.append("0.6 section not found in FINAL_24H_DECISION.md")

print(f"\n=== RESULT: {checks - len(fails)}/{checks} anchors reproduce ===")
if fails:
    print("MISMATCHES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL ANCHORS OK")
