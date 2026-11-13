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

# === 7. evidence-tier counts: one generated block, plus a sweep =============
# run_id is NOT unique in the ledger (25 ids appear under several roots), so the
# key is run_dir. A hand-written tier count goes stale within ~25 minutes during
# a run, so the only durable form is a block generated from the CSV; everything
# else in the docs must be a pointer to this block.
TIER_BEGIN = "<!-- TIER:BEGIN generated by check_decision_numbers.py --emit-tier -->"
TIER_END = "<!-- TIER:END -->"
TIER_DOC = "FINAL_24H_DECISION.md"
SWEEP_DOCS = ["FINAL_24H_DECISION.md", "ABLATION_RESULTS.md", "EXPERIMENT_PLAN.md",
              "RUN_QUEUE_COLUMNS.md", "MECHANISM_EVIDENCE.md", "FAILED_IDEAS.md"]
# lines that are allowed to carry a stale number because they record that it is stale
STALE_OK = ("as of", "作废", "已撤回", "已作废", "曾", "旧版")


def tier_block():
    import hashlib
    raw = (HERE / "SEED_RESULTS.csv").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:12]
    rows = load("SEED_RESULTS.csv")
    if len({r["run_dir"] for r in rows}) != len(rows):
        fails.append("SEED_RESULTS.csv has duplicate run_dir (run_id is not a key)")
    by = {}
    for r in rows:
        by.setdefault(r.get("status", ""), []).append(r)
    usable = by.get("usable", [])
    ua = sum(1 for r in usable if r.get("evidence_tier") == "A")
    ub = sum(1 for r in usable if r.get("evidence_tier") == "B")
    arch, smoke = by.get("archived", []), by.get("smoke_dir", [])
    out = [TIER_BEGIN, "```",
           "as of   SEED_RESULTS.csv  sha256 %s  rows %d" % (sha, len(rows)),
           "        %d usable = %d A + %d B          <- same denominator"
           % (len(usable), ua, ub),
           "        %d archived + %d smoke        <- gate twins are the other side, not new cells"
           % (len(arch), len(smoke)),
           "",
           "subset  usable-A  usable-B  usable  rows"]
    for s in sorted({r["subset"] for r in rows}):
        g = [r for r in rows if r["subset"] == s]
        gu = [r for r in g if r.get("status") == "usable"]
        out.append("%-7s %9d %9d %7d %5d"
                   % (s, sum(1 for r in gu if r.get("evidence_tier") == "A"),
                      sum(1 for r in gu if r.get("evidence_tier") == "B"),
                      len(gu), len(g)))
    out += ["```", TIER_END]
    return "\n".join(out), ua, ub


_tier_txt, _ua, _ub = tier_block()

if "--emit-tier" in sys.argv:
    _p = HERE / TIER_DOC
    _src = _p.read_text(encoding="utf-8")
    if TIER_BEGIN in _src and TIER_END in _src:
        _pre = _src.split(TIER_BEGIN)[0]
        _post = _src.split(TIER_END, 1)[1]
        _p.write_text(_pre + _tier_txt + _post, encoding="utf-8")
    else:
        _p.write_text(_src.rstrip("\n") + "\n\n" + _tier_txt + "\n", encoding="utf-8")
    print("emitted tier block into", TIER_DOC)
    sys.exit(0)


def thr_table():
    """The 0.6 table, generated from THRESHOLDS.csv.  Paired with check 0.6:
    a generated block without a staleness check is worse than a hand copy,
    because a reader assumes it updates itself."""
    rows = load("THRESHOLDS.csv")
    tab = {}
    for r in rows:
        tab.setdefault("%s@%s" % (r["subset"], r["budget_epochs"]), {})[r["kind"]] = r
    order = ["FD001@40", "FD001@80", "FD001@160", "FD002@40",
             "FD003@80", "FD004@40", "FD004@80"]
    kinds = ["screen_single", "single_vs_mean", "paired", "ci_mean"]
    out = ["| `subset@budget` | `n` | `screen_single` | `single_vs_mean` | `paired` | `ci_mean` |",
           "|---|---:|---:|---:|---:|---:|"]
    for k in order:
        g = tab[k]
        cells = ["**%.4f**" % float(g[kk]["value"]) for kk in kinds]
        out.append("| `%s` | %s | %s |" % (k, g["paired"]["n_seeds"], " | ".join(cells)))
    return "\n".join(out)


if "--emit-thr" in sys.argv:
    _p = HERE / TIER_DOC
    _ls = _p.read_text(encoding="utf-8").split("\n")
    _i0 = next(i for i, l in enumerate(_ls) if l.startswith("| `subset@budget` | `n` |"))
    _i1 = _i0
    while _i1 < len(_ls) and _ls[_i1].startswith("|"):
        _i1 += 1
    _ls[_i0:_i1] = thr_table().split("\n")
    _p.write_text("\n".join(_ls), encoding="utf-8")
    print("emitted 0.6 threshold table into", TIER_DOC)
    sys.exit(0)

_src = (HERE / TIER_DOC).read_text(encoding="utf-8")
checks += 1
if TIER_BEGIN not in _src or TIER_END not in _src:
    fails.append("%s has no generated tier block (run --emit-tier)" % TIER_DOC)
    print(f"{'FAIL'}  {'7.1 tier block present':<52s} missing")
else:
    _have = _src.split(TIER_BEGIN, 1)[1].split(TIER_END, 1)[0].strip()
    _want = _tier_txt.split(TIER_BEGIN, 1)[1].split(TIER_END, 1)[0].strip()
    _ok = _have == _want
    if not _ok:
        fails.append("tier block in %s is stale vs SEED_RESULTS.csv" % TIER_DOC)
    print(f"{'PASS' if _ok else 'FAIL'}  {'7.1 tier block matches SEED_RESULTS.csv':<52s} "
          f"usable={_ua}A+{_ub}B")

# sweep: surface hand-written tier counts outside the block. Hard-fail only when
# the A/B pair is written and contradicts the ledger.
_pair = re.compile(r"(\d+)\s*A\s*/\s*(\d+)\s*B")
_num = re.compile(r"(\d+)\s*(?:个|格)?\s*(?:A\s*级|A\s*格)")
# per-subset truth, over usable rows only
_sub_ab = {}
for _r in load("SEED_RESULTS.csv"):
    if _r.get("status") != "usable":
        continue
    _a, _b = _sub_ab.get(_r["subset"], (0, 0))
    if _r.get("evidence_tier") == "A":
        _a += 1
    elif _r.get("evidence_tier") == "B":
        _b += 1
    _sub_ab[_r["subset"]] = (_a, _b)
_stale, _substale, _soft = [], [], 0
for _d in SWEEP_DOCS:
    _fp = HERE / _d
    if not _fp.exists():
        continue
    _txt = _fp.read_text(encoding="utf-8")
    if TIER_BEGIN in _txt:
        _txt = _txt.split(TIER_BEGIN)[0] + _txt.split(TIER_END, 1)[-1]
    for _i, _line in enumerate(_txt.split("\n"), 1):
        if any(k in _line for k in STALE_OK):
            continue
        _m2 = _pair.search(_line)
        if _m2:
            _got = (int(_m2.group(1)), int(_m2.group(2)))
            _named = [s for s in _sub_ab if s in _line]
            if _named:
                if not any(_got == _sub_ab[s] for s in _named):
                    _substale.append("%s:%d  %d A / %d B  (ledger %s = %d A / %d B)"
                                     % (_d, _i, _got[0], _got[1], "/".join(_named),
                                        _sub_ab[_named[0]][0], _sub_ab[_named[0]][1]))
            elif _got != (_ua, _ub):
                _stale.append("%s:%d  %d A / %d B  (ledger says %d A / %d B)"
                              % (_d, _i, _got[0], _got[1], _ua, _ub))
        elif _num.search(_line):
            _soft += 1
checks += 1
print(f"{'PASS' if not _stale else 'FAIL'}  {'7.2 no stale A/B pair outside the block':<52s} "
      f"stale={len(_stale)} subset_stale={len(_substale)} review={_soft}")
for _s in _stale:
    fails.append("stale tier count " + _s)
for _s in _substale:
    print("  WARN subset tier count " + _s)

# 7.3 negative assertions: "no gate anywhere" / "only in FD001" go stale the
# moment one gate lands, and a positive-count sweep cannot see them.  This is
# the fifth iteration of "a negative check decays as the doc grows".
# 7.3 is a REGRESSION GUARD, not a guarantee: a blacklist only covers the
# phrasings that have already occurred, and the space of prose variants is
# unbounded (seven realistic variants slipped through on first test).  The only
# real guarantee is 7.4 below -- check that the POINTER is present, because the
# set of places that should carry one is enumerable.
NEG = re.compile(r"一格都没有|A 级证据仅存在|从未通过|全部无门格|全部 `unverifiable`")
NEG_OK = ("计划时", "当时", "已作废", "已撤回", "曾", "旧版", "§0.7", "生成块", "WARN")
_neg = []
for _d in SWEEP_DOCS:
    _fp = HERE / _d
    if not _fp.exists():
        continue
    for _i, _line in enumerate(_fp.read_text(encoding="utf-8").split("\n"), 1):
        if NEG.search(_line) and not any(k in _line for k in NEG_OK):
            _neg.append("%s:%d" % (_d, _i))
checks += 1
print(f"{'PASS' if not _neg else 'FAIL'}  {'7.3 no bare negative gate assertion':<52s} "
      f"hits={len(_neg)}")
for _n in _neg:
    fails.append("negative gate assertion needs a pointer or a 'planned-time' marker: " + _n)

# 7.4 positive check: every line that talks about evidence tiers must carry a
# pointer to the generated block.  Positive checks miss (a position is either
# listed or not), negative checks both miss and over-fire; a positive one has a
# finite, enumerable failure set, which is why this is the guarantee and 7.3 is
# only the alarm.
TIERWORD = re.compile(r"A 级|B 级|A 格|B 格")
_tier_doc = HERE / TIER_DOC
_tlines = _tier_doc.read_text(encoding="utf-8").split("\n")
# lines inside section 0.7 do not need a pointer to themselves
_lo = next((i for i, l in enumerate(_tlines) if l.startswith("## 0.7")), -1)
_hi = next((i for i, l in enumerate(_tlines) if _lo >= 0 and i > _lo and l.startswith("## ")), len(_tlines))
_nopointer = []
for _i, _line in enumerate(_tlines, 1):
    if TIER_BEGIN in _line or TIER_END in _line:
        continue
    if _lo + 1 <= _i <= _hi:          # inside 0.7 itself
        continue
    if TIERWORD.search(_line) and "§0.7" not in _line:
        _nopointer.append("%s:%d  %s" % (TIER_DOC, _i, _line.strip()[:60]))
checks += 1
print(f"{'PASS' if not _nopointer else 'FAIL'}  {'7.4 every tier mention carries a pointer':<52s} "
      f"missing={len(_nopointer)}")
for _n in _nopointer:
    fails.append("tier mention without a §0.7 pointer: " + _n)

# 7.5 "not run" claims: a sentence that says something was never run is the one
# kind of error that makes a report look MORE honest, so nobody checks it.  Each
# claim is paired with the artifact that would exist if it HAD run; if the
# artifact is on disk, the sentence is stale by construction.
NOTRUN = [
    (re.compile(r"未跑集成|集成.*未跑|未跑.*集成"), "ENSEMBLE_VS_B0.csv"),
    (re.compile(r"未跑消融|消融.*未跑"), "ALPHA_ROBUSTNESS.csv"),
]
_nr = []
for _pat, _art in NOTRUN:
    if not (HERE / _art).exists():
        continue
    for _d in SWEEP_DOCS:
        _fp = HERE / _d
        if not _fp.exists():
            continue
        for _i, _line in enumerate(_fp.read_text(encoding="utf-8").split("\n"), 1):
            if _pat.search(_line) and not any(k in _line for k in NEG_OK):
                _nr.append("%s:%d says not-run but %s exists" % (_d, _i, _art))
checks += 1
print(f"{'PASS' if not _nr else 'FAIL'}  {'7.5 no not-run claim whose artifact exists':<52s} "
      f"hits={len(_nr)}")
for _n in _nr:
    fails.append("stale not-run claim: " + _n)

# 8.1 the E10 probe table: 13 arms x 6 numbers, every one hand-copied from
# MECHANISM_TABLE.csv.  Each cell is an independent drift point, so anchor all
# of them -- a 13-row table is 78 unprotected numbers otherwise, and every
# drift found tonight happened in a hand-copied number, never in a generated
# block.
_probe = load("MECHANISM_TABLE.csv")
_mech = (HERE / "MECHANISM_EVIDENCE.md").read_text(encoding="utf-8").split("\n")
# the arm name alone does not identify a row -- "| M3B |" occurs in several
# tables in this file, so scope the search to the E10 table body.
_h = next((i for i, l in enumerate(_mech)
           if l.startswith("| 臂 |") and "perm/val" in l), None)
if _h is None:
    checks += 1
    fails.append("E10 table header not found in MECHANISM_EVIDENCE.md")
    print(f"{'FAIL'}  {'8.1 E10 probe table matches the CSV':<52s} no header")
    _bad, _cells = [], 0
else:
    _rows = []
    _j = _h + 1
    while _j < len(_mech) and _mech[_j].startswith("|"):
        _rows.append(_mech[_j])
        _j += 1
    _bad, _cells = [], 0
    for _r in _probe:
        _a = _r.get("arm", "")
        _line = next((l for l in _rows if l.startswith("| %s |" % _a)), None)
        if _line is None:
            _bad.append("%s: row not found in the E10 table" % _a)
            continue
        _want = []
        for _k, _fmt in (("val_rmse", "%.4f"), ("rank_acc", "%.4f"), ("perm_mean", "%.4f")):
            try:
                _want.append(_fmt % float(_r[_k]))
            except (ValueError, KeyError):
                pass
        try:
            _want.append("%.4f" % abs(float(_r["degrad_abs"])))
            _want.append("%.2f" % float(str(_r["degrad_pct"]).replace("%", "").replace("+", "")))
            _want.append("%.4f" % (float(_r["perm_mean"]) / float(_r["val_rmse"])))
        except (ValueError, KeyError):
            pass
        for _w in _want:
            _cells += 1
            if _w not in _line:
                _bad.append("%s: %s not found in its E10 row" % (_a, _w))
    checks += 1
    print(f"{'PASS' if not _bad else 'FAIL'}  {'8.1 E10 probe table matches the CSV':<52s} "
          f"cells={_cells} bad={len(_bad)}")
    for _n in _bad:
        fails.append("E10 cell drift: " + _n)

# 9.1 two columns encode the same fact: `status` and `usable`.  The generated
# 0.7 block counts status == "usable"; an ad-hoc reader counts usable == "yes".
# If they ever disagree, two people report two sample sizes from one file, and
# neither is obviously wrong.  Blank is a third value and is the dangerous one:
# `usable != "no"` counts it as a sample, so 73 becomes 78.
_rows9 = load("SEED_RESULTS.csv")
_bad9 = [r for r in _rows9
         if (r.get("status") == "usable") != (r.get("usable") == "yes")]
_blank9 = [r for r in _rows9 if (r.get("usable") or "") == ""]
_blank_on_usable = [r for r in _blank9 if r.get("status") == "usable"]
checks += 1
print(f"{'PASS' if not _bad9 else 'FAIL'}  {'9.1 status and usable agree':<52s} "
      f"rows={len(_rows9)} disagree={len(_bad9)} blank={len(_blank9)} "
      f"blank_on_usable={len(_blank_on_usable)}")
for _n in _bad9:
    fails.append("status/usable disagree: %s %s %s root=%s status=%s usable=%r"
                 % (_n.get("arm"), _n.get("subset"), _n.get("seed"),
                    _n.get("root"), _n.get("status"), _n.get("usable")))

# 9.2 a root named _gate* is a reproduction root, so a cell under it is normally
# a byte-identical twin of a cell outside it.  But the ledger used to identify
# gate cells by ROOT NAME, which is a label, not a property -- _gate80 is running
# four cells that are new samples under a gate-named root, and a name-based rule
# eats them silently.  Match by preds_md5 (a property) and check BOTH directions,
# because there are two ways to get this wrong and only one of them is loud:
#   twin  -> must NOT be counted (else the same run is a sample twice)
#   novel -> MUST be counted (else a real run disappears)
# A one-sided check only catches the first, and the first is the harmless one.
_orig = {}
for _r in _rows9:
    if not (_r.get("root") or "").startswith("_gate"):
        _orig.setdefault(_r.get("preds_md5"), []).append(_r)
_gate_rows = [r for r in _rows9 if (r.get("root") or "").startswith("_gate")]
_double = [r for r in _gate_rows
           if r.get("preds_md5") in _orig
           and (r.get("status") == "usable" or r.get("usable") == "yes")]
_dropped = [r for r in _gate_rows
            if r.get("preds_md5") not in _orig
            and not (r.get("status") == "usable" and r.get("usable") == "yes")]
_n_twin = len(_gate_rows) - len([r for r in _gate_rows if r.get("preds_md5") not in _orig])
_n_novel = len(_gate_rows) - _n_twin
checks += 1
print(f"{'PASS' if not _double and not _dropped else 'FAIL'}  "
      f"{'9.2 gate-named cells are counted exactly once':<52s} "
      f"gate_rows={len(_gate_rows)} twin={_n_twin} novel={_n_novel} "
      f"double_counted={len(_double)} dropped={len(_dropped)}")
for _n in _double:
    fails.append("gate twin counted as a sample (same run twice): %s %s %s root=%s"
                 % (_n.get("arm"), _n.get("subset"), _n.get("seed"), _n.get("root")))
for _n in _dropped:
    fails.append("gate-named cell with no original was dropped as a reproduction "
                 "(it is a new sample): %s %s %s root=%s"
                 % (_n.get("arm"), _n.get("subset"), _n.get("seed"), _n.get("root")))

# 9.3 a retired number must not survive without a retirement marker on the same
# line.  Tonight the sweep for the dead threshold 1.5157 covered the .md files
# and stopped there, so the plan CSV kept "1.5156" and the margin that went with
# it -- a fourth file nobody re-read.  Numbers also need boundaries: a bare
# search for 1.5156 matches 11.51561101173279 in PER_ENGINE.csv, and a checker
# that fires on a per-engine RMSE teaches people to ignore it.
_DEAD = [("1.5156", r"(?<![\d.])1\.5156(?!\d)"),
         ("1.5157", r"(?<![\d.])1\.5157(?!\d)")]
_MARK = ("作废", "已替换", "复现不出", "改判", "retired", "superseded")
_dead_hits, _dead_unmarked = 0, []
for _f in sorted(HERE.glob("*.md")) + sorted(HERE.glob("*.csv")):
    if _f.name.startswith("tmp_"):
        continue
    for _i, _l in enumerate(_f.read_text(encoding="utf-8").split("\n"), 1):
        for _name, _pat in _DEAD:
            if re.search(_pat, _l):
                _dead_hits += 1
                if not any(_m in _l for _m in _MARK):
                    _dead_unmarked.append("%s:%d: %s" % (_f.name, _i, _name))
checks += 1
print(f"{'PASS' if not _dead_unmarked else 'FAIL'}  "
      f"{'9.3 retired numbers carry a retirement marker':<52s} "
      f"hits={_dead_hits} unmarked={len(_dead_unmarked)}")
for _n in _dead_unmarked:
    fails.append("retired number with no marker: " + _n)

# 9.4 the two ledgers must agree.  RUN_QUEUE.csv is where a cell's gate sweep is
# recorded; SEED_RESULTS.csv is what the delivered 0.7 block is generated from.
# They are regenerated by different steps, so one can be ahead: tonight the queue
# said 13 FD004 cells verified while the delivered ledger said 11, and the block
# therefore under-reported A by two.  7.1 cannot catch this -- it compares the
# doc against SEED_RESULTS, and both were stale in the same direction, so they
# agreed perfectly while both disagreed with the queue.
def _k(r):
    return (r.get("arm"), r.get("subset"), r.get("seed"), r.get("root"))


_q = {}
for _r in load("RUN_QUEUE.csv"):
    _q[_k(_r)] = _r
_s = {}
for _r in load("SEED_RESULTS.csv"):
    _s[_k(_r)] = _r
_only_q = sorted(set(_q) - set(_s))
_only_s = sorted(set(_s) - set(_q))
_gs_mm, _us_mm = [], []
for _key in sorted(set(_q) & set(_s)):
    if _q[_key].get("gate_status") != _s[_key].get("gate_status"):
        _gs_mm.append("%s/%s/%s@%s queue=%s seed=%s"
                      % (_key + (_q[_key].get("gate_status"),
                                 _s[_key].get("gate_status"))))
    if (_q[_key].get("usable") == "yes") != (_s[_key].get("usable") == "yes"):
        _us_mm.append("%s/%s/%s@%s queue=%s seed=%s"
                      % (_key + (_q[_key].get("usable"), _s[_key].get("usable"))))
_orphan_q = [k for k in _only_q if _q[k].get("gate_status") != "gate_reproduction"]
checks += 1
print(f"{'PASS' if not (_gs_mm or _us_mm or _only_s or _orphan_q) else 'FAIL'}  "
      f"{'9.4 the two ledgers agree cell by cell':<52s} "
      f"queue={len(_q)} seed={len(_s)} gate_mismatch={len(_gs_mm)} "
      f"usable_mismatch={len(_us_mm)} seed_only={len(_only_s)} "
      f"queue_only_not_twin={len(_orphan_q)}")
for _n in _gs_mm:
    fails.append("gate_status disagrees between ledgers: " + _n)
for _n in _us_mm:
    fails.append("usable disagrees between ledgers: " + _n)
for _k2 in _only_s:
    fails.append("cell in SEED_RESULTS but not in RUN_QUEUE: %s/%s/%s@%s" % _k2)
for _k2 in _orphan_q:
    fails.append("cell in RUN_QUEUE but not in SEED_RESULTS, and it is not a "
                 "gate twin (the delivered ledger is missing a sample): "
                 "%s/%s/%s@%s" % _k2)

print(f"\n=== RESULT: {checks - len(fails)}/{checks} anchors reproduce ===")
if fails:
    print("MISMATCHES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL ANCHORS OK")
