"""Build every table in CALIBRATION_V2_RESULTS.md straight from compare.json.

Hand-copying twelve instances by seven methods is how transcription errors get into a paper,
so every number that appears in the report is printed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["A", "BE", "PE"]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MAIN = ["cens", "trunc", "emos_c", "crude_exact", "crude_km", "isot_exact", "homosc"]
LABEL = {"cens": "censored MLP (66p)", "trunc": "density-on-grid (66p)",
         "emos_c": "censored affine (4p)", "crude_exact": "CRUDE, cap as exact",
         "crude_km": "CRUDE, KM-censored", "isot_exact": "isotonic PIT, cap as exact",
         "homosc": "constant sigma (1p)"}


def load():
    out = {}
    for a in ARMS:
        out[a] = json.load(open(ROOT / "experiments" / "cal" / f"v2_{a}" / "compare.json"))
    return out


def main():
    R = load()
    lines = []

    # ---- table 1: every method, arm A, all four fleets ---------------------------------
    lines.append("### T1  全方法对比（A 臂），宽度各自在交叉拟合 dev 上校准到 PICP90=0.90\n")
    lines.append("| 子集 | 方法 | CRPS | E\\|X−y\\| | NLL | PICP90 | MPIW90 | Winkler90 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in SUBSETS:
        m = R["A"]["subsets"][s]["methods"]
        for k in ["cens", "trunc", "emos_c", "homosc", "crude_exact", "crude_km",
                  "isot_exact", "scp", "nscp", "cqr"]:
            v = m[k]
            crps = "—" if v["crps"] is None else f"{v['crps']:.2f}"
            l1 = "—" if v.get("crps_l1") is None else f"{v['crps_l1']:.2f}"
            nll = "—" if v["nll"] is None else f"{v['nll']:.3f}"
            lines.append(f"| {s} | {LABEL.get(k, k)} | {crps} | {l1} | {nll} | "
                         f"{v['picp90']:.3f} | {v['mpiw90']:.1f} | {v['wink90']:.1f} |")
        lines.append("| | | | | | | | |")

    # ---- table 2: the 12-instance CRPS grid --------------------------------------------
    lines.append("\n### T2  12 实例 CRPS（宽度按各自 dev 校准）\n")
    hdr = " | ".join(LABEL[k].split(" (")[0] for k in MAIN)
    lines.append(f"| 骨干 | 子集 | {hdr} |")
    lines.append("|---|---|" + "---:|" * len(MAIN))
    for a in ARMS:
        for s in SUBSETS:
            m = R[a]["subsets"][s]["methods"]
            vals = " | ".join(f"{m[k]['crps']:.2f}" for k in MAIN)
            lines.append(f"| {a} | {s} | {vals} |")

    # ---- per-method win counts and means ----------------------------------------------
    lines.append("\n### T3  逐方法汇总（12 实例）\n")
    lines.append("| 方法 | 平均 CRPS | 中位 CRPS | 平均 PICP90 | CRPS 最优次数 |")
    lines.append("|---|---:|---:|---:|---:|")
    rows = {}
    for k in MAIN:
        c = [R[a]["subsets"][s]["methods"][k]["crps"] for a in ARMS for s in SUBSETS]
        p = [R[a]["subsets"][s]["methods"][k]["picp90"] for a in ARMS for s in SUBSETS]
        rows[k] = (float(np.mean(c)), float(np.median(c)), float(np.mean(p)))
    best = {}
    for a in ARMS:
        for s in SUBSETS:
            m = R[a]["subsets"][s]["methods"]
            b = min(MAIN, key=lambda k: m[k]["crps"])
            best[b] = best.get(b, 0) + 1
    order = sorted(MAIN, key=lambda k: rows[k][0])
    for k in order:
        mu, md, pc = rows[k]
        lines.append(f"| {LABEL[k]} | {mu:.2f} | {md:.2f} | {pc:.3f} | {best.get(k, 0)} |")

    # ---- table 4: paired bootstrap on the contrasts that carry the argument -----------
    lines.append("\n### T4  配对 bootstrap（发动机为单位，2000 次），ΔCRPS = 该方法 − cens\n")
    lines.append("| 对比 | 实例数 | 平均 ΔCRPS | 各实例 CI 汇总 | 判读 |")
    lines.append("|---|---:|---:|---|---|")
    for other in ["trunc", "emos_c", "crude_exact", "crude_km", "isot_exact", "homosc"]:
        los, his, means = [], [], []
        for a in ARMS:
            for s in SUBSETS:
                pb = R[a]["subsets"][s]["paired_bootstrap_vs_cens"].get(other)
                if not pb or pb.get("crps") is None:
                    continue
                lo, hi, mn = pb["crps"]
                los.append(lo); his.append(hi); means.append(mn)
        if not means:
            continue
        los, his, means = np.array(los), np.array(his), np.array(means)
        n_excl = int(np.sum((los > 0) | (his < 0)))
        if n_excl == len(means) and means.mean() > 0:
            verdict = "**cens 显著更好（全部实例）**"
        elif n_excl == len(means):
            verdict = "**对手显著更好（全部实例）**"
        elif los.mean() > 0:
            verdict = "cens 略优"
        elif his.mean() < 0:
            verdict = "对手略优"
        else:
            verdict = "无差别"
        lines.append(f"| {LABEL.get(other, other)} | {len(means)} | {means.mean():+.2f} | "
                     f"CI 中位 [{np.median(los):+.2f}, {np.median(his):+.2f}] | "
                     f"{verdict}；{n_excl}/{len(means)} 个实例 CI 排除 0 |")

    # ---- table 5: conditional coverage, the part marginal PICP hides ------------------
    lines.append("\n### T5  按真实 RUL 分箱的覆盖率与偏差（cens），三骨干四舰队汇总\n")
    lines.append("| 分箱 | 发动机数 | 池化覆盖率 | Wilson 95% CI | 平均偏差 | 偏差范围 |")
    lines.append("|---|---:|---:|---|---:|---|")
    for band in ["[0,25)", "[25,50)", "[50,75)", "[75,100)", "[100,126)"]:
        hits, ns, biases = 0.0, 0, []
        for a in ARMS:
            for s in SUBSETS:
                for b in R[a]["subsets"][s]["bands"]:
                    if b["band"] == band:
                        v = b["cens"]
                        hits += v["picp"] * b["n"]
                        ns += b["n"]
                        biases.append(v["bias"])
        p = hits / ns
        z = 1.96
        d = 1 + z * z / ns
        c = (p + z * z / (2 * ns)) / d
        h = z * np.sqrt(p * (1 - p) / ns + z * z / (4 * ns * ns)) / d
        lines.append(f"| {band} | {ns} | {p:.3f} | [{c - h:.3f}, {c + h:.3f}] | "
                     f"{np.mean(biases):+.2f} | "
                     f"[{min(biases):+.2f}, {max(biases):+.2f}] |")

    # ---- table 6: LOSO vs pooled, does the calibrator transfer across fleets? ---------
    loso = {}
    for a in ARMS:
        f = ROOT / "experiments" / "cal" / f"v2_loso_{a}" / "loso.json"
        if f.exists():
            loso[a] = json.load(open(f))
    if loso:
        lines.append("\n### T6  LOSO（在三个舰队 dev 上拟合，在第四个舰队 test 上评价）"
                     "对比 pooled\n")
        lines.append("| 骨干 | 目标舰队 | pooled CRPS | LOSO CRPS | Δ | pooled PICP90 | LOSO PICP90 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for a in ARMS:
            if a not in loso:
                continue
            for s in SUBSETS:
                po = R[a]["subsets"][s]["methods"]["cens"]
                lo_ = loso[a][s]["eval"]["methods"]["cens"]
                lines.append(f"| {a} | {s} | {po['crps']:.2f} | {lo_['crps']:.2f} | "
                             f"{lo_['crps'] - po['crps']:+.2f} | {po['picp90']:.3f} | "
                             f"{lo_['picp90']:.3f} |")

    # ---- table 7: per-subset, the only regime where 66 parameters run short of data ----
    psub = {}
    for a in ARMS:
        f = ROOT / "experiments" / "cal" / f"v2_persub_{a}" / "per_subset.json"
        if f.exists():
            psub[a] = json.load(open(f))
    if psub:
        lines.append("\n### T7  per-subset（只用本舰队 dev 拟合）对比 pooled\n")
        lines.append("| 方法 | per-subset 平均 CRPS | pooled 平均 CRPS | 差 |")
        lines.append("|---|---:|---:|---:|")
        for k in MAIN:
            a1 = np.mean([psub[a][s]["eval"]["methods"][k]["crps"]
                          for a in ARMS if a in psub for s in SUBSETS])
            a2 = np.mean([R[a]["subsets"][s]["methods"][k]["crps"]
                          for a in ARMS for s in SUBSETS])
            lines.append(f"| {LABEL[k]} | {a1:.2f} | {a2:.2f} | {a1 - a2:+.2f} |")
        lines.append("\n各舰队 dev 规模：FD001/003 为 100 台，FD002 为 260 台，FD004 为 249 台。\n")
        lines.append("| 骨干 | 子集 | n_dev | censored MLP | constant sigma | 层是否胜过无层 |")
        lines.append("|---|---|---:|---:|---:|---|")
        for a in ARMS:
            if a not in psub:
                continue
            for s in SUBSETS:
                m = psub[a][s]["eval"]["methods"]
                n = psub[a][s]["meta"]["n_dev"]
                verdict = "是" if m["cens"]["crps"] < m["homosc"]["crps"] else "**否**"
                lines.append(f"| {a} | {s} | {n} | {m['cens']['crps']:.2f} | "
                             f"{m['homosc']['crps']:.2f} | {verdict} |")

    txt = "\n".join(lines)
    (ROOT / "experiments" / "cal" / "v2_tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
