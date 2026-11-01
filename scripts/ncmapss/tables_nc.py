"""Emit the N-CMAPSS result tables from the three protocol JSONs.

Numbers are pulled from the files rather than transcribed, because the last round of this
project had to walk back two published claims that came from hand-copied comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FLEETS = ["DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07", "DS08a", "DS08c"]
PROTOS = ["pooled", "loso", "perfleet"]
MAIN = ["cens", "trunc", "emos_c", "emos_u", "homosc",
        "crude_exact", "crude_drop", "crude_km", "isot_exact", "isot_drop"]
LABEL = {"cens": "censored MLP (66p)", "trunc": "density-grid MLP (66p)",
         "emos_c": "affine censored (4p)", "emos_u": "affine density (4p)",
         "homosc": "constant sigma", "crude_exact": "CRUDE (125 kept)",
         "crude_drop": "CRUDE (capped dropped)", "crude_km": "KM-CRUDE",
         "isot_exact": "isotonic PIT", "isot_drop": "isotonic PIT (dropped)",
         "scp": "split conformal", "nscp": "normalised split conformal",
         "cqr": "CQR"}


def load():
    return {p: json.load(open(ROOT / "experiments/ncmapss" / f"cal_{p}.json"))
            for p in PROTOS}


def t1(R):
    L = ["\n### N1  骨干质量（test 集，单位：飞行周期）\n",
         "| 舰队 | n_dev | n_test | dev OOF RMSE | test RMSE | test R² | test 平均偏差 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    Q = json.load(open(ROOT / "experiments/ncmapss/preds/quality.json"))
    for f in FLEETS:
        q = Q[f]
        L.append(f"| {f} | {q['n_dev']} | {q['n_test']} | {q['rmse_dev_oof']:.2f} | "
                 f"{q['rmse_test']:.2f} | {q['r2_test']:.3f} | {q['mean_bias_test']:+.2f} |")
    L.append("")
    return L


def t2(R):
    L = ["\n### N2  CRPS（9 舰队平均，越低越好）与协议敏感性\n",
         "| 方法 | pooled | LOSO | per-fleet | 三协议极差 |",
         "|---|---:|---:|---:|---:|"]
    for m in MAIN:
        row = [np.mean([R[p]["fleets"][f]["methods"][m]["crps"] for f in FLEETS])
               for p in PROTOS]
        L.append(f"| {LABEL[m]} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | "
                 f"{max(row) - min(row):.3f} |")
    L.append("")
    return L


def t3(R):
    L = ["\n### N3  边际覆盖与宽度（cens，名义 90%）\n",
         "| 舰队 | n | pooled PICP / MPIW | LOSO PICP / MPIW | per-fleet PICP / MPIW |",
         "|---|---:|---|---|---|"]
    for f in FLEETS:
        cells = []
        for p in PROTOS:
            v = R[p]["fleets"][f]["methods"]["cens"]
            cells.append(f"{v['picp90']:.3f} / {v['mpiw90']:.1f}")
        L.append(f"| {f} | {R['pooled']['fleets'][f]['n']} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def band_matrix(R, proto):
    d = R[proto]
    names = []
    for f in FLEETS:
        for b in d["fleets"][f]["bands"]:
            if b["band"] not in names:
                names.append(b["band"])
    names.sort(key=lambda s: int(s.split(",")[0][1:]))
    L = [f"\n#### {proto}\n",
         "| 真实 RUL 分箱 | n | " + " | ".join(["cens", "trunc", "emos_c",
                                               "homosc", "crude_exact", "scp"]) + " |",
         "|---|---:|" + "---:|" * 6]
    for b in names:
        ns, acc = 0, None
        for f in FLEETS:
            for bb in d["fleets"][f]["bands"]:
                if bb["band"] == b:
                    ns += bb["n"]
                    vals = [bb[m]["picp"] * bb["n"] for m in
                            ("cens", "trunc", "emos_c", "homosc", "crude_exact", "scp")]
                    acc = vals if acc is None else [a + v for a, v in zip(acc, vals)]
        if not acc or ns == 0:
            continue
        L.append(f"| {b} | {ns} | " + " | ".join(f"{a/ns:.3f}" for a in acc) + " |")
    return L


def t4(R):
    L = ["\n### N4  分箱覆盖率（按真实 RUL，全部 test 行合并）\n"]
    L += band_matrix(R, "pooled")
    L += band_matrix(R, "perfleet")
    L.append("")
    return L


def t5(R=None):
    D = json.load(open(ROOT / "experiments/ncmapss/diag_nc.json"))
    L = ["\n### N5  分箱偏差（预测 − 真值），以及 MSE 最优单调映射能修掉多少\n",
         "| 舰队 | 原始最差 | 分箱 | 等渗（在 test 上拟合）最差 | 分箱 | 残余比例 |",
         "|---|---:|---|---:|---|---:|"]
    for f in FLEETS:
        v = D["per_fleet"][f]
        r0, r1 = v["raw"], v["iso_on_test"]
        L.append(f"| {f} | {r0['worst_abs_bias']:.1f} | {r0['worst_band']} | "
                 f"{r1['worst_abs_bias']:.1f} | {r1['worst_band']} | "
                 f"{r1['worst_abs_bias']/r0['worst_abs_bias']:.2f} |")
    p = D["pooled"]
    L.append(f"| **合并** | {p['raw backbone']['worst_abs_bias']:.1f} | "
             f"{p['raw backbone']['worst_band']} | "
             f"{p['isotonic on test (shift-free)']['worst_abs_bias']:.1f} | "
             f"{p['isotonic on test (shift-free)']['worst_band']} | "
             f"{p['isotonic on test (shift-free)']['worst_abs_bias']/p['raw backbone']['worst_abs_bias']:.2f} |")
    L.append("\n原始与等渗两列的分箱偏差全向量：\n")
    L.append("| 分箱 | 原始 | 等渗(test) |")
    L.append("|---|---:|---:|")
    for k, v in p["raw backbone"]["bands"].items():
        iv = p["isotonic on test (shift-free)"]["bands"].get(k, {})
        L.append(f"| {k} | {v['bias']:+.1f} | {iv.get('bias', float('nan')):+.1f} |")
    L.append("")
    return L


def t6(R):
    L = ["\n### N6  配对 bootstrap（以单元为块，2000 次），ΔCRPS = 该方法 − cens；"
         "负值表示该方法优于 cens\n",
         "| 对比 | 舰队数 | 平均 ΔCRPS | 95% CI | CI 排除 0 的实例 |",
         "|---|---:|---:|---|---:|"]
    for other in ["trunc", "emos_c", "homosc", "crude_km", "isot_exact"]:
        ds = []
        for f in FLEETS:
            pb = R["pooled"]["fleets"][f]["paired_bootstrap_vs_cens"].get(other)
            if pb:
                ds.append(pb)
        if not ds:
            continue
        v = np.array(ds)
        n_ex = int(np.sum((v[:, 1] > 0) | (v[:, 2] < 0)))
        L.append(f"| {LABEL[other]} − cens | {len(ds)} | {v[:,0].mean():+.3f} | "
                 f"[{v[:,1].mean():+.3f}, {v[:,2].mean():+.3f}] | {n_ex}/{len(ds)} |")
    L.append("")
    return L


def main():
    R = load()
    out = ["# N-CMAPSS 结果表（自动生成，勿手改）\n"]
    for f in (t1, t2, t3, t4, t5, t6):
        out += f(R)
    p = ROOT / "experiments/ncmapss/tables.md"
    p.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
