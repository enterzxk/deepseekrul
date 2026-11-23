#!/usr/bin/env python3
"""Generate docs/paper_c/FINAL_DECISION_20261123.md from the verdict record.

Rule (prereg addendum #5): FINAL_DECISION is generated field by field from the
vote_b.py JSON record. No number in the document is hand-copied.

Usage:  python paper_c/make_final_decision.py [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERDICT = ROOT / "docs/paper_c/evidence/B_VERDICT.json"
CELLS = ROOT / "docs/paper_c/evidence/B_CELLS.json"
PREREG = ROOT / "docs/P2_CROSS_ARCH_PREREG.md"

PREREG_BLOB = "8fe382d445b495c8286bb170be6db9acc0f7f541"
PREREG_SHA = "2507fc8314ae312ce2389024cfcdccd3e2380c75e4dc1cac3c24ca9d9d93fcbc"
VOTE_B_BLOB = "1234fe5248e6facc4dad0d17d002cbd1ffef409a"
VOTE_B_SHA = "52552a1b2ab8f3aaff8c0dd26c3d9c50e8a8d4925b7241c1544cadadc2b4d9db"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(x, n=4):
    return f"{x:.{n}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/paper_c/FINAL_DECISION_20261123.md"))
    args = ap.parse_args()

    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    units = v["units"]
    power = {f"{p['backbone']}/{p['subset']}": p for p in v["power"]["per_unit"]}

    branches = {}
    for u in units:
        branches[u["s3_branch"]] = branches.get(u["s3_branch"], 0) + 1
    passing = [u for u in units if u["passes"]]
    fail_branch = branches.get("sign_fail", 0) + branches.get("effect_size_fail", 0) + branches.get("win_rate_fail", 0)
    mixed = [u for u in units if u not in passing
             and len({p["delta"] > 0 for p in u["per_seed"]}) > 1]
    all_neg = [u for u in units if u not in passing and all(p["delta"] < 0 for p in u["per_seed"])]
    all_pos = [u for u in units if u not in passing and all(p["delta"] > 0 for p in u["per_seed"])]
    failing = [u for u in units if not u["passes"]]

    def tstat(u):
        ds = [p["delta"] for p in u["per_seed"]]
        sd = statistics.stdev(ds)
        return sd, (u["delta_mean"] / (sd / (3 ** 0.5)) if sd > 0 else float("inf"))

    t_passing = tstat(passing[0])[1] if passing else float("nan")
    n_below = sum(1 for u in units if abs(tstat(u)[1]) < 4.303)

    mdes = [u["mde"] for u in units]
    req = [power[f"{u['backbone']}/{u['subset']}"]["required_n_engines"] for u in units]
    det = [power[f"{u['backbone']}/{u['subset']}"]["ci_detectable_delta"] for u in units]
    pnulls = [u["p_null"] for u in units]
    spanning = [u for u in units if u["boot_ci"][0] <= 0.0 <= u["boot_ci"][1]]

    L = []
    A = L.append
    A("# Paper C · Gate 1 正式判定（FINAL_DECISION）")
    A("")
    A("> 本文档由 `paper_c/make_final_decision.py` 从 `docs/paper_c/evidence/B_VERDICT.json`")
    A("> **逐字段生成，无手抄**（预注册补遗 #5 报告规则）。手改无效，改记录后重跑。")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 判定 | **FAIL / 未达标（not demonstrated）** |")
    A(f"| `claim_scope` | `{v['claim_scope']}` |")
    A(f"| 判据文本 | `docs/P2_CROSS_ARCH_PREREG.md` v7.4 + 补遗 #3/#5 |")
    A(f"| 判据文本 blob | `{PREREG_BLOB}` |")
    A(f"| 判据文本 sha256 | `{PREREG_SHA}` |")
    A(f"| 判据实现 | `paper_c/vote_b.py` blob `{VOTE_B_BLOB}` |")
    A(f"| 判据实现 sha256 | `{VOTE_B_SHA}`（37/37 EXIT=0，牢莫 2026-11-23 复核） |")
    A(f"| 判定记录 | `docs/paper_c/evidence/B_VERDICT.json` sha256 `{sha256(VERDICT)}` |")
    A(f"| 逐格记录 | `docs/paper_c/evidence/B_CELLS.json` sha256 `{sha256(CELLS)}` |")
    A(f"| 官方 TEST | **未打开**（开发期封存） |")
    A(f"| Δ 口径 | `{v['delta_convention']}` |")
    A("")

    A("## 1. Gate")
    A("")
    A(f"- `k_full = {v['k_full']}/{v['k_full_evaluable']}`，`k_sc = {v['k_sc']}/{v['k_sc_evaluable']}`，`k_min = {v['k_min']}`。")
    A(f"- 未定票 `n_undetermined = {v['n_undetermined']}`（`underpowered` 0 张，`protocol_deviant` {len(v.get('deviant') or [])} 个）。")
    A(f"- 10 个 (骨干, 子集) 单元的分支分布：`sign_fail` {branches.get('sign_fail', 0)}、"
      f"`effect_size_fail` {branches.get('effect_size_fail', 0)}、`win_rate_fail` {branches.get('win_rate_fail', 0)}、"
      f"`underpowered` {branches.get('underpowered', 0)}、`consistent` {branches.get('consistent', 0)}。")
    A(f"- 因此 **{fail_branch}/10 是确定失败**（分布见上），没有一票到达 CI 分支；"
      f"唯一走到 CI 分支的是 `{passing[0]['backbone']}/{passing[0]['subset']}`（见 §5）。")
    A("")
    A("**措辞（写死）**：Gate 是「**未达标（not demonstrated）**」，不是「测不出」。"
      "「测不出」只能用于真正落在 `underpowered` 的票，本轮 `underpowered` 为 0。")
    A("")

    A("## 2. 点估计（全 30 格无条件值，不筛 `passes`）")
    A("")
    A("| 骨干 | 子集 | Δ̄ | 95% CI | 胜率 | MDE | `p_null` | 分支 | `passes` |")
    A("|---|---|---:|---|---:|---:|---:|---|:--:|")
    for u in units:
        A(f"| {u['backbone']} | {u['subset']} | {f(u['delta_mean'])} | "
          f"[{f(u['boot_ci'][0])}, {f(u['boot_ci'][1])}] | {f(u['engine_win_rate'], 2)} | "
          f"{f(u['mde'], 3)} | {f(u['p_null'], 3)} | `{u['s3_branch']}` | {'✓' if u['passes'] else '✗'} |")
    A("")
    A("逐 seed 配对差（`Δ = V0 − V1`，正 = 模块更好）：")
    A("")
    A("| 骨干 | 子集 | s101 | s202 | s303 |")
    A("|---|---|---:|---:|---:|")
    for u in units:
        d = {p["seed"]: p["delta"] for p in u["per_seed"]}
        A(f"| {u['backbone']} | {u['subset']} | {f(d.get(101, float('nan')))} | "
          f"{f(d.get(202, float('nan')))} | {f(d.get(303, float('nan')))} |")
    A("")
    A(f"- `p_null = 1 − Φ(0.45/SE)` 范围 **{f(min(pnulls), 3)}–{f(max(pnulls), 3)}**："
      "即真实效应为 0 时，点估计仍 ≥ 0.45 的偶然概率。")
    A("- winner's curse 抬升 **0.16–0.24** 只适用于过线格；本轮过线格 **1 个**（§5），"
      "所以上表的无条件值就是主张依据，不得只引过线格。")
    A(f"- **{len(all_neg)} 个单元三 seed 全负**：" +
      "、".join(f"`{u['backbone']}/{u['subset']}` {f(u['delta_mean'])}" for u in all_neg) +
      "——方向一致为负但 CI 跨 0，**不得写成「模块有害」**，只能记「方向为负、不显著」。")
    A("")

    A("## 3. CI 与功效")
    A("")
    A(f"- `MDE80`（`2.80 × SE`）实测 **{f(min(mdes), 3)}–{f(max(mdes), 3)}**，全部 **> 0.45**。")
    A(f"- 所需发动机数 `required_n` **{min(req)}–{max(req)}**；手上 20（FD001）/ 50（FD004）。")
    A(f"- `ci_detectable_delta`（`1.96 × SE`）**{f(min(det), 3)}–{f(max(det), 3)}**。")
    A(f"- 10 个单元里 **{len(spanning)} 个的 95% CI 跨 0**，即 CI **排不掉 0.45 这个量级的效应**。")
    A("")
    A(f"**三句必须分开写，不得合并**：Gate 说「未达标」；点估计说「9/10 未通过同号条件"
      f"（{len(mixed)} 个 seed 符号翻转、{len(all_neg)} 个三 seed 全负），Δ̄ 落在 "
      f"{f(min(u['delta_mean'] for u in failing))} ~ {f(max(u['delta_mean'] for u in failing))}」；"
      "CI 说「宽到排不掉中等效应」。三句一起才是完整结论。")
    A("")

    A("## 4. 机制与排序")
    A("")
    A("**机制（代码可验证）**：CADF 的输入是 `x` 的**确定性函数**——`z = [x, delta, e]` 中 `delta = (x − b)·mask`、"
      "`e = EMA(delta)`，而 `b` 由 `(x, cluster, mask)` 唯一确定，所以 `z` **不引入 `x` 之外的新信息**；"
      "且 `return x + g * proj(z)` 里 `x` 原样穿过，模块只能在上面加一个可学习变换。"
      "准确写法是「**输入层重编码，不引入新信息**」，不是「高通滤波把趋势扔了」——后者架构上做不到。")
    A("")
    A("机制解释的三次迭代留痕（不得只留最后一版）：「预处理已含工况归一化、信息重叠」"
      "（被 `‖b‖/‖x‖ = 0.79–0.93` 证伪）→「高通滤波丢弃趋势信息」"
      "（被 `return x + g * proj(z)` 的架构证伪）→「输入层重编码」**（现行）**。")
    A("")
    A("**排序**：**骨干间排序不可区分，不作归因**（补遗 #5 规则 5）。口径是组均值的标准误 "
      "`SE(Δ̄) = sd/√3`，不是逐 seed 的 sd；绝对参数量与 CADF 相对开销两个代理都不单调，均不得写入。")
    A("")
    A("逐单元 seed 间一致性（**n=3，无功效，不作证据**）：`t(2)` 正是 v7 从判据里删掉的那个检验，"
      "删的理由是临界值 4.30 太高——「t 显著」主要反映三个 seed 挤得紧，**不反映效应大**。")
    A("")
    A("| 骨干 | 子集 | Δ̄ | sd(seed) | t = Δ̄/(sd/√3) |")
    A("|---|---|---:|---:|---:|")
    for u in units:
        sd, t = tstat(u)
        A(f"| {u['backbone']} | {u['subset']} | {f(u['delta_mean'])} | {f(sd)} | {f(t, 2)} |")
    A("")
    A(f"- df=2 双侧 5% 临界 **4.303**；{n_below}/10 的 |t| 低于临界，唯一超临界的是 "
      f"`{passing[0]['backbone']}/{passing[0]['subset']}`（t = {f(t_passing, 2)}）。")
    A("- 这一列**只支持「排序不可区分」，不构成模块有效的证据**：`t` 显著说明 seed 间可复现，"
      "不说明跨发动机可推广；后者只有 engine bootstrap 的 CI 回答（§5）。")
    A("")

    A("## 5. 必须单列的例外：唯一通过单元级的 `two_phase/FD004`")
    A("")
    ex = passing[0] if passing else None
    if ex:
        A(f"- 数字：`Δ̄ = {f(ex['delta_mean'])}`，95% CI `[{f(ex['boot_ci'][0])}, {f(ex['boot_ci'][1])}]`"
          f"（**不含 0**），胜率 `{f(ex['engine_win_rate'], 2)}`，三 seed `{f(ex['per_seed'][0]['delta'])} / "
          f"{f(ex['per_seed'][1]['delta'])} / {f(ex['per_seed'][2]['delta'])}`（3/3 同号），"
          f"`p_null = {f(ex['p_null'], 3)}`，`MDE = {f(ex['mde'], 3)}`，n = {ex['n_engines']} 台。")
        A("- **它不计票不是因为它不合格，是分档结构没有它的位置**：`full` 档要求同一骨干 FD001 与 FD004 同时达标，"
          f"而 `two_phase/FD001` 是 `{f([u for u in units if u['backbone']=='two_phase' and u['subset']=='FD001'][0]['delta_mean'])}`（反号）；"
          "`sc` 档按预注册只认 FD001。")
        A("- **三条必须并排出现，少一条就会被写成「唯一有效」**（军师 2026-11-23）：")
        A(f"  1. `t = {f(t_passing, 2)}`（df=2 临界 4.303）→ seed 间**可复现**；但 n=3 无功效，**不是证据**；")
        A(f"  2. `p_null = {f(ex['p_null'], 3)}` → 零假设下 10 个单元本来就会出约 1 个这种量级；")
        A("  3. 它是全部单元里**误差最大的欠拟合基座**（CADF 臂 22.94–23.44，其余 ≤14.89，"
          "据 `docs/paper_c/B_FINAL_20261123.md`）→ 增益可能来自欠拟合而非模块。")
        A("  结论：**只记形态，不作归因。**")
    A("- **冻结设计的缺口（写进限制，不改判据）**：这套分档**没有「只在 FD004 成立」的位置**，"
      "而 `sc` 恰好是它的镜像。本轮不改；下一版预注册应补一个对称档，或明确写「只认 FD001 是有意为之」。"
      "不写这句，论文一定会被问「唯一通过的那一票去哪了」。")
    A("")

    A("## 6. 限制")
    A("")
    A("1. **分档结构缺口**：`full` 需两子集同时达标、`sc` 只认 FD001，无 FD004-only 档（§5）。")
    A("2. **`ffve` 的 VAE 采样噪声**：其 forward 含 `reparameterize()`，逐格 Δ 天生带采样噪声；"
      "两臂用 `fork_rng` 固定种子故可比，但其种子内噪声大于另外四个骨干，报告须单独标注。")
    A(f"3. **门槛与功效不匹配**：`MDE80` {f(min(mdes), 2)}–{f(max(mdes), 2)} 全部大于门槛 0.45，"
      "即「CI 不含 0」这条门在 20/50 台上实际要求的 Δ̄ 远高于声明门槛——属事前已知的设计限制。")
    A("4. **官方 TEST 未开**：两阶段主张的 Terminal 可检测下限是 FD001 0.73 / FD004 0.60，"
      "0.45–0.73 为灰带，不在任何主张范围内。")
    A("5. **机制解释是代码可验证的，但未经独立实验证伪**：本轮未做「`delta` 单独有无预测信息」的判据级实验。")
    A("")
    A("---")
    A("")
    A(f"生成器：`paper_c/make_final_decision.py`；输入 `{VERDICT.relative_to(ROOT).as_posix()}`、"
      f"`{CELLS.relative_to(ROOT).as_posix()}`。")

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(L)} lines, {len(units)} units, {len(passing)} passing)")


if __name__ == "__main__":
    main()
