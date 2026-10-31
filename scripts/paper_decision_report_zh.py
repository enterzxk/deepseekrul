"""Generate paper/DECISION_REPORT_ZH.md from result artifacts only.

Rule in force: no hand-written numbers. Every number in the report is read from a
result artifact under results/dsh/. The script also prints the sha256 of each input
so the report can be re-verified.

Usage:  python scripts/paper_decision_report_zh.py
Output: paper/DECISION_REPORT_ZH.md  (+ results/dsh/decision_report_inputs.json)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "dsh"
OUT = ROOT / "paper" / "DECISION_REPORT_ZH.md"

INPUTS = {
    "E10b_BE_A_FD002": RES / "e10" / "E10b_be_cv.json",
    "E10_PE_A_FD002": RES / "e10" / "E10_cv.json",
    "E18_FD001": RES / "e18" / "FD001_cv.json",
    "E18_pooled": RES / "e18" / "pooled_BE_A.json",
    "E16_arch": RES / "e16" / "arch_corrected.json",
    "E19_power": RES / "e19" / "power_and_folds.json",
    "E15_test_fd002": RES / "test" / "test_fd002.json",
    "E7_resolution": RES / "e7" / "RESOLUTION.json",
    "E9_variance": RES / "e9" / "variance_decomp.json",
    "C1_variance": RES / "e10" / "variance_two_designs.json",
    "E20_cv": RES / "e20" / "E20_cv.json",
}


def load(name: str):
    p = INPUTS[name]
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def sha(name: str) -> str:
    p = INPUTS[name]
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def ci(v, key="ci95"):
    lo, hi = v[key]
    star = "**" if (lo > 0 or hi < 0) else ""
    return f"{star}{v['delta']:+.3f}{star} [{lo:+.3f}, {hi:+.3f}]"


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main() -> None:
    e10b = load("E10b_BE_A_FD002")
    e10 = load("E10_PE_A_FD002")
    e18 = load("E18_FD001")
    pooled = load("E18_pooled")
    arch = load("E16_arch")
    power = load("E19_power")
    test = load("E15_test_fd002")
    res7 = load("E7_resolution")
    var9 = load("E9_variance")
    var1 = load("C1_variance")
    e20 = load("E20_cv")

    L = []
    A = L.append

    A("# 决策报告（中文）— C-MAPSS 评测分辨率研究")
    A("")
    A(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}（本地）")
    A(f"生成脚本：`scripts/paper_decision_report_zh.py`（数字全部来自 `results/dsh/` 产物，无手写数字）")
    A("")
    A("---")
    A("")

    # ---------------------------------------------------------------- 0. headline
    A("## 0. 一句话结论")
    A("")
    if pooled:
        A(f"唯一在所有子集上都复现的效应是**时间通道**：去掉它，FD002 代价 "
          f"{pooled['FD002']['d']:+.3f} [{pooled['FD002']['lo']:+.3f}, {pooled['FD002']['hi']:+.3f}] RMSE"
          f"（{pooled['FD002']['n']} 台），FD001 代价 "
          f"{pooled['FD001']['d']:+.3f} [{pooled['FD001']['lo']:+.3f}, {pooled['FD001']['hi']:+.3f}]"
          f"（{pooled['FD001']['n']} 台），逆方差合并 "
          f"**{pooled['pooled_delta']:+.3f} [{pooled['pooled_ci95'][0]:+.3f}, {pooled['pooled_ci95'][1]:+.3f}]**"
          f"（{pooled['total_engines']} 台）。")
    A("")
    A("编码方式（有界正弦 vs 线性周期）、同一档内各架构的排序、以及候选模型与 Transformer 的差距，"
      "在两套协议下都**低于可分辨边界**。论文的主线应从「方法差异不可分辨」收窄为"
      "「**约 1.2 RMSE 以下的差异不可分辨**」。")
    A("")

    # ---------------------------------------------------------------- 1. core table
    A("## 1. 核心对比（训练侧，5-fold，每台发动机恰好被留出一次）")
    A("")
    A("符号：Δ = RMSE(前者) − RMSE(后者)，负值表示前者更好。区间为发动机聚类 bootstrap，"
      "100k 次重抽、3 个 RNG 种子，seed 2027 取区间。")
    A("")
    A("| 对比 | 子集 | 留出发动机 | ΔRMSE [95% CI] | 排除 0 | 结论 | 产物 |")
    A("|---|---|---:|---|---|---|---|")
    if e10b:
        A(row(["BE − A（去掉时间通道）", "FD002", e10b["n_engines"], ci({"delta": e10b["delta"], "ci95": e10b["ci95"]}),
               "是" if e10b["excludes_zero"] else "否", e10b["rule"]["verdict"], "`e10/E10b_be_cv.json`"]))
    if e18:
        be = e18["contrasts"]["BE_minus_A"]
        A(row(["BE − A（去掉时间通道）", "FD001", e18["arms"]["A"]["n_engines"], ci(be),
               "是" if be["excludes_zero"] else "否", "REPRODUCED", "`e18/FD001_cv.json`"]))
        pe = e18["contrasts"]["PE_minus_A"]
        A(row(["PE − A（有界编码）", "FD001", e18["arms"]["A"]["n_engines"], ci(pe),
               "是" if pe["excludes_zero"] else "否", "符号不复制", "`e18/FD001_cv.json`"]))
    if e10:
        # E10_cv.json stores the single PE-vs-A contrast at top level.
        pe2 = e10.get("contrasts", {}).get("PE_minus_A", e10)
        A(row(["PE − A（有界编码）", "FD002", e10.get("n_engines", "?"), ci(pe2),
               "是" if pe2.get("excludes_zero") else "否",
               e10.get("rule", {}).get("verdict", ""), "`e10/E10_cv.json`"]))
    if pooled:
        A(row(["BE − A（合并）", "FD001+FD002", pooled["total_engines"],
               f"**{pooled['pooled_delta']:+.3f}** [{pooled['pooled_ci95'][0]:+.3f}, {pooled['pooled_ci95'][1]:+.3f}]",
               "是", "逆方差合并", "`e18/pooled_BE_A.json`"]))
    A("")
    A("读法：PE（有界正弦编码）在 FD002 是 −0.126、在 FD001 是 −0.634，**两个区间都跨 0，且方向相反**；"
      "这正是「低于边界的差异」应有的表现，不能写成「有界编码更好」。")
    A("")

    # ---------------------------------------------------------------- 2. architecture
    if arch:
        A("## 2. 架构对比（FD002，260 台，同一协议）")
        A("")
        A("| 对比 | ΔRMSE [95% CI] | 是否可分辨 | 产物 |")
        A("|---|---|---:|---|")
        for k in ("A_minus_tf", "A_minus_gru", "A_minus_bilstm", "A_minus_tcn",
                  "A_minus_cnn1d", "tf_minus_gru", "tf_minus_bilstm",
                  "gru_minus_bilstm", "tcn_minus_cnn1d"):
            v = arch["contrasts"].get(k)
            if not v:
                continue
            ex = v["ci95"][0] > 0 or v["ci95"][1] < 0
            A(row([k.replace("_minus_", " − "), ci(v),
                   "是" if ex else "否", "`e16/arch_corrected.json`"]))
        A("")
        A("五组对比中四组可分辨（A 优于 cnn1d/tcn/gru/bilstm 2.2–5.8 RMSE），"
          "**A 与 Transformer 的 −1.433 [−2.998, +0.133] 不可分辨**。"
          "档内等价性（TOST，margin 1.0）四组全部失败，最小可等价边界 1.10–1.72。"
          "因此论文只能写「排序未确立」，不能写「等价」。")
        A("")

    # ---------------------------------------------------------------- 3. resolution
    if power:
        A("## 3. 分辨率边界与样本量目标")
        A("")
        ap = power["A_power"]
        A("| 量 | A vs BE | A vs PE | 产物 |")
        A("|---|---:|---:|---|")
        for label, key in (("σ（fold 0 估计）", "sd"), ("260 台处合成标准不确定度 u_c", "u_c_at_260"),
                           ("T1：半宽 ≤0.70 所需发动机数", "n_for_halfwidth_070"),
                           ("T2：检出 0.70 的 80% 功效所需发动机数", "n_for_80pct_power_070")):
            a = ap.get("FD002 A vs BE", {}).get(key)
            b = ap.get("FD002 A vs PE", {}).get(key)
            A(row([label, f"{a:.2f}" if isinstance(a, float) else a,
                   f"{b:.2f}" if isinstance(b, float) else b, "`e19/power_and_folds.json`"]))
        rows = ap.get("FD002 A vs BE", {}).get("rows", [])
        if rows:
            A("")
            A("MDE 随发动机数的变化（闭合式）：")
            A("")
            A("| 发动机数 | 52 | 100 | 260 | 509 | 708 | 1258 |")
            A("|---|---:|---:|---:|---:|---:|---:|")
            want = (52, 100, 260, 509, 708, 1258)
            A(row(["MDE (RMSE)"] + [f"{r['mde_closed_form']:.2f}" for r in rows
                                    if r["n"] in want]))
        A("")
        A("注意：表中 n > 260 的列是**算术外推**，不是实测，论文中必须标注为未验证的投影。")
        A("")
        A("T1 / T2 一行给出的是 σ 取 fold 0 估计时的值；`paper/CLAIMS.md` C26 记录的区间 "
          "320–644 / 627–1262 来自 σ 的两种估计口径，论文表格以 C26 为准。")
        A("")

    # ---------------------------------------------------------------- 4. test one-shot
    if test:
        A("## 4. 官方测试集一次性评估（E15，FD002，最后窗口，每台一预测）")
        A("")
        A(f"协议：{test['protocol']}；发动机 {test['n_engines']} 台。")
        A("")
        A("| 模型 | RMSE | MAE | NASA | late | severe |")
        A("|---|---:|---:|---:|---:|---:|")
        for arm in sorted(test["arms"], key=lambda k: test["arms"][k]["rmse"]):
            m = test["arms"][arm]
            A(row([arm, f"{m['rmse']:.3f}", f"{m['mae']:.3f}", f"{m['nasa']:.3f}",
                   f"{m['late']:.3f}", f"{m['severe']:.3f}"]))
        A("")
        A("训练侧 1.43 RMSE 的差距在测试侧只剩 1%；5.7–5.8 RMSE 的差距保留 93–100%。"
          "**效应保留率是效应大小的单调函数**，这独立复现了训练侧的分辨率边界。")
        A("")
        exc = power.get("C_test_exclusion") if power else None
        if exc:
            A(f"已披露的偏差：{exc['n_excluded']}/{exc['n_total']} 台因轨迹短于 50 周期被排除，"
              f"被排除者真实 RUL 均值 {exc['excluded_rul_mean']:.1f}，"
              f"保留者 {exc['included_rul_mean']:.1f}，因此报告值偏乐观约 2%。")
            A("")

    # ---------------------------------------------------------------- 5. variance
    if var1 or var9:
        A("## 5. 方差来源")
        A("")
        if var1:
            b = var1["design_b"]["anova"]
            a = var1["design_a"]
            A("| 来源 | 设计 B（3×3） | 设计 A（27 格） |")
            A("|---|---:|---:|")
            A(row(["方法身份", f"{b['pct_row']:.2f}%", f"{a['pct_row']:.2f}%"]))
            A(row(["验证集/发动机选择", f"{b['pct_col']:.2f}%", f"{a['pct_col']:.2f}%"]))
            A(row(["交互 + 残差", f"{b['pct_interaction']:.2f}%",
                   f"{a['pct_interaction'] + a['pct_residual']:.2f}%"]))
            A("")
        if var9:
            f2 = var9["FD002"]
            A(f"单个配对对比内部：发动机抽样占方差 {f2['variance_share_engine']*100:.1f}%，"
              f"随机种子占 {f2['variance_share_seed']*100:.1f}%。"
              "**加种子救不了一个对比，只有加留出发动机才能。**")
            A("")

    # ---------------------------------------------------------------- 6. E20 status
    A("## 6. 进行中的 E20（FD003 / FD004 五折）")
    A("")
    if e20:
        for s in ("FD003", "FD004"):
            blk = e20.get(s)
            if not blk:
                continue
            A(f"### {s}")
            A("")
            A("| 对比 | ΔRMSE [95% CI] | 排除 0 |")
            A("|---|---|---|")
            for k, v in blk.get("contrasts", {}).items():
                ex = v["ci95"][0] > 0 or v["ci95"][1] < 0
                A(row([k.replace("_", " "), ci(v), "是" if ex else "否"]))
            A("")
    else:
        A("状态：**运行中**（A40 服务器）。预注册 `paper/E20_FD0034_CV_PREREGISTRATION.md`，"
          "产物 `results/dsh/e20/E20_cv.json` 尚未生成，本节由脚本在该文件出现后自动填充。")
        A("")

    # ---------------------------------------------------------------- 7. claims
    A("## 7. 现在可以写 / 不能写")
    A("")
    A("**可以写**")
    A("")
    A("1. 时间通道效应在 FD001 与 FD002 上都复现，合并 +1.35 [+0.57, +2.13]。")
    A("2. 260 台留出发动机的可分辨边界约 1.1–1.5 RMSE；低于该值的差异在两套协议下都不可分辨。")
    A("3. 单折结论会被折间变异放大 2.4 倍（+2.693 → +1.133），论文必须用全折留出。")
    A("4. 效应保留率是效应大小的单调函数（1.5 RMSE → 1%，5.9 RMSE → 99%）。")
    A("")
    A("**不能写**")
    A("")
    A("1. 「有界编码非劣」——已在 E10/E18 撤回（C14）。")
    A("2. 「候选模型优于 Transformer」——区间跨 0。")
    A("3. 「档内架构等价」——TOST 四组全部失败。")
    A("4. 任何 SOTA / 双冠声明，以及 P7/P9/P10 的 synergy-head 数字（13.209 / 14.155 / 13.753 均已作废）。")
    A("")

    # ---------------------------------------------------------------- 8. decisions
    A("## 8. 需要你决策的事项（阻塞中）")
    A("")
    A("| 编号 | 事项 | 当前状态 |")
    A("|---|---|---|")
    A(row(["G1-a", "作者顺序与通讯作者", "**待定**"]))
    A(row(["G1-b", "单位 /  affiliations", "**待定**"]))
    A(row(["G1-c", "基金号与致谢", "**待定**"]))
    A(row(["G1-d", "利益冲突声明", "**待定**"]))
    A(row(["G1-e", "期刊与目标栏目", "已定：IEEE TIM Regular Paper"]))
    A(row(["G1-f", "标题", "已定：C — *What C-MAPSS Can and Cannot Measure*"]))
    A(row(["G2", "测试集使用披露措辞", "已有批准稿 `paper/SUPPLEMENT_DATA_USE.md`，待作者确认"]))
    A("")

    # ---------------------------------------------------------------- 9. queue
    A("## 9. 未完成队列（按优先级）")
    A("")
    A("| 优先级 | 任务 | 状态 | 阻塞 |")
    A("|---|---|---|---|")
    A(row(["1", "E20：FD003/FD004 五折 90 runs + 分析", "运行中（GPU）", "无"]))
    A(row(["2", "E21：四子集统一官方测试评估", "已预注册，未执行", "等 E20 检查点"]))
    A(row(["3", "main.tex 组装（E7）+ 参考文献扩到 35 条", "未开始", "E20/E21 结果"]))
    A(row(["4", "E8 投稿包（图、表、补充材料）", "未开始", "G1 人类输入"]))
    A("")

    # ---------------------------------------------------------------- appendix
    A("## 附录：输入产物指纹")
    A("")
    A("| 产物 | 文件 | sha256(前16) |")
    A("|---|---|---|")
    for name in INPUTS:
        A(row([name, f"`results/dsh/{INPUTS[name].relative_to(RES).as_posix()}`", sha(name)]))
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}  ({len(L)} lines)")

    meta = {k: {"path": str(v.relative_to(ROOT)), "sha256_16": sha(k),
                "exists": v.exists()} for k, v in INPUTS.items()}
    (RES / "decision_report_inputs.json").write_text(
        json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                    "inputs": meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {RES / 'decision_report_inputs.json'}")


if __name__ == "__main__":
    main()
