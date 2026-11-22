"""Generate separate published, author and unified tables from audited artifacts."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from .common import digest, write_json

NAMES = {"aidgn": "AIDGN-RUL", "two_phase": "Two-Phase Attention", "dual_mixer": "Dual-Mixer+FSGRI",
         "ffve": "FFVE", "fcstgnn": "FC-STGNN"}
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def aggregate(table):
    records = []
    for keys, group in table.groupby(["protocol", "model", "subset"]):
        record = dict(zip(["protocol", "model", "subset"], keys))
        record["seeds"] = sorted(map(int, group.seed))
        record["n_seeds"] = len(group)
        for metric in ["rmse", "nasa_sum", "training_time_s", "normalized_author_rmse", "author_reported_rmse_separate_stochastic_draw", "validation_rmse"]:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            record[metric + "_mean"] = float(values.mean())
            if len(values) > 1:
                sd = float(values.std(ddof=1))
                half = float(t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
                record[metric + "_sd"] = sd
                record[metric + "_ci95"] = [float(values.mean() - half), float(values.mean() + half)]
        record["params"] = sorted(set(map(int, group.params)))
        record["trainable_params"] = sorted(set(map(int, group.trainable_params)))
        record["evaluation_units"] = sorted(set(group.evaluation_unit))
        records.append(record)
    return records


def format_metric(row, metric, digits=2):
    if row is None or metric + "_mean" not in row:
        return "—"
    mean = row[metric + "_mean"]
    if row["n_seeds"] != 3:
        return f"{mean:.{digits}f} (n={row['n_seeds']}/3)"
    return f"{mean:.{digits}f} ± {row.get(metric + '_sd', 0):.{digits}f}"


def generate(artifacts, source_audit, report_dir):
    artifacts, report_dir = Path(artifacts), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(artifacts / "runs.csv")
    sources = json.loads(Path(source_audit).read_text())["models"]
    summaries = aggregate(table)
    write_json(artifacts / "summary.json", summaries)
    lookup = {(x["protocol"], x["model"], x["subset"]): x for x in summaries}
    provenance = json.loads((artifacts / "evidence/EXECUTION_PROVENANCE.json").read_text())
    for protocol in ["author", "unified"]:
        table[table.protocol == protocol].to_csv(report_dir / f"{protocol}_per_seed.csv", index=False)
    states = {}
    matrix_finished = (artifacts / "evidence/matrix_complete.json").exists()
    for model in NAMES:
        author = table[(table.protocol == "author") & (table.model == model)]
        unified = table[(table.protocol == "unified") & (table.model == model)]
        reasons = []
        if len(author) != 12 or len(unified) != 12:
            reasons.append(f"coverage author={len(author)}/12, unified={len(unified)}/12")
        published = sources[model].get("reported_rmse")
        gaps = []
        if published is None:
            reasons.append("published cycles-RMSE/NASA table unverified or not applicable")
        else:
            for subset, reference in zip(SUBSETS, published):
                row = lookup.get(("author", model, subset))
                if row is not None and row["n_seeds"] == 3:
                    # FFVE has separate upstream RMSE and Score draws. The
                    # author-matching gate uses the actually reported RMSE.
                    value = row.get("author_reported_rmse_separate_stochastic_draw_mean", row["rmse_mean"])
                    gaps.append((value - reference) / reference)
            if len(gaps) != 4 or any(abs(gap) > 0.10 for gap in gaps):
                reasons.append("published RMSE ±10% gate not met on every subset")
        if model == "dual_mixer":
            reasons.append("public-code versus preprint hyperparameters differ; paper result is normalized/all-window")
        missing_source = [r for r in provenance["author_runs"] if r["model"] == model and r["status"] != "EXACT_SOURCE_RETAINED"]
        if missing_source:
            reasons.append(f"exact historical launcher source missing/unrecorded for {len(missing_source)} author runs; upstream/core/runtime patches retained")
        state = "SUCCESS" if not reasons else ("PARTIAL" if len(author) or len(unified) or not matrix_finished else "FAILED")
        states[model] = {"status": state, "author_runs": len(author), "unified_runs": len(unified),
                         "reasons": reasons, "author_rmse_relative_gaps": gaps}
    candidate = json.loads((artifacts / "evidence/candidate_execution.json").read_text())
    states["cpt"] = {"status": candidate["status"], "scope": "released pretraining interface probe only",
                     "reason": candidate["reason"], "training_completed": False}
    write_json(report_dir / "STATUS.json", states)
    lines = ["# Paper B：公开源码强基线复现报告", "",
             "数据与代码审计日期：2026-11-11。执行人为 Codex；三个独立 `gpt-6.1-sol/high` 子代理只读审计。",
             "冻结 DSPT、Paper A、旧实验均未修改。以下数字全部由保存的真实预测与指标文件生成；缺测不补填。", "",
             "> **2026-11-12 协议更正：** 本报告的旧 `unified` 批次共享DSPT划分/标签/测试口径，但不严格匹配当前目录的发动机平均RMSE选模及全部周期训练窗口规则。SUCCESS判定不代表通过当前协议对照。见[逐项核查与结果](PROTOCOL_COMPARISON_20261112.md)。", "",
             "## 1. 复现判定与证据覆盖", "",
             "SUCCESS/PARTIAL/FAILED 的门见 [PROTOCOL.md](PROTOCOL.md)。SUCCESS 要求两协议各12次完整运行、",
             "审计通过，以及每个子集的作者公开 RMSE 相对差在±10%以内。PARTIAL 不等于模型被简化：",
             "也用于真实复现完成但公开数值不可核实、论文/代码协议有差异或数值门未达到。", "",
             "| 方法 | 判定 | 作者协议 | 统一协议 | 未满足项 |", "|---|---|---:|---:|---|"]
    for model, name in NAMES.items():
        state = states[model]
        reason = "; ".join(state["reasons"]) or "全部预注册门通过"
        lines.append(f"| {name} | **{state['status']}** | {state['author_runs']}/12 | {state['unified_runs']}/12 | {reason} |")
    lines.extend([f"| CPT（第六候选） | **{candidate['status']}** | 原始预训练接口检查 | 未纳入优先训练矩阵 | forward输出[B,1]被再次传入期望36维的classifier；没有用普通回归模型替代 |", "",
                  "本表判定不代表模型相对 DSPT 获胜。数值差异与协议风险完整保留。", "",
                  "## 2. 论文身份、官方来源与固定源码", "",
                  "逐库实现/数据/训练/测试/checkpoint 审计见 [SOURCE_AUDIT.json](SOURCE_AUDIT.json)。",
                  "源锁与逐文件 Git blob/SHA256 见 [sources.lock.json](../../paper_b/baselines/open_source/sources.lock.json)。",
                  "原始源码按commit下载后逐字节保留，不在原文件上打补丁。因部分上游未授予再分发许可，",
                  "本仓库提交获取器、精确manifest、适配器与运行补丁，原始代码由获取器恢复。", "",
                  "| 方法 | DOI | 固定 commit | 可公开核查的结果来源 |", "|---|---|---|---|"])
    for model, name in {**NAMES, "cpt": "CPT"}.items():
        source = sources[model]
        result_link = f"[来源]({source['reported_source']})" if source.get("reported_source") else "数值表未取得"
        lines.append(f"| {name} | [{source['doi']}](https://doi.org/{source['doi']}) | [`{source['commit'][:12]}`]({source['repository']}/commit/{source['commit']}) | {result_link} |")
    lines.extend(["", "公开 checkpoint：AIDGN tree含checkpoint/result，但也有其他模型/数据集；Two-Phase发布两份paired预训练权重；",
                  "其余四库未见公开checkpoint。统一协议全部从头训练，包括排除双方验证机后的Two-Phase预训练。", "",
                  "## 3. 作者公开报告（引用值，非本次训练）", "",
                  "下表每格为 RMSE / NASA Sum，顺序FD001–004。AIDGN来自作者README，未取得正文结果表；",
                  "FFVE采用作者2024-01-22披露RMSE计算bug后的修正表；FC-STGNN为原文Table1的10次均值。", "",
                  "| 方法及来源口径 | FD001 | FD002 | FD003 | FD004 |", "|---|---:|---:|---:|---:|"])
    for model in ["aidgn", "ffve", "fcstgnn"]:
        source = sources[model]
        values = [f"{r:.2f} / {s:.2f}" for r, s in zip(source["reported_rmse"], source["reported_nasa"])]
        lines.append("| " + NAMES[model] + " | " + " | ".join(values) + " |")
    lines.extend(["| Two-Phase | 未核实 | 未核实 | 未核实 | 未核实 |", "",
                  "Dual-Mixer-F作者预印本Table4报告**归一化全窗口RMSE**（3次均值）：0.1002/0.1338/0.0887/0.1141，",
                  "并报告MAPE，未报告NASA。该表不能转换成官方逐机RMSE排名，也不能直接乘125。", "",
                  "## 4. 忠实作者代码协议：实际训练", "",
                  "AIDGN、Two-Phase和FC-STGNN为逐机cycles指标。FFVE本表的RMSE与NASA从同一次实际随机预测重算；",
                  "下一表另保留作者代码所报的另一随机draw RMSE。Dual-Mixer单独列全窗口结果。所有均值±样本SD；",
                  "完整逐seed与训练时间、参数量、config/checkpoint hash见 [author_per_seed.csv](author_per_seed.csv)。", "",
                  "| 方法 | 子集 | RMSE cycles | NASA Sum | seeds |", "|---|---|---:|---:|---:|"])
    for model in ["aidgn", "two_phase", "ffve", "fcstgnn"]:
        for subset in SUBSETS:
            row = lookup.get(("author", model, subset))
            lines.append(f"| {NAMES[model]} | {subset} | {format_metric(row,'rmse')} | {format_metric(row,'nasa_sum')} | {row['n_seeds'] if row else 0}/3 |")
    lines.extend(["", "作者公开值差异：每格为本次作者协议RMSE均值减公开值（cycles），括号内相对差。Two-Phase无已核实值不计算。", "",
                  "| 方法 | FD001 | FD002 | FD003 | FD004 |", "|---|---:|---:|---:|---:|"])
    for model in ["aidgn", "ffve", "fcstgnn"]:
        cells = []
        for subset, reference in zip(SUBSETS, sources[model]["reported_rmse"]):
            row = lookup.get(("author", model, subset))
            if row is None:
                cells.append("—")
            else:
                value = row.get("author_reported_rmse_separate_stochastic_draw_mean", row["rmse_mean"])
                cells.append(f"{value-reference:+.2f} ({100*(value/reference-1):+.1f}%)")
        lines.append("| " + NAMES[model] + " | " + " | ".join(cells) + " |")
    lines.extend(["", "NASA Sum差异（本次作者协议均值减作者公开值，括号内相对差）：", "",
                  "| 方法 | FD001 | FD002 | FD003 | FD004 |", "|---|---:|---:|---:|---:|"])
    gaps_export = []
    for model in ["aidgn", "ffve", "fcstgnn"]:
        cells = []
        for subset, reference in zip(SUBSETS, sources[model]["reported_nasa"]):
            row = lookup.get(("author", model, subset))
            if row is None:
                cells.append("—")
            else:
                value = row["nasa_sum_mean"]
                cells.append(f"{value-reference:+.2f} ({100*(value/reference-1):+.1f}%)")
                rr = sources[model]["reported_rmse"][SUBSETS.index(subset)]
                actual_r = row.get("author_reported_rmse_separate_stochastic_draw_mean", row["rmse_mean"])
                gaps_export.append({"model": model, "subset": subset, "seeds": row["n_seeds"],
                                    "published_rmse": rr, "actual_author_rmse": actual_r,
                                    "rmse_absolute_gap": actual_r-rr, "rmse_relative_gap": actual_r/rr-1,
                                    "published_nasa_sum": reference, "actual_author_nasa_sum": value,
                                    "nasa_absolute_gap": value-reference, "nasa_relative_gap": value/reference-1})
        lines.append("| " + NAMES[model] + " | " + " | ".join(cells) + " |")
    pd.DataFrame(gaps_export).to_csv(report_dir / "published_gaps.csv", index=False)
    lines.extend(["", "| FFVE原代码实际报告（独立随机draw） | FD001 | FD002 | FD003 | FD004 |", "|---|---:|---:|---:|---:|",
                  "| RMSE cycles | " + " | ".join(format_metric(lookup.get(("author","ffve",subset)),"author_reported_rmse_separate_stochastic_draw") for subset in SUBSETS) + " |", "",
                  "### Dual-Mixer+FSGRI：原始全窗口口径单独列示", "",
                  "完整FSGRI已显式开启。保留原始Gaussian sampler、正样本噪声和weighted InfoNCE+MSE。",
                  "代码默认lr/batch/threshold与预印本Table3不同，不能声称严格复现其论文超参。", "",
                  "| 子集 | 原归一化RMSE | 相对原文差异 | 逆变换后全窗口RMSE cycles | 全窗口NASA（非逐机） |", "|---|---:|---:|---:|---:|"])
    for subset, reference in zip(SUBSETS, sources["dual_mixer"]["reported_rmse_normalized"]):
        row = lookup.get(("author", "dual_mixer", subset))
        gap = f"{100*(row['normalized_author_rmse_mean']/reference-1):+.1f}%" if row and "normalized_author_rmse_mean" in row else "—"
        lines.append(f"| {subset} | {format_metric(row,'normalized_author_rmse',4)} | {gap} | {format_metric(row,'rmse')} | {format_metric(row,'nasa_sum')} |")
    lines.extend(["", "作者标签为 `(capped_RUL+1)/denominator`；真实四子集的denominator均126，故逆变换为`126*x-1`。",
                  "更正后每个窗口真值已与raw unit/cycle/RUL核对。作者丢弃短机，endpoint仅FD002 253/259、FD004237/248；",
                  "这些retained-engine endpoint文件只作附录，不进入完整官方TEST表。统一协议保留全部发动机。", "",
                  "## 5. DSPT匹配的统一协议：官方TEST", "",
                  "相同engine-ID划分（80/20，split_seed7/fold0）、seeds101/202/303、cap125、cycles输出、NASA Sum。",
                  "NASA定义：d=预测cycles−真实cycles；逐机d<0取exp(−d/13)−1，否则exp(d/10)−1，最后求和。",
                  "Validation为每个独立验证机生命周期40/55/70/85/95%五截断点。所有统计仅fit训练机。按validation选checkpoint，",
                  "保存config/source/checkpoint freeze后才允许读取official test；访问审计禁止训练期读test或RUL文件。",
                  "统一输入采用DSPT的训练机拟合KMeans(k=6)工况簇+sensor z-score；FD001/3的六个簇并非六种真实工况。短验证/测试历史左零填充，训练使用完整方法窗口。",
                  "Two-Phase另保留标准化OC输入，FFVE另保留因果指数平滑。这些预处理与部分作者管道不同；本表比较统一输入协议下的完整架构/目标，非原端到端预处理性能。",
                  "每次run只评一次官方TEST；以下结果不与历史OOF或验证窗口指标混用。", "",
                  "窗口范围为作者默认的单点：AIDGN50、Two-Phase45、Dual30、FFVE30、FC30/50/50/50；调窗预算0次。",
                  "Two-Phase45由特征注意力硬编码决定。每个子集允许各自原始学习率与架构参数；完整搜索预算已预注册。", "",
                  "单点窗口预算未证明每个方法的最佳调优性能；不得因已看到TEST差异而追加调参。", "",
                  "### 5.1 单子集训练方法", "",
                  "| 方法 | 子集 | 官方RMSE cycles | 官方NASA Sum | 参数量 |", "|---|---|---:|---:|---:|"])
    for model in ["aidgn", "dual_mixer", "ffve", "fcstgnn"]:
        for subset in SUBSETS:
            row = lookup.get(("unified", model, subset))
            params = "/".join(map(str, row["params"])) if row else "—"
            lines.append(f"| {NAMES[model]} | {subset} | {format_metric(row,'rmse')} | {format_metric(row,'nasa_sum')} | {params} |")
    lines.extend(["", "### 5.2 跨子集预训练资源条件：Two-Phase单列", "",
                  "完整方法需要FD001+3或FD002+4预训练，双方验证机均排除；不把额外训练数据隐藏进单子集表。",
                  "encoder完整迁移并冻结，GRL与分类/回归预训练均保留。", "",
                  "统一实现按整条流水线设seed，先初始化FT再初始化PRE，阶段切换不重置RNG；作者两个脚本分别重置同一seed。",
                  "独立初始化审计确认该差异，未读取统一TEST数字，并决定保留冻结实现、不据已见结果切换随机流。",
                  "因此不主张逐阶段初始化/训练轨迹与作者同seed完全一致；旁注绑定原config哈希，原config不改。见[SEED_POLICY_AUDIT.json](SEED_POLICY_AUDIT.json)。", "",
                  "| 子集 | 官方RMSE cycles | 官方NASA Sum | 参数量（总量 / 可训练） |", "|---|---:|---:|---:|"])
    for subset in SUBSETS:
        row = lookup.get(("unified", "two_phase", subset))
        params = ("/".join(map(str, row["params"])) + " / " + "/".join(map(str, row["trainable_params"]))) if row else "—"
        lines.append(f"| {subset} | {format_metric(row,'rmse')} | {format_metric(row,'nasa_sum')} | {params} |")
    lines.extend(["", "统一协议完整逐seed数据见 [unified_per_seed.csv](unified_per_seed.csv)。逐指标Student-t 95%CI、",
                  "训练时间、参数量见 [summary.json](../../paper_b/experiments/open_source_reproduction_20261111/summary.json)。",
                  "CI仅反映三次训练随机性，不增加独立发动机数量，也不消除固定开发集被重复使用的限制。", "",
                  "官方TEST在历史项目和本次作者协议中已经被观测，本次不是全新盲测。统一配置在新增统一测试前锁定，冻结后不依据TEST调参。", "",
                  "## 6. 核心一致性、协议差异与失败留痕", "",
                  "- AIDGN：原aging投影、动态图、图卷积保留；OCC=可见cycle/训练机平均寿命×1e-3。原代码滑窗验证改为engine验证。",
                  "- Two-Phase：原代码保留paired预训练和冻结encoder；作者协议fit train+test特征统计，统一协议仅fit训练机。",
                  "- Dual-Mixer：双路径gated exchange和完整FSGRI保留；统一标签为RUL/125，统一test对短机零padding、不丢engine。",
                  "- FFVE：按作者四子集脚本保留1/2/1/2 blocks和FD004长残差；保留随机重参数与KL+RMSE；统一validation/test用固定种子的单次实际随机draw，非取mu、非ensemble。",
                  "- FC-STGNN：两图分支与每FD patch/CNN参数完整保留；作者test-selected epoch只出现在作者表，统一表只按validation选模。",
                  "- 原始作者码始终独立保存；可选依赖导入、子集参数化、随机种子、纯可视化callback及test行序调整均有runtime patch。",
                  "- FFVE首次启动缺imageio：校验wheel后仅安装到隔离目录；FC-STGNN首次启动因IterableDataset抽象方法失败：仅为原数组容器加不可迭代stub。两者v3失败证据保留，v4完整重跑；独立审计确认网络和原数组处理未改。",
                  "- 早期Two-Phase行序和Dual×125导出记录保留；最终使用RECONCILED_metrics及已校验预测，旧导出不得引用为cycles/engine证据。",
                  "- 第六候选原预训练接口在真实训练输入上报shape错误，证据保存为candidate_execution.json；没有宣称完成训练。", "",
                  "## 7. 可复算与完整性", "",
                  f"本次共收集并独立复算 **{len(table)}** 次完整训练。所有数字源于逐run checkpoint/预测/配置；",
                  "RMSE与NASA由独立math.fsum/expm1实现复算，checkpoint/config hash核对，权重有限性检查，统一engine数量与官方标签核对。",
                  "20组真实train-only前向/反向检查及最终FFVE配置复检均有原始日志。原始失败、初始错误导出与修正链不删除。", "",
                  f"历史执行器证据：作者协议{len(provenance['author_runs'])}次中，{provenance['exact_source_retained']}次保留了与config记录哈希完全一致的launcher源码；",
                  f"{provenance['exact_source_missing']}次有历史launcher哈希但未找回完全匹配文本，{provenance['runner_hash_not_recorded']}次早期config未记录该哈希。",
                  "已从Git与初始部署的纯文本差异中按精确SHA恢复两版自身脚本；未执行候选文本或改写原config。缺口逐run列于",
                  "[EXECUTION_PROVENANCE.json](../../paper_b/experiments/open_source_reproduction_20261111/evidence/EXECUTION_PROVENANCE.json)。",
                  "固定上游代码、实际runtime patch、checkpoint及预测均保留，但不将这20次声称为完整逐字节launcher追溯；相关方法判为PARTIAL。",
                  "统一协议60次的四个冻结核心文件均有config哈希并经独立核对。", "",
                  "主要产物：", "",
                  "- [逐run记录](../../paper_b/experiments/open_source_reproduction_20261111/runs.csv)",
                  "- [指标与holdout审计](../../paper_b/experiments/open_source_reproduction_20261111/metric_audit.json)",
                  "- [工件SHA256目录](../../paper_b/experiments/open_source_reproduction_20261111/ARTIFACT_MANIFEST.json)",
                  "- [复现命令与目录说明](../../paper_b/baselines/open_source/README.md)",
                  "", "作者elapsed为包含预处理/最终测试的整段wall-time；统一training_time至冻结checkpoint（包含预训练/预处理、不含TEST）。",
                  "并发run的wall-time之和是process-hours，不冒充独占GPU卡时。",
                  "没有从官方test结果挑seed、删子集、改窗口或修改冻结DSPT。"])
    (report_dir / "OPEN_SOURCE_REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    inventory = [{"path": str(file.relative_to(artifacts)).replace("\\", "/"), "bytes": file.stat().st_size, "sha256": digest(file)}
                 for file in sorted(artifacts.rglob("*")) if file.is_file() and file.name != "ARTIFACT_MANIFEST.json"]
    write_json(artifacts / "ARTIFACT_MANIFEST.json", inventory)
    print("REPORT_GENERATED", len(table), "runs", {k: v["status"] for k, v in states.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--source-audit", required=True)
    ap.add_argument("--report-dir", required=True)
    a = ap.parse_args()
    generate(a.artifacts, a.source_audit, a.report_dir)
