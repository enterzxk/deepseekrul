"""Archive completed joint comparisons and write evidence-derived reports.

No training or neural prediction is performed. Official labels are opened
only after the global completion seal and all 94 output seals are checked.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import torch
from .common import ROOT, PROJECT, digest, state_hash, write_json
from . import joint
from .collect import NAMES, table_metrics
from .verify import MODELS, FDS, SEEDS, read, score, summarize_rows, compare_rows


FIELDS=["model","subset","seed","rmse","nasa_sum","n","params","trainable_params",
        "training_time_s","validation_engine_mean_rmse","validation_pooled_rmse",
        "n_train_windows","best_epoch","epochs_run","config_sha256","checkpoint_sha256","state_content_sha256","test_forward_passes"]


def write_csv(path, rows):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def inventory(root):
    write_json(root/"ARTIFACT_MANIFEST.json",[
        {"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p)}
        for p in sorted(root.rglob("*")) if p.is_file() and p!=root/"ARTIFACT_MANIFEST.json"])


def make_report(target, docdir, physical, logical, master):
    summary=read(target/"summary.json");paired=read(target/"paired_comparisons.json")
    main=[r for r in summary if r["condition"]=="author_limits"]
    fixed=[r for r in summary if r["condition"]=="fixed40"]
    same=["dspt_v0_cycle","dspt_train_defaults","aidgn","dual_mixer","ffve","fcstgnn"]
    policy=read(ROOT/"AUTHOR_LIMIT_POLICY.json")
    lines=["# Paper B：按 DSPT 协议重训的公开源码基线对比","",
           "本轮已完成五篇公开方法及两个预先固定的 DSPT 参考配置，在 FD001–FD004、种子 101/202/303 上真实训练和评测。",
           "共 **94 次物理训练、168 个逻辑条件结果、91 次实际官方 TEST 预测及 3 次有等价证明的预测复用**；所有训练先冻结，再打开本轮 TEST。",
           "主表采用作者训练上限条件，固定 40 轮仅作预算敏感性分析。旧作者协议、旧统一协议、历史 OOF、Validation 和本轮官方 TEST 不混表。","",
           "## 1. 与当前 DSPT 对齐的共同规则","",
           "- 按发动机 ID 划分 80% Train / 20% Validation；split_seed=7、fold=0；三个训练种子为 101/202/303。",
           "- 所有模型训练样本从 cycle 1 开始，早期窗口左零填充；验证采用每机生命周期 40/55/70/85/95% 五截断点。",
           "- RUL cap=125，输出单位 cycles；先算每台验证机 RMSE，再对发动机等权平均，以此选 checkpoint；patience=12、min_delta=1e-4。",
           "- 统计量仅 fit 训练机；共同工况 KMeans(k=6)+sensor z-score。FFVE 保留原因果平滑，Two-Phase 保留 operating 输入。",
           "- 官方 TEST 每机末周期一次预测，覆盖 FD001/002/003/004 的 100/259/100/248 台发动机，不丢短机。",
           "- NASA Sum：d=预测−真实；d<0 用 exp(−d/13)−1，否则用 exp(d/10)−1，再逐机求和。",
           "- 全部共同 FP32，禁 autocast 和 TF32。这是新的统一精度条件，不与历史 DSPT 默认 BF16 结果声称数值等价。","",
           "主参考始终固定为 DSPT-V0-Cycle（full/basic/hf/cycle=True）；辅助参考固定为 literal train DEFAULTS（full/off/none/cycle=True）。",
           "两者都列出，没有根据 TEST 挑选参考；同时改变 SA 和 HI，因此不能当作单一 HI 消融。DSPT 原模型及 loss 字节未改，运行时禁用原 autocast。",
           "参考来自桌面 DSPT 目录的 9 文件固定快照，捕获时 HEAD 为 7d8fa3ecc4a47bfd089c4822b24942ce8466b436；不会导入随后变化的 live 目录。","",
           "五个基线保留原核心架构、loss、优化器、LR、batch；不是简化替代网络。窗口只取作者默认单点：AIDGN 50、Two-Phase 45、Dual-Mixer 30、FFVE 30、FC-STGNN 30/50/50/50，两个 DSPT 均 70。",
           "窗口搜索预算为 0；45 是 Two-Phase 原特征注意力的硬编码长度。各方法的优化器、窗口与预处理特性有明确差异，故本实验统一评估规则，不声称相同算力或穷尽调参后的最优性能。","",
           "## 2. 训练预算与公平性修订","",
           "最初固定 40 轮；训练/验证显示 Two-Phase FD002 仍在明显改善。首次本轮 TEST 前，独立 Codex 决策 PB-OS-009 将作者上限条件设为主比较，保留 40 轮条件。",
           "该修订发生在验证之后，不能称为从最初就完全预注册；没有依据本轮 TEST 做修改。共同资格规则：原 run 完成 40 轮、40−best_epoch<12、作者上限>40。",
           "符合者共 10 组，全部从相同种子完整重训，包含 Two-Phase 预训练；没有从仅含模型权重的 checkpoint 续训。",
           "其余 74 个主条件引用实际早停或固定 DSPT 的原配置/权重，不伪造高上限配置。Adam/StepLR/Plateau 对未到达的最大 epoch 独立；DSPT Cosine Tmax 仍为 40。",
           "全部 10 个扩展 run 的前 40 轮共 400 对权重内容 hash 和验证数值完全一致。作者上限只是停止上界，不等于数学收敛保证。","",
           "| 方法 | FD001/FD002/FD003/FD004 监督 epoch 上限 |","|---|---|"]
    for model in MODELS:
        lines.append("| "+NAMES[model]+" | "+" / ".join(map(str,policy["ceilings"][model]))+" |")
    lines += ["","## 3. 主比较：单子集训练资源，官方 TEST RMSE（cycles）","",
              "各格为三个种子的均值 ± 样本 SD，越低越好；没有挑最好 seed。",""]
    lines += table_metrics(main,same,"rmse")
    lines += ["","## 4. 主比较：单子集训练资源，官方 TEST NASA Sum",""]
    lines += table_metrics(main,same,"nasa_sum")
    lines += ["","## 5. 额外跨子集预训练：Two-Phase 单列","",
              "完整方法在 FD001+FD003 或 FD002+FD004 上预训练 20/100 轮，双方验证机均排除，保留 GRL、分类/回归及严格 encoder 迁移/冻结。",
              "该数据资源不同于单子集方法，不能隐藏到相同资源表。保留冻结的 FT 先初始化、PRE 不重新设 seed 的全流水线随机流，非作者两阶段分别重置 seed 的逐轨迹复现。",
              "预训练权重、encoder 前后 hash、样本数和时间逐 run 保存，训练总时间包含预训练。","","RMSE：",""]
    lines += table_metrics(main,["two_phase"],"rmse")
    lines += ["","NASA Sum：",""]+table_metrics(main,["two_phase"],"nasa_sum")
    lines += ["","## 6. 固定 40 轮敏感性分析：官方 TEST","",
              "本表是另一个预算条件，不能与主表拼接取最优值；没有把同一实际训练的复用结果算成新训练。","","RMSE：",""]
    lines += table_metrics(fixed,same+["two_phase"],"rmse")
    lines += ["","NASA Sum：",""]+table_metrics(fixed,same+["two_phase"],"nasa_sum")
    lines += ["","## 7. 主比较的配对差异","",
              "下表为基线减预先固定的 DSPT-V0-Cycle；负值表示基线更低。95% CI 来自三个同 seed 差值的 Student-t 区间（df=2）。",
              "这是固定数据划分下训练随机性的描述，n=3 不支持强统计优越性结论。全部对辅助参考的 RMSE/NASA 配对结果也保存在 paired_comparisons.json。","",
              "| 基线 | 子集 | RMSE 差 | 95% CI | NASA 差 | 95% CI | RMSE 更低 seed |",
              "|---|---|---:|---|---:|---|---:|"]
    for row in paired:
        if row["condition"]!="author_limits" or row["reference"]!="dspt_v0_cycle":continue
        a,b=row["rmse_difference_ci95"];c,d=row["nasa_sum_difference_ci95"]
        name=NAMES[row["model"]]+("（额外数据）" if row["model"]=="two_phase" else "")
        lines.append(f'| {name} | {row["subset"]} | {row["rmse_difference_mean"]:+.2f} | [{a:+.2f}, {b:+.2f}] | {row["nasa_sum_difference_mean"]:+.2f} | [{c:+.2f}, {d:+.2f}] | {row["rmse_baseline_better_seeds"]}/3 |')
    lines += ["","## 8. 预算效应、成本与判定","",
              "Two-Phase 延长预算后的 FD002/FD004 RMSE 均值明显低于固定 40 轮；据此不能用固定 40 轮差距代表该方法能力。",
              "在本轮主条件下，DSPT-V0-Cycle 的四子集 RMSE 均值均低于五个已跑基线。NASA 不能概括为全胜：FD004 的 FC-STGNN 均值低于主参考，辅助 DSPT 的排序也不同。",
              "没有把辅助参考的更优单元替换进主参考行；完整数据、CI 和各 seed 均保留。","",
              "| 配置 | 参数量 FD001/002/003/004 | 主条件关联训练时间（process-h） | 本轮实际训练总计（process-h） |",
              "|---|---|---:|---:|"]
    for model in MODELS:
        chosen=[r for r in logical if r["condition"]=="author_limits" and r["model"]==model]
        actual=[r for r in physical if r["model"]==model]
        params=" / ".join(str(next(r for r in chosen if r["subset"]==fd)["params"]) for fd in FDS)
        lines.append(f'| {NAMES[model]} | {params} | {sum(r["training_time_s"] for r in chosen)/3600:.3f} | {sum(r["training_time_s"] for r in actual)/3600:.3f} |')
    lines += ["",f'实际 94 次训练的 process-wall-time 之和为 {sum(r["training_time_s"] for r in physical)/3600:.3f} 小时，包含预处理、验证与预训练，不是独占 GPU 卡时。',
              "共享 A40 上其他既有任务未停止；原 CPU 调度器被共享资源控制暂停的事实及采用新内存队列的恢复链均保留，没有重跑已完成单元。",
              "主条件实际达到 epoch 上限、尚未触发 patience 的单元如下；这表示预算截断，不声称已收敛。","",
              "| 配置 | 子集 | seed | 实际 epoch / 配置上限 | 最佳 epoch |","|---|---|---:|---|---:|"]
    for r in logical:
        if r["condition"]=="author_limits" and r["budget_truncated"]:
            lines.append(f'| {NAMES[r["model"]]} | {r["subset"]} | {r["seed"]} | {r["epochs_run"]} / {r["configured_epoch_ceiling"]} | {r["best_epoch"]} |')
    lines += ["","| 方法 | 本轮共同协议执行判定 | 原作者数值/来源复现判定（旧批次，未改） |","|---|---|---|"]
    original={"aidgn":"PARTIAL","two_phase":"PARTIAL","dual_mixer":"PARTIAL","ffve":"SUCCESS","fcstgnn":"SUCCESS"}
    for model in MODELS:
        lines.append(f'| {NAMES[model]} | SUCCESS：12 个主条件结果及 12 个 40 轮结果完整、断言通过 | {original.get(model,"不适用：固定 DSPT 参考")} |')
    lines += ["","SUCCESS 仅指本轮注册共同协议的真实执行与证据完整，不表示已经复现原论文所有数值或达到调参最优。第六候选 CPT 原接口 FAILED 的历史判定不变，没有用其他网络补位。","",
              "## 9. 证据、独立审计与复算","",
              "训练源分别冻结 462/460 文件，checkpoint/config/data/source 全部 hash 绑定。94 个 checkpoint 均经 CPU 有限性、canonical state hash 与选模重放检查；94 个训练 access audit 均为 0 官方 TEST 读取。",
              "READY 之前完成所有训练及两条件冻结。独立 Codex 只读 CPU 审计核对了全部 94 输出、永久一次额度、完整 engine ID、原始官方 RUL cap125、实际数据 SHA 和 RMSE/NASA 公式，最终 PASS、0 项失败。",
              "独立重算最大差异：RMSE 8.39e-9、NASA 1.42e-6；所有本轮官方文件访问均晚于 READY，最早晚 262.36 秒。",
              "另一独立 Codex 只读审计逐项重算了 56 个汇总组、80 个配对组、两条件映射、报告表格及成本，全部 PASS，无新增 P1/P2。两份审计记录见 INDEPENDENT_MATCHED_AUDIT_20261112.json。",
              "3 次预测复用为 Dual-Mixer FD001/s101、FC-STGNN FD003/s101、AIDGN FD004/s303：权重相同、原预测封印有效、来源/候选/实际 forward 输入三向 hash 相等，共享永久 TEST_ATTEMPT，明确 forward=0；不是 3 次新预测。",
              "实际生产 joint.test 在运行前强制完整 READY。独立审计发现保留的旧单 run CLI 可绕过联合入口检查；没有实际提前开闸证据。封印代码保持原样，新增 guarded_test.py 作为支持的单 run 入口，强制联合 READY 和物理身份检查。它是事后入口加固，不冒充本轮已执行版本。",
              "原始 TEST 曾在历史项目/作者复现中被观测，因此本轮不称全新盲测。所有本轮模型与配置在新 TEST 前固定；测试后仅做取证、汇总和入口加固。",
              "四子集及三 seed 不作为更多独立数据集；没有测试集调参、集成、挑选 seed 或替换冻结 DSPT。窗口搜索为单点，保留完整模型不等于证明其最强可达性能。","",
              "- [逐物理训练记录](../../paper_b/experiments/open_source_matched_20261112/physical_runs.csv)",
              "- [两条件逐 seed 对照](../../paper_b/experiments/open_source_matched_20261112/logical_runs.csv)",
              "- [均值、SD、95% CI](../../paper_b/experiments/open_source_matched_20261112/summary.json)",
              "- [两参考配对差异](../../paper_b/experiments/open_source_matched_20261112/paired_comparisons.json)",
              "- [工件 SHA256 清单](../../paper_b/experiments/open_source_matched_20261112/ARTIFACT_MANIFEST.json)",
              "- [源码、协议与运行说明](../../paper_b/baselines/open_source_matched_20261112/README.md)",
              "- [旧作者协议及数值复现报告](OPEN_SOURCE_REPRODUCTION_REPORT.md)",""]
    docdir=Path(docdir);docdir.mkdir(parents=True,exist_ok=True)
    (docdir/"MATCHED_COMPARISON_REPORT_20261112.md").write_text("\n".join(lines),encoding="utf-8")


def collect(original, extended, data_dir, target, docdir):
    original,extended,target=Path(original).resolve(),Path(extended).resolve(),Path(target).resolve()
    master=joint.validate_ready(original,extended)
    master_sha=digest(original/"JOINT_COMPARISON_FREEZE.json")
    complete=read(original/"JOINT_TEST_COMPLETE.json")
    assert complete["joint_freeze_sha256"]==master_sha and complete["physical_runs"]==94
    # All outputs must be closed and hash-bound before any official-label read.
    for record in master["physical_runs"]:joint.result_identity(record,master_sha)
    assert complete["unique_TEST_forward_passes"]==91
    target.mkdir(parents=True,exist_ok=True)
    (target/".gitattributes").write_bytes(b"* -text -whitespace\n")
    truth=[]
    for fd in FDS:
        p=Path(data_dir)/f"RUL_{fd}.txt"
        assert digest(p)==read(ROOT/"DATASET_EXPECTED_SHA256.json")[p.name]
        truth += [{"subset":fd,"unit":i,"y_true":min(125,float(v))} for i,v in enumerate(p.read_text().split(),1)]
    write_csv(target/"official_truth.csv",truth)
    truths={(r["subset"],r["unit"]):r["y_true"] for r in truth}
    physical=[]
    for record in master["physical_runs"]:
        source=Path(record["path"]);m=read(source/"metrics.json");cfg=read(source/"config.json")
        state=torch.load(source/"checkpoint.pt",map_location="cpu",weights_only=True)
        assert all(torch.isfinite(t).all() for t in state.values())
        assert state_hash(state)==m["state_content_sha256"] and digest(source/"checkpoint.pt")==m["checkpoint_sha256"]
        with (source/"test_predictions.csv").open() as f:pred=list(csv.DictReader(f))
        for r in pred:assert float(r["y_true"])==truths[(m["subset"],int(r["unit"]))]
        r,n=score(pred)
        assert abs(r-m["rmse"])<1e-5 and math.isclose(n,m["nasa_sum"],rel_tol=2e-7,abs_tol=.01)
        dest=target/"physical_runs"/record["id"];dest.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if path.is_file():shutil.copy2(path,dest/path.name)
        write_json(dest/"cpu_weight_audit.json",{"finite":True,"canonical_state_sha256":state_hash(state),"checkpoint_sha256":digest(source/"checkpoint.pt"),
                   "tensor_shapes":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in state.items()}})
        for phase in ["train","test"]:
            log=Path(record["root"])/"logs"/f'{phase}_{m["model"]}_{m["subset"]}_{m["seed"]}.log'
            if phase=="train" or m["test_forward_passes"]:assert log.exists(),log
            if log.exists():(dest/(phase+"ing_stdout.log.gz" if phase=="train" else "test_stdout.log.gz")).write_bytes(gzip.compress(log.read_bytes(),mtime=0))
        physical.append({"physical_id":record["id"],"source_condition":record["condition"],**{k:m[k] for k in FIELDS},
                         "configured_epoch_ceiling":cfg["epochs"],"budget_truncated":m["epochs_run"]==cfg["epochs"] and m["epochs_run"]-m["best_epoch"]<12,
                         "path":dest.relative_to(target).as_posix()})
    write_csv(target/"physical_runs.csv",physical)
    by={r["physical_id"]:r for r in physical};policy=read(ROOT/"AUTHOR_LIMIT_POLICY.json")
    logical=[{"condition":r["condition"],**by[r["physical_id"]],"registered_author_ceiling":policy["ceilings"][r["model"]][FDS.index(r["subset"])]} for r in master["logical_conditions"]]
    write_csv(target/"logical_runs.csv",logical)
    summaries=[];paired=[]
    for condition in ["author_limits","fixed40"]:
        rows=[r for r in logical if r["condition"]==condition]
        summaries += [{"condition":condition,**r} for r in summarize_rows(rows)]
        paired += [{"condition":condition,**r} for r in compare_rows(rows)]
    write_json(target/"summary.json",summaries);write_json(target/"paired_comparisons.json",paired)
    original_status={"aidgn":"PARTIAL","two_phase":"PARTIAL","dual_mixer":"PARTIAL","ffve":"SUCCESS","fcstgnn":"SUCCESS"}
    write_json(target/"STATUS.json",{m:{"matched_execution_status":"SUCCESS","scope":"registered common-protocol execution and evidence; not original-number replication or tuned optimum",
              "primary_logical_runs":12,"fixed40_logical_runs":12,"actual_training_runs":sum(r["model"]==m for r in physical),"historical_author_reproduction_status":original_status.get(m,"NOT_APPLICABLE")} for m in MODELS})
    evidence=target/"evidence"
    for condition,folder in [("fixed40",original),("author_limits",extended)]:
        d=evidence/condition;d.mkdir(parents=True,exist_ok=True)
        for p in folder.glob("*.json"):shutil.copy2(p,d/p.name)
    src=evidence/"source";src.mkdir(exist_ok=True)
    for p in ROOT.glob("*.json"):shutil.copy2(p,src/p.name)
    for namespace,package in [("fixed40",ROOT),("author_limit",joint.author.ROOT)]:
        d=src/namespace;d.mkdir(exist_ok=True)
        for p in package.glob("*.json"):shutil.copy2(p,d/p.name)
    # Archive our executable code and reference snapshot; upstream source
    # without redistribution rights is restored by the pinned fetcher instead.
    for item in read(ROOT/"EXECUTION_SOURCE_MANIFEST.json")["files"]+read(joint.author.ROOT/"EXECUTION_SOURCE_MANIFEST.json")["files"]:
        rel=item["path"]
        allowed=rel.startswith(("paper_b/baselines/open_source_matched_20261112/","paper_b/baselines/open_source_author_limits_20261112/")) or rel in {"paper_b/baselines/open_source/adapters.py","paper_b/baselines/open_source/common.py","paper_b/baselines/open_source/sources.lock.json"} or rel.endswith("/SOURCE_MANIFEST.json")
        if allowed:
            p=PROJECT/rel;d=evidence/"execution_code"/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for package in [ROOT,joint.author.ROOT]:
        for p in package.glob("*"):
            if p.is_file() and p.suffix in {".py",".json",".md"}:
                d=evidence/"execution_code"/p.relative_to(PROJECT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    for p in original.parent.glob("*.log"):
        if p.name.startswith("collect_joint"):continue
        (evidence/(p.name+".gz")).write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    for name in ["README.md","acquire.py"]:
        p=PROJECT/"paper_b/baselines/open_source"/name
        if p.exists():
            d=evidence/"upstream_restore"/name;d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
    for name in ["SOURCE_AUDIT.json","SEED_POLICY_AUDIT.json","INDEPENDENT_MATCHED_AUDIT_20261112.json"]:
        p=PROJECT/"docs/paper_b_baselines"/name
        if p.exists():shutil.copy2(p,evidence/name)
    inventory(target)
    from .verify_joint import verify
    verify(target)
    make_report(target,docdir,physical,logical,master)
    write_json(Path(docdir)/"MATCHED_STATUS_20261112.json",read(target/"STATUS.json"))
    print("JOINT_COLLECTED",json.dumps({"actual_training":94,"logical_cases":168,"actual_TEST":91,"proven_prediction_reuse":3}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["root","author-root","data-dir","target","docdir"]:p.add_argument("--"+name,required=True)
    a=p.parse_args();collect(a.root,a.author_root,a.data_dir,a.target,a.docdir)
