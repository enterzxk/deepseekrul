"""Collect only genuine matched runs; no new model inference is performed."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import torch
from .common import ROOT, PROTOCOL, digest, state_hash, write_json
from .verify import MODELS, FDS, SEEDS, read, score, replay, summarize_rows, compare_rows

NAMES={"aidgn":"AIDGN-RUL","two_phase":"Two-Phase Attention","dual_mixer":"Dual-Mixer+FSGRI",
       "ffve":"FFVE","fcstgnn":"FC-STGNN","dspt_v0_cycle":"DSPT-V0-Cycle-FP32（主参考）",
       "dspt_train_defaults":"DSPT-TrainDefaults-FP32（辅助参考）"}

def manifest(root):
    records=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":digest(p)}
             for p in sorted(root.rglob("*")) if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"]
    write_json(root/"ARTIFACT_MANIFEST.json",records)

def table_metrics(summary,models,metric):
    by={(r["model"],r["subset"]):r for r in summary}
    lines=["| 方法 | FD001 | FD002 | FD003 | FD004 |","|---|---:|---:|---:|---:|"]
    for model in models:
        cells=[f'{by[(model,fd)][metric+"_mean"]:.2f} ± {by[(model,fd)][metric+"_sd"]:.2f}' for fd in FDS]
        lines.append("| "+NAMES[model]+" | "+" | ".join(cells)+" |")
    return lines

def report(target,docdir,rows):
    summary=summarize_rows(rows);comparisons=compare_rows(rows)
    write_json(target/"summary.json",summary);write_json(target/"paired_comparisons.json",comparisons)
    docdir.mkdir(parents=True,exist_ok=True)
    lines=["# Paper B：按当前DSPT共同数据与选模规则重跑的对照实验","",
           "注册：2026-11-12。全部84次监督训练完成并独立冻结后，才进行本批次官方TEST。",
           "七个预设配置×四子集×三种子101/202/303；未改动原DSPT源码、PaperA或旧实验。",
           "这些是新的FP32条件结果，不与旧BF16、池化选模批次或历史OOF数字合并。","",
           "## 1. 统一规则与执行证据","",
           "80/20发动机ID划分，split_seed7/fold0；全周期训练、早期左零补齐；cap125；Validation每机40/55/70/85/95%五截断点。",
           "共同最多40监督epoch、patience12、min_delta1e-4；先逐机算验证RMSE，再等权平均，依该值选checkpoint和早停。",
           "同时保存池化验证RMSE作为次指标。TEST为每台官方发动机末周期的一次预测，报告标准跨机RMSE和NASA Sum。",
           "NASA：d=预测−真实；d<0用exp(−d/13)−1，否则exp(d/10)−1，最后逐机求和。","",
           "全部参数/计算输入FP32，禁autocast与TF32；原DSPT训练/预测默认BF16通过有记录的runtime context禁用，模型与原loss字节未改。",
           "DSPT主参考固定为V0-Cycle(full/basic/hf/cycle=True)；literalDEFAULTS(full/off/none/cycle=True)始终独立报告，不根据TEST二选一。",
           "两参考同时改变SA/HI，不能解释为单一HI消融。参考源码9文件快照及462执行文件的身份由manifest固定，来自桌面目录捕获版本，非不断变化的live导入。","",
           "五个公开模型保留完整原架构/核心loss/优化器/LR/batch及原窗口；窗口AIDGN50、Two-Phase45、Dual30、FFVE30、FC30/50/50/50，DSPT70。",
           "窗口搜索预算0。相同监督epoch不等于相同梯度步数或算力，也不代表每个方法充分调优后的最优值。",
           "公共train-only工况归一化相同；FFVE保留因果平滑，Two-Phase保留3个operating输入。没有原生padding mask的方法不私加mask模块。","",
           "## 2. 相同单子集训练资源：官方TEST RMSE（cycles）","",
           "所有表为三个种子的均值±样本SD，越低越好。没有选最好seed。以下不与验证数字混表。",""]
    main=["dspt_v0_cycle","dspt_train_defaults","aidgn","dual_mixer","ffve","fcstgnn"]
    lines+=table_metrics(summary,main,"rmse")
    lines += ["","## 3. 相同单子集训练资源：官方TEST NASA Sum",""]+table_metrics(summary,main,"nasa_sum")
    lines += ["","## 4. 跨子集预训练资源条件：Two-Phase单列","",
              "完整方法含paired FD001+FD003或FD002+FD004预训练20/100epoch；双方验证机均排除，encoder完整迁移并冻结。",
              "该额外数据/预训练不纳入“相同单子集资源”比较。保留此前冻结的整条流水线seed策略（FT先初始化、PRE不另重置seed），不声称与作者逐阶段同seed轨迹相同。",
              "预训练时间、样本数与encoder前后hash逐run保存于pretraining_metrics.json，总训练时间包含预训练。","",
              "RMSE：",""]+table_metrics(summary,["two_phase"],"rmse")
    lines += ["","NASA Sum：",""]+table_metrics(summary,["two_phase"],"nasa_sum")
    lines += ["","## 5. 相同种子的配对差异：基线减固定主参考","",
              "负值表示该基线RMSE更低；95%CI为三组同seed差值的Student-t区间（df2）。仅反映固定数据下训练随机性，n3不能支撑强优越性主张。",
              "Two-Phase行另标额外数据条件。对辅助参考的全部配对结果保留在paired_comparisons.json，不按TEST挑参考或方法。","",
              "| 基线 | 子集 | RMSE差均值 | 95%CI | 基线更低的seed数 |","|---|---|---:|---|---:|"]
    for row in comparisons:
        if row["reference"]!="dspt_v0_cycle":continue
        lo,hi=row["rmse_difference_ci95"];name=NAMES[row["model"]]+("（额外数据）" if row["model"]=="two_phase" else "")
        lines.append(f'| {name} | {row["subset"]} | {row["rmse_difference_mean"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {row["rmse_baseline_better_seeds"]}/3 |')
    lines += ["","## 6. 验证、成本、可比性与边界","",
              "Validation结果均标select-on-val；原始逐机SSE/双口径/状态机接受epoch及权重内容hash均保存。",
              "逐run n_train_windows、验证engine/cut集合与统一样本数检查通过；每个子集所有配置使用相同训练/验证发动机。",
              "训练期Guard无官方文件读取；所有84run冻结后批次解锁，永久TEST_ATTEMPT独占消费一次额度；失败不会释放重试额度。",
              "官方测试已有历史暴露，本次不是全新盲测。全部设置在新TEST前固定，没有根据本次TEST改窗口、优化器或seed。",
              "四子集不是四次独立重复，FD001/FD003验证ID相同；跨子集结论仅作分情境描述。",
              "所有预测与权重有限性、config/checkpoint/source/data hashes及逐机标签由CPU/独立公式核查，无简化替代模型。",
              "复现可比性SUCCESS要求12个预设run完整且协议/证据断言通过，不要求统一条件下重现作者原协议RMSE；作者原协议复现状态仍见旧报告。","",
              "| 配置 | 参数量（各子集） | 训练总wall-time/秒 |","|---|---|---:|"]
    for model in MODELS:
        selected=[r for r in rows if r["model"]==model]
        params=" / ".join(str(next(r for r in selected if r["subset"]==fd)["params"]) for fd in FDS)
        lines.append(f'| {NAMES[model]} | {params} | {sum(r["training_time_s"] for r in selected):.1f} |')
    lines += ["","各run的wall-time含预处理/验证/预训练与取证；并发进程wall-time之和是process-hours，不等于独占GPU卡时。",
              "机器上其他既有任务未停止，训练速度只能作为本次环境下的实测成本。完整配置与逐seed数据见工件runs.csv，",
              "均值/SD/95%CI见summary.json，配对RMSE/NASA差值见paired_comparisons.json；原始预测与checkpoint逐run保留。"]
    (docdir/"MATCHED_COMPARISON_REPORT_20261112.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def collect(remote_root,data_dir,target,docdir):
    remote_root,target=Path(remote_root).resolve(),Path(target).resolve();target.mkdir(parents=True,exist_ok=True)
    # Progress collection must fail before opening any official labels.
    expected={(m,fd,s) for m in MODELS for fd in FDS for s in SEEDS}
    batch=read(remote_root/"BATCH_FREEZE.json")
    registered=[(r["model"],r["subset"],r["seed"]) for r in batch["runs"]]
    assert batch["frozen"] and len(registered)==len(set(registered))==84 and set(registered)==expected
    for model,fd,seed in expected:
        source=remote_root/"matched"/model/fd/f"s{seed}"
        assert (source/"COMPLETED.json").exists() and (source/"TEST_ATTEMPT.json").exists(),source
    (target/".gitattributes").write_bytes(b"* -text -whitespace\n")
    rows=[];audits=[]
    for model in MODELS:
        for fd in FDS:
            truth=[min(125,float(x)) for x in (Path(data_dir)/f"RUL_{fd}.txt").read_text().split()]
            for seed in SEEDS:
                source=remote_root/"matched"/model/fd/f"s{seed}"
                assert (source/"COMPLETED.json").exists(),source
                m=read(source/"metrics.json");replay(read(source/"history.json"),m)
                pred=list(csv.DictReader((source/"test_predictions.csv").open()))
                for r in pred:assert float(r["y_true"])==truth[int(r["unit"])-1]
                r,n=score(pred);assert abs(r-m["rmse"])<1e-5 and math.isclose(n,m["nasa_sum"],rel_tol=2e-7,abs_tol=.01)
                state=torch.load(source/"checkpoint.pt",map_location="cpu",weights_only=True)
                assert all(torch.isfinite(t).all() for t in state.values())
                assert state_hash(state)==m["state_content_sha256"] and digest(source/"checkpoint.pt")==m["checkpoint_sha256"]
                dest=target/"runs"/model/fd/f"s{seed}";dest.mkdir(parents=True,exist_ok=True)
                for path in source.iterdir():
                    if path.is_file():shutil.copy2(path,dest/path.name)
                write_json(dest/"checkpoint_shapes.json",{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in state.items()})
                for phase in ["train","test"]:
                    log=remote_root/"logs"/f"{phase}_{model}_{fd}_{seed}.log"
                    (dest/("training_stdout.log.gz" if phase=="train" else "test_stdout.log.gz")).write_bytes(gzip.compress(log.read_bytes(),mtime=0))
                row={k:m[k] for k in ["model","subset","seed","rmse","nasa_sum","n","params","trainable_params","training_time_s","validation_engine_mean_rmse","validation_pooled_rmse","n_train_windows","best_epoch","epochs_run","config_sha256","checkpoint_sha256"]}
                row["path"]=dest.relative_to(target).as_posix();rows.append(row)
                audits.append({"model":model,"subset":fd,"seed":seed,"recomputed_rmse":r,"recomputed_nasa_sum":n,"checkpoint_finite":True,"checkpoint_state_hash_valid":True,"official_labels_match":True})
    with (target/"runs.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(target/"metric_audit.json",audits)
    evidence=target/"evidence";evidence.mkdir(exist_ok=True)
    for name in ["BATCH_FREEZE.json","preflight.json","preflight_attempt1.json","train_matrix.json","test_matrix.json","RESOURCE_LAUNCH_CONTROL.json","RESOURCE_LAUNCH_CONTROL_V2.json","MEMORY_SCHEDULER_train_V1_REPLACED.json","metadata_acceptance_checks.json","ORCHESTRATOR_train_RESOURCE_PAUSE_REPLACED.json"]:
        if (remote_root/name).exists():shutil.copy2(remote_root/name,evidence/name)
    for name in ["PROTOCOL.json","SNAPSHOT_MANIFEST.json","EXECUTION_SOURCE_MANIFEST.json","DATASET_EXPECTED_SHA256.json","ANALYSIS_PLAN.json"]:
        shutil.copy2(ROOT/name,evidence/name)
    for name in ["preflight.log","preflight_attempt1.log","train_matrix.log","train_resume_matrix.log","train_resume_v2.log","test_matrix.log"]:
        path=remote_root.parent/name
        if path.exists():(evidence/(name+".gz")).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    tools=evidence/"evidence_tools";tools.mkdir(exist_ok=True)
    for name in ["collect.py","verify.py","monitor.py","memory_queue.py","memory_queue_v2.py"]:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,tools/name)
    frozen=read(ROOT/"EXECUTION_SOURCE_MANIFEST.json")
    for item in frozen["files"]:
        rel=item["path"]
        if rel.startswith("paper_b/baselines/open_source_matched_20261112/") or rel in {"paper_b/baselines/open_source/adapters.py","paper_b/baselines/open_source/common.py","paper_b/baselines/open_source/sources.lock.json"} or rel.endswith("/SOURCE_MANIFEST.json"):
            source=ROOT.parents[2]/rel;dest=evidence/"execution_code"/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    write_json(target/"summary.json",summarize_rows(rows));write_json(target/"paired_comparisons.json",compare_rows(rows));manifest(target)
    from .verify import verify
    verify(target)
    report(target,Path(docdir),rows)
    write_json(Path(docdir)/"MATCHED_STATUS_20261112.json",{m:{"status":"SUCCESS","scope":"new matched protocol evidence/comparability; not author-number gate","n_runs":12} for m in MODELS})
    print("COLLECTED_REAL_MATCHED_RUNS",len(rows),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--remote-root",required=True);p.add_argument("--data-dir",required=True);p.add_argument("--target",required=True);p.add_argument("--docdir",required=True)
    a=p.parse_args();collect(a.remote_root,a.data_dir,a.target,a.docdir)
