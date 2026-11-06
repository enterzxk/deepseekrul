# INDEX — full_20261105 目录说明（2026-11-05 21:45）

服务器权威路径：`/home/user/rulb/tim_20261105/full_20261105`
本地镜像：`C:\Users\20562\Desktop\deepseek-rul\tim_20261105_full\`

## 文档

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结协议 + §8 变更记录（淘汰、缺陷修正、端点、预算、初始化噪声） |
| `FINAL_DECISION.md` | A/B/C/D 逐项判定（UNMET）与证据路径 |
| `KEY_FINDINGS.md` | 一页结论汇总 |
| `SESSION_REPORT.md` | 会话执行报告 |
| `NOVELTY_MATRIX.md` | N1–N4 最近邻工作（DOI 已核实）与可证伪实验 |
| `IDENTIFIABILITY.md` | 可辨识性负结果（门控、证据、灵敏度） |
| `ENDPOINT_DEPENDENCE.md` | 端点/预算依赖 |
| `REPRODUCIBILITY.md` | 初始化噪声披露与复算顺序 |
| `LEAKAGE_AUDIT.md` | 泄漏审计 + 4/4 因果性测试 |
| `RESULTS_NARRATIVE.md` | 结果叙述（主表、消融、可观测性、可辨识性、端点、NASA 尾部、胜率、功效、测量条件） |
| `PAPER_DRAFT.md` | 英文论文完整草稿（I–IX） |
| `PAPER_METHODS.md` / `PAPER_DISCUSSION.md` / `PAPER_RESULTS.md` | 分节草稿 |
| `TIM_PAPER_PACKAGE.md` | 写作材料包（含"可以主张/不可以主张"） |
| `CAPTIONS.md` | 图注与表注 |
| `HANDOFF.md` / `GOAL_STATE.json` | 交接与目标状态 |
| `INDEX.md` | 本文件 |

## 代码

| 文件 | 作用 |
|---|---|
| `run_full_fold.py` | 完整预算单折训练（M0/OPS/FULL/NORM） |
| `run_full_fold2.py` | 同上 + PHASE 家族 + 权重保存 |
| `run_full_fold3.py` | 表示筛选（DIFF/PHASE/GRAPH）+ 窗口/轮数/权重选项 + 权重保存 |
| `measurement_conditions.py` | MC1/MC2 评测侧扰动（用保存的权重） |
| `summarize_full.py` / `build_results.py` | 四指标与配对区间 |
| `make_paper_assets.py` / `make_extra_figures.py` / `make_ablation_table.py` / `per_engine.py` / `per_fold_table.py` | 表图 |
| `observability.py` | 可观测性曲线 |
| `evidence_gate.py` / `evidence_engine.py` / `agree_gate.py` / `sensitivity_gate.py` | 可辨识性 |
| `deploy_endpoint.py` / `test_fusion.py` / `fusion_lofo.py` | 端点与融合 |
| `which_engines.py` / `nasa_anom.py` | 逐机胜率与 NASA 尾部 |
| `repro_check.py` | 官方封存预测复算（16/16） |
| `lock_reference.py` / `make_spec.py` | 参考臂锁定与汇总配置 |
| `verify_delivery.py` | 交付完整性检查 |
| `assemble_delivery.sh` | 一键重生成表图与清单 |
| `queue*.sh` / `tasks*.txt` | 队列与任务表（实验记录） |
| `patch_*.py` | 开发修补脚本（保留以便审计） |

## 数据产物

| 路径 | 内容 |
|---|---|
| `runs/{subset}/{fold}/{arm}_{seed}.csv` | 逐窗口预测（主实验） |
| `runs2/` `runs3*/` | 消融与权重（含 .pt） |
| `measurement/` | MC1/MC2 逐窗口预测 |
| `observability.csv/json` | 可观测性曲线 |
| `RESULTS*.csv/json` `ABLATIONS.csv` `PER_ENGINE.csv` `MEASUREMENT_CONDITIONS.csv` | 汇总表 |
| `REFERENCE_LOCK.json` | 锁定参考臂 |
| `paper_tables/` `paper_figures/` | 论文表图 |
| `logs/` | 全部运行日志 |
| `MANIFEST.sha256` | 交付文件哈希 |

## 复算

```bash
PY=/home/user/miniconda3/envs/pmv/bin/python
cd /home/user/rulb/tim_20261105/full_20261105
bash assemble_delivery.sh
$PY run_full_fold.py FD002 0 --seed 101 --arms M0,OPS,FULL,NORM
```
