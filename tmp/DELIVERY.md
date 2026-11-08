# DELIVERY — 交付清单与复算入口（2026-11-05 22:50）

## 位置

- 服务器权威目录：`/home/user/rulb/tim_20261105/full_20261105`
- 本地镜像：`C:\Users\20562\Desktop\deepseek-rul\tim_20261105_full\`
- 前序产物（保留，未改动）：`/home/user/rulb/tim_20261105/`（官方测试结果、封存预测、
  协议、交接文档）

## 交付物（对应验收 D）

| 要求 | 文件 | 状态 |
|---|---|---|
| 冻结协议 + 变更记录 | `PROTOCOL.md` | 完成（§8 含 5 条变更记录） |
| 最近邻工作与可证伪实验 | `NOVELTY_MATRIX.md` | 完成（N1–N4，DOI 已核实；N2 公式待全文） |
| 训练/评估代码 | `run_full_fold*.py`、`measurement_conditions.py`、`build_results.py` 等 | 完成 |
| 配置与版本 | `runtime.json`、`spec.json`、`REFERENCE_LOCK.json` | 完成 |
| 运行记录 | `logs/`（39+ 个日志）、`tasks*.txt`、`queue*.sh` | 完成 |
| 模型文件 | `runs2/**/*.pt`、`runs3_w50*/**/*.pt` | 部分（主臂可由命令重训） |
| 逐发动机预测 | `runs/{subset}/{fold}/{arm}_{seed}.csv`（63 个） | 完成（seed 101 全矩阵） |
| 四指标主表 | `RESULTS.csv` / `RESULTS.json` | 完成（seed 101） |
| 消融表 | `ABLATIONS.csv` | 完成 |
| 测量条件 | `MEASUREMENT_CONDITIONS.csv` | 完成（3/5 折） |
| 逐发动机分析 | `PER_ENGINE.csv`、`paper_tables/per_fold.csv` | 完成 |
| 论文图 | `paper_figures/`（6 图，png+pdf） | 完成 |
| 论文表 | `paper_tables/`（5 个） | 完成 |
| 写作材料包 | `TIM_PAPER_PACKAGE.md`、`PAPER_DRAFT.md`、`PAPER_RESULTS.md`、`CAPTIONS.md` | 完成 |
| 最终判定 | `FINAL_DECISION.md` | 完成（UNMET） |
| 泄漏审计 | `LEAKAGE_AUDIT.md` + 4/4 因果性测试 | 完成 |
| 可复现性披露 | `REPRODUCIBILITY.md` | 完成 |
| 完整性清单 | `MANIFEST.sha256`（154 项）、`verify_delivery.py` | 完成 |

## 复算入口

```bash
PY=/home/user/miniconda3/envs/pmv/bin/python
cd /home/user/rulb/tim_20261105/full_20261105
bash assemble_delivery.sh            # 重生成全部表图 + 校验 + 清单
$PY run_full_fold.py FD002 0 --seed 101 --arms M0,OPS,FULL,NORM   # 单折训练
$PY repro_check.py                   # 官方封存预测 16/16 复算
$PY verify_delivery.py               # 交付完整性
```

## 主要结论（供写作）

1. **A FAIL**：冻结候选 NORM 在 FD002（4 折 14.513）与 FD004（5 折 14.789）上
   显著劣于参考 FULL（13.054 / 13.253），G = −11.2% / −11.6%。
2. 其余机制候选：PHASE +3.909（显著更差）；DIFF/EQBAL/GRAPH 中性；
   逐观测门控全部劣于固定融合。
3. **C PASS**：可观测性曲线（0.26→0.92）、可辨识性负结果（|ρ|<0.08，oracle 仍差 16%）、
   测量条件（缺测 +14%、偏移 +142%）、NASA 尾部主导、逐机胜率 55%/53%。
4. **B PARTIAL**：没有可辩护的新机制；贡献在测量学分析。

## 未完成（如实标注）

- FD002 种子 202/303 的全 5 折（202 已有 4 折、303 已有 2 折）。
- FD003 folds 2/4。
- 跨折测量条件 folds 4/5。
- 主臂模型权重（仅消融臂保存）。

这些完成后重跑 `assemble_delivery.sh` 即可把 3 种子均值与 FD003 护栏写入最终表。
