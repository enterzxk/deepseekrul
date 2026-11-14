# 阶段六最终验收记录（2026-11-14）

验收对象：`docs/paper_b_edc/attribution_24h/` 与本轮推送（分支 `paper-b-contribution-attribution`）。
口径：只读核对；未改动任何实验产物；所有数字可由下列命令复算。

---

## 1. 判定

| 项 | 结果 | 证据 |
|---|---|---|
| 证据文件齐备 | ✅ 20 份 + 本记录 | 见 §2 |
| 检查脚本全绿 | ✅ 5 个脚本 | 见 §3 |
| 无未推送提交 | ✅ `unpushed=0` | `git log origin/..HEAD` → 0 行 |
| 工作树干净（本目录） | ✅ | `git status --porcelain docs/paper_b_edc/attribution_24h` → 0 行 |
| 无 checkpoint | ✅ `model.pt` / `.ckpt` = 0 | `git ls-files | grep '\.pt$|\.ckpt$'` → 0 |
| 预测文件（见 §4） | ⚠ 保留（按 A 方案） | 167 个 / 443,335 B，**本轮新增 0** |

---

## 2. 文件清单（20 份）

核心九份：`IDENTITIES.md`、`ATTRIBUTION_B1_B5.md`、`A8_96_REUSE_INVENTORY.md`、`RUN_QUEUE.csv`、`RUN_QUEUE_V0N.md`、
`LEGACY_VAL_FIELDS.md`、`UNRESOLVED_ERRORS.md`、`PIVOT_QUESTION.md`、`PREREGISTRATION.md`。

阶段五新增：`REUSABLE_ASSETS_FOR_PIVOT.md`。
Task C：`TASK_C_FEASIBILITY.md`、`TASK_C_NOTES.md`、`TASK_C_CONDITIONAL_ERRORS.csv`（283 行）。
数据/脚本：`A0_RECOMPUTE.csv`、`CYCLE_PAIR_ENGINE_VAL.csv`（24 行）、`V0N_PAIRED_DELTAS.csv`（12 行）、
`VAL_RECOMPUTE.csv`（265 行，76.9 KB）、`recompute_a0.py`、`recompute_val.py`、`task_c_conditional_errors.py`。

---

## 3. 检查脚本与实测输出

```
identity_check.py     compared=12 mismatched=0
                      F_full n=12 val=10.9814 test=11.5116 ; A8_ms n=12 val=11.3404 test_n=4 11.3867
                      A8_base n=12 val=11.3118 ; runs.csv rows=128 ; F_ 各族 rows=8 seeds=[202,303]
a8_reuse_check.py     A8 rows=96 arms=8 subsets=4 seeds=3 ; test filled=36 (frozen 32 + extra 4)
                      main effects ms −0.0818 / sa −0.1625 / hi −0.0495（主口径）
pivot_assets.py       A8 rows=96 ; V0N rows=12 ; distinct runs=108 ; points 17040 / 2130 / 710
                      train-side engines {FD001:80, FD002:208, FD003:80, FD004:199}
reg_diff.py           runs.csv 128 = ok 97 + failed 24 + smoke 6 + superseded 1 ; A8 rows in runs.csv=4
attribution_tables.py base 四子集 + vanilla_plus_inputmode 12 对逐位复现
```

---

## 4. 预测文件与体积（按 A 方案保留）

```
paper_b/results/ 下有 167 个 *pred*.csv（合计 443,335 字节），全部早于本轮
（merge-base aaa4e8ba，本轮新增 0）；model.pt/.ckpt 为 0；保留以维持可复算性。
```

构成（`git cat-file -s` 逐文件求和）：

| 类别 | 个数 | 字节 |
|---|---:|---:|
| `val_pred.csv` / `test_pred.csv`（逐 run 逐点预测） | 166 | 437,015 |
| `legacy_preds_bins.csv`（历史分箱表） | 1 | 6,320 |
| **合计** | **167** | **443,335** |

**保留理由**：这 167 个文件是本轮所有可复算性的来源——`VAL_RECOMPUTE.csv`（265 run）、
`TASK_C_CONDITIONAL_ERRORS.csv`、B-1 的 12 对配对差全部由它们算出。移出版本控制等于拆掉「每个数都能复核」这条硬资产。
规则的原文是「不上传**大型**原始数据与 checkpoint」：443 KB / 167 文件（平均 2.6 KB），不是大型数据，本轮新增 0。

**我上一版报的 166 / 437,015 是窄口径**（只匹配 `val_pred|test_pred|preds`），漏了 `legacy_preds_bins.csv`；
经理的 167 / 443,335 是完整口径，本记录采用后者。

---

## 5. 环境与路径事实（下一轮接手须知）

| 资产 | 完整副本在哪 | 本地缺什么 |
|---|---|---|
| A8 96 格逐点预测 | 3090 `~/rulb/paper_b/results/raw/A8_*` | 本地只有 `A8_base_L70`、`A8_mssa_L70` 两臂 |
| V0N 12 格 | 3090 `~/rulb/paper_b/results/raw/V0N_L70` | 本地无（`V0N_PAIRED_DELTAS.csv` 是摘要） |
| A0 四格预测 | a40 `/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/HC-C-A0-*/preds.csv` | 本地无 |

**任何依赖完整 96 格或 V0N 的重算，必须在 3090 上做**；本地只能复算已提交的摘要表。

---

## 6. 已知限制（验收不含以下主张）

1. `0/4`、`0/28`、`0/36` 都是**未检出**，不是"均值已无增量"。
2. `A8_engine_level.csv` 是主口径（标签封顶、预测不裁）；另裁预测的版本只作敏感性。
3. `F_full_basic_hi1` ≡ `A8_full_L70` 是别名，独立 run 数 108 而非 120。
4. 区间须按发动机内重复处理（同一台机 5 个截断点不是 5 个独立样本）。
