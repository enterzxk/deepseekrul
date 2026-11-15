# FAIR_COMPARISON_AUDIT：旧 sensor_aug 结论的代码溯源

> 2026-11-15。只读核对，不改旧 study 的任何产物。
> 问题：从旧 study 取回的模型侧实现，是否就是产出旧 G1–G4 数字的那一份？§6「直接引用旧 clean 结论」是否成立？

## 1. 结论（三句）

1. **是同一份代码。** 36 格全部在 **03:31–05:32** 之间启动，晚于旧 study 最后一次代码提交（`8734e126`，02:08:13）。所有格用的是同一版：7 文件指纹 **`90f20d35953d`**。
2. **manifest 里记的 `fda45915b492` 是建 manifest 时（约 01:40–01:50）未提交工作树的指纹**，早于三次代码提交，**不能代表跑数时的代码**。这是记录缺陷（指纹冻结太早、之后没重算），不是代码版本混杂，**不需要重跑**。
3. **§6 可以引用，但必须带三条边界**：`fp32=false`（BF16 训练）、`split_seed=7`、只有 FD001/FD004 两个子集。它不是本轮的 FP32/四子集对照。

## 2. 取回了什么

| 文件 | sha256[:16] | 字节 |
|---|---|---:|
| `paper_b/models/dspt.py` | `7eedabf768307db2` | 18748 |
| `paper_b/training/train.py` | `e9fbcc5b386edf2d` | 17856 |
| `paper_b/training/data.py` | `c58a3db0d419f4f6` | 15117 |
| `paper_b/training/metrics.py` | `7d8bb2add76d8a91` | 4540 |
| `paper_b/sensor_aug/augmentation.py` | `67d755f588b494c1` | 5719 |
| `paper_b/sensor_aug/noise_stats.py` | `1072d2ec657e3e70` | 6526 |
| `paper_b/sensor_aug/evaluation.py` | `643f1dd40770244b` | 12100 |
| **合并（旧 study 的 `code_fingerprint` 口径）** | **`90f20d35953d`** | |

旧 study 仓库工作树与新分支 worktree 的这 7 个文件**逐字节相同**（上表两边一致）。

## 3. 指纹为什么对不上

| 时点 | 记录 | 实际 |
|---|---|---|
| 建 manifest（约 01:40–01:50） | `code_fingerprint=fda45915b492`，`git_head=51307439` | 工作树已有 `paper_b/sensor_aug/`，但 `51307439` 里没有该目录（`git show` 报 MISSING）→ 代码当时**未提交** |
| 01:52:43 `ff262eb1` | — | 首次提交 harness，**`sensor_keep` 在这一提交引入 `models/dspt.py`**；该提交指纹 `4eb45201ac56` |
| 01:58:48 `5fc87694` | — | evaluation 调整，指纹 `87ff6d495804` |
| 02:06:25 `3ab62ead` | — | `train.py` 验证改为 FP32 评测（影响选轮），指纹 `d23f1f837ed6` |
| 02:07:10 `2f663291` | — | 只改 `reporting.py` |
| 02:08:13 `8734e126` | — | `data.py` 测试窗带 cutoff cycle，指纹 **`90f20d35953d`** |
| 03:31–05:32 | — | **36 格实际训练**（首格 config.json 03:31，末格 05:32） |
| 05:40/05:41 | `46934d5d` / `2cc26ebb` | 报告与文档，指纹仍 `90f20d35953d` |

`fda45915b492` 不对应任何一个 commit：它是「建 manifest 那一刻的工作树」，而那一刻的代码**从未训练过任何一格**（第一格 03:31 才启动）。

## 4. 36 格是否同一版本

是。启动时间 03:31–05:32，全部晚于 `8734e126`（02:08:13），因此都加载 `90f20d35953d`。
- 02:06 的 `train.py` 改动（验证 FP32）在**所有格之前**生效，不存在「前半格用旧选轮、后半格用新选轮」。
- 02:08 的 `data.py` 改动只影响 test-role 窗口，不影响 Stage 1 的 validation 数字。

**若任何一格在 02:08 之前启动，本节结论作废。** 实测最早 03:31，不成立。

## 5. 旧数字的协议边界（引用时必须写）

从每格 `config.json` 实测：

```
"fp32": false            ← BF16 训练（验证在 02:06 之后是 FP32 评测）
"split_seed": 7
"subset": FD001 / FD004  ← 只有两个子集
"rul_cap": 125, "cuts": [0.4,0.55,0.7,0.85,0.95]
```

所以旧 clean 结论（五组全负、−2.45%~−0.14%）可以说「在这个协议下没有 clean 增益」，**不能**说「在 FP32、四子集、split 17 下也没有」。这正是本轮 E1a/E1b 要重跑的理由——不是因为代码不可信，而是因为协议不同。

## 6. 我这次移植漏了什么（根因）

`models/dspt.py` 没被搬过来，原因不是「模型侧改动没进 git」——它在 `ff262eb1` 里，一直在 git 里。
根因是我取 diff 时用了 `git diff --stat 51307439 f237e80b | tail -20`，**只看到末尾 20 行**，`paper_b/models/dspt.py` 在列表上方被截掉了；随后按可见文件复制，漏了 models。

已修正：`paper_b/models/dspt.py` 从旧 study 取回（含 `sensor_keep` 的 attention key-mask、encode 清洗、pooling 重归一化），新分支 7 文件指纹与旧 study 一致（`90f20d35953d`）。

**教训**：diff 列表不得用 `tail` 截断后当作完整清单；搬运代码必须用合并指纹验收，而不是「文件看起来齐了」。

## 7. 对 §6 的最终裁定

| 原表述 | 裁定 |
|---|---|
| 「mask 无 clean 增益，直接引用、不重跑」 | **成立**，但必须附 §5 三条边界（BF16 / split 7 / 两子集） |
| 「噪声注入臂无增益（G3/G4）」 | 同上，引用时注明协议 |
| 「G2 的 +8.73% 是噪声鲁棒性、不是噪声注入」 | 成立，见 `SENSOR_RESEARCH_PLAN.md` §5 |
| E1a/E1b 的性质 | **仍为重做**（协议不同），不是「复核同一协议」 |

## 8. 复核命令

```bash
# 旧 study
python fp_check.py /home/user/sensor_aug_20261114        # → COMBINED 90f20d35953d
# 新分支
python fp_check.py <worktree>/paper-b-sensor-generalization  # → COMBINED 90f20d35953d
# 每格启动时间
ls -l --time-style=+%H:%M runs/*/*/*/config.json | sort -k6
```
