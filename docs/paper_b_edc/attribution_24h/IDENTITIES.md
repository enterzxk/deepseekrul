# 三个基线身份的索引（本轮 attribution_24h 的入口）

本文件只做索引与口径声明，**不复述数字**。数字的唯一来源是下列两份文档与
`python tmp/audit_hc/identity_check.py` 的输出；本文若与之冲突，以那两处为准。

## 权威文档

| 内容 | 路径 |
|---|---|
| 身份钉死与协议收窄 | `docs/paper_b_baselines/DSPT_IDENTITY_PINNING_20261114.md` |
| 身份核验、可复用清单、口径陷阱 | `docs/paper_b_hc/RESULTS_AUDIT_20261114_IDENTITIES.md` |
| 机器核验 | `python tmp/audit_hc/identity_check.py` → `compared=12 mismatched=0` |

## 三个身份（定义行，数字见上表）

| 名字 | 配置 | 对应 run 族 |
|---|---|---|
| `DSPT-V0-Cycle` | `ms=full, sa=basic, hi=hf, use_cycle=True` | `F_full_basic_hi1` |
| `DSPT-TrainDefaults` | `ms=full, sa=off, hi=none, use_cycle=True` | `A8_ms_L70`（`params=82114`，与 `DSPT(max_span=70)` 默认构造逐位相等） |
| 历史 `Selected` | `ms=single_proj, sa=basic, hi=hf, use_cycle=True` | `F_single_proj_basic_hi1` |

## 三条必须一起读的口径声明

1. **`A0` 不是 TrainDefaults**。`A0` 是 `cycle_only=True`（只喂年龄、无传感器）的年龄臂，
   把它当基线会把基线写差约 3 倍。TrainDefaults 是 `A8_ms_L70`。
2. **`runs.csv` 的 128 行不是全量**。`F_*` 各族只登了 `s202/s303`（8 行），`A8` 只登了
   冻结清单外那 4 格；引用「12 run」必须引 `paper_b/results/`，引 `runs.csv` 会得到 8。
   且 `runs.csv` 里 `F_*` 的 val 字段是 **pooled**，不是 engine-level。
3. **同 27 字段不等于只剩结构差**。至少还有三项未排除：选轮准则（pooled vs engine-level）、
   聚合方式（`pooled ≥ engine-level`，Jensen）、数值精度（未记录 vs FP32）。
   其中「seed 数」一项已排除，但只对 `V0-Cycle` 这一对成立；`TrainDefaults` 的 test
   只有 `s101`（`n=1`/子集），那里 seed 数是真差异。

## 原件引用

`docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md` 存在三个 ref，**正文相同、横幅不同**：

| ref | blob | 字节 | 说明 |
|---|---|---|---|
| `codex/paper-b-open-source-reproduction` @ `d7bd9c17` | `d76c4aef` | 15454 | **原件**，无横幅 |
| `paper-b-contribution-attribution` @ `042a1405` | `4df7b410` | 16895 | 原件 + 恢复横幅 |
| `origin/punet-interval` @ `042a1405` | `4df7b410` | 16895 | 同上，已合入 |

**引用「原件」要指 `d7bd9c17` 那个 blob**，否则日后「原件 16895 字节」与 git 历史对不上。
