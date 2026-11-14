# Paper B · Task A 三身份钉定与口径对账（2026-11-14）

> **本文不是 `MATCHED_COMPARISON_REPORT_20261112.md` 的重建。**
> 那份报告**从未丢失**：它存在于分支 `codex/paper-b-open-source-reproduction`、提交 `d7bd9c17`，
> blob `d76c4aef0a4e`、`15454` 字节、193 行、9 节，已于本日逐字节恢复到本分支（恢复件顶部带来源横幅）。
> **它比较的是 DSPT 与 5 篇公开基线**，不是本文要做的「`V0-Cycle / TrainDefaults / Selected` 三者身份对账」。
> **后者此前从未作为单一文件存在**，本文是它的首次汇编。
>
> **来源**：`paper_b/reports/SEED_ROBUSTNESS.md`（`seed_report` 机器生成）、
> `docs/paper_b_baselines/RUN_INVENTORY.md` §5、`docs/paper_b_baselines/BASELINE_PROTOCOL_AUDIT.md` §6b/§7、
> `docs/decision_log.md` D039（X2/X3 裁决）、`docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md`（恢复件）。
> **本文不含任何新训练结果**，只做对账与标注。

## 0. 结论摘要

1. **三个身份都能钉死到配置字段**，依据是 D039 X3 的显式表 + `RUN_INVENTORY` §5 的字段级 diff。
2. **但同一个「`DSPT-V0-Cycle`」名下有两个不同的 TEST 数**（`11.512` 与 `11.1725`），
   差别来自**精度协议**（默认 BF16 vs 统一 FP32），**不是来自模型**。原件 §1 原文即声明「不与历史 DSPT 默认 BF16 结果声称数值等价」。
   **因此 Task A 固定身份时必须同时固定精度口径**，否则后续任何「新机制 vs 强基线」的比较都是跨口径相减。
3. **`V0N`（无 cycle）从未跑过**——`runs.csv` 里 V0C/V0N 都无记录，`V0-Cycle` 是以 `F_full_basic_hi1` 的 tag 存在的。
   所以 Task B 第 1 项「Cycle 的独立贡献」**必须新训练**，不可能靠复用。
4. **`DSPT-TrainDefaults` 的身份仍待核**（见 §4）；它在 FP32 口径下的数在原件 §3 里，但**没有 3 seed 的 engine-level val 记录**。
5. 「base」在本项目里有 **4 个不同含义**，引用前必须写三开关取值 + tag 前缀。

## 1. 三个身份的配置钉定

| 名字 | `ms_mode` | `sa_mode` | `hi_source` | `use_cycle` | `input_mode` | 可定位的 tag | 依据 |
|---|---|---|---|---|---|---|---|
| **`DSPT-V0-Cycle`**（主参考） | `full` | `basic` | `hf` | `True` | `raw` | **`F_full_basic_hi1`** ≡ **`A8_full_L70`** | D039 X3 表；`RUN_INVENTORY` §5.2 #5 |
| **`DSPT-TrainDefaults`**（辅助参考） | `full` | `off` | `none` | `True` | `raw` | **待核**（疑似 `A0`，4 行） | 原件 §1「literal train DEFAULTS」；D039 X2 记 DEFAULTS 为 `sa_mode='off'` |
| **历史 `Selected`** | `single_proj` | `basic` | `hf` | `True` | `raw` | **`F_single_proj_basic_hi1`** | `SEED_ROBUSTNESS` 的 `cand` 行；factorial 线 |
| （Task B 需要，**未跑**）`V0-NoCycle` | `full` | `basic` | `hf` | **`False`** | `raw` | 无 | D039 X3 已定义，`runs.csv` 无记录 |
| （Task B 需要，已存在）`V0-NoHI` | `full` | `basic` | **`none`** | `True` | `raw` | **≡ `A3`**（D039 X3 明写「同配置，只跑一次、双向引用」） | D039 X3 |

**三个身份共享的 27 个训练/划分字段**（`RUN_INVENTORY` §5.1 已逐字段核对）：
`span=70, epochs=40, patience=12, batch=512, lr=1e-3, weight_decay=1e-4, d_model=64, layers=2, dropout=0.2,
mode=dual, hi_mono_coef=0.1, rul_cap=125, split_seed=7, fold=0, val_frac=0.2, n_cond=6, cuts=[0.4,0.55,0.7,0.85,0.95], …`
**所以三者之间的差不是预算或划分造成的，只能归到结构/输入表示。**

## 2. 口径分裂：同一配置的两个 TEST 数

**这是本文最要紧的一节。**

| 配置 | 口径 | val（engine-level，4 子集均值） | test（4 子集均值） | 逐子集 test | 来源 |
|---|---|---:|---:|---|---|
| `V0-Cycle`（= `full`） | **默认 BF16** | **10.981** | **11.512** | `10.379 / 10.988 / 12.317 / 12.362` | `SEED_ROBUSTNESS` test 表 `full` 行 |
| `V0-Cycle-FP32` | **统一 FP32** | 无记录 | **11.1725** | `10.60 / 11.03 / 10.68 / 12.38` | 原件 §3/§6 主参考行 |
| `Selected`（= `cand`） | **默认 BF16** | **10.896** | **11.134** | `10.854 / 10.960 / 10.799 / 11.922` | `SEED_ROBUSTNESS` test 表 `cand` 行 |
| `Factorial base`（= `base`） | **默认 BF16** | **11.364** | **11.655** | `11.140 / 11.544 / 11.130 / 12.806` | `SEED_ROBUSTNESS` test 表 `base` 行 |
| `TrainDefaults-FP32` | **统一 FP32** | 无记录 | **11.23** | `10.07 / 11.20 / 11.16 / 12.49` | 原件 §3 辅助参考行 |

（均值均为本文按四个子集等权计算：`V0-Cycle-FP32` = `(10.60+11.03+10.68+12.38)/4 = 11.1725`；`TrainDefaults-FP32` = `11.23`。）

**三条硬规则**：

1. **`11.512` 与 `11.1725` 不能互相替代**，也不能各自与对方的基线相减。原件的 FP32 条件是为「与 5 篇公开基线同台」而设的，**它明确不声称与历史 BF16 数值等价**（原件 §1 末句）。
2. **`11.512` 与 `11.1725` 的差（`0.34`）与 `Selected` 对 `full` 的效应（`−0.378`）是同一量级——占其 `90%`**（`0.3395 / 0.378 = 0.898`）。**精度口径的位移几乎等于要被检验的效应**，这是「不混口径」在本项目里最具体的一次理由：**换一次精度，就能把一条「无法区分」的比较推向任何一侧。**
3. **`SEED_ROBUSTNESS` 的 val 是事后重算的 engine-level 值，而选模当时用的是 pooled**（该文件 §Validation 明写：GPU 主机上仍是 pre-R2 `metrics.py`，pooled 选轮，engine-level 事后重算）。**所以引用它的 val 时必须带这句**，否则会被读成「用 engine-level 选模」。

## 3. `Selected` 对两套参考的配对结果（Task A 的现实起点）

**对 `full`（`Selected − full`，负 = `Selected` 更低）**：`t` 区间**0/4 子集排除 0**，符号保持 **1/4**，12 个配对差均值 **`−0.378`**（`sd 1.057`）。
逐 seed 里只有 `FD003 s202`（`−1.758`，CI `[−3.092, −0.411]`）与 `FD003 s303`（`−2.646`，CI `[−4.520, −0.992]`）、`FD004 s303`（`−1.007`，CI `[−1.988, −0.019]`）三对排除 0，**即 3/12**。

**对 `base`（`Selected − base`）**：`t` 区间**1/4 子集排除 0**（仅 `FD004`，`−0.883`，CI `[−1.322, −0.445]`，`3/3` 同号），符号保持 **1/4**，12 个配对差均值 **`−0.521`**（`sd 0.814`）。

> **这就是 Task E 里 STOP 的现实概率**：`Selected` 相对 `full` 在 test 上 **0/4 子集可判**。
> 「架构优势」本身没有被证实——**这不是新发现，是 Task A 必须继承的起点**。
> **同时注意**：这两组都是 **test** 上的描述性比较，`Selected` 是**用 validation 选的**（`SEED_ROBUSTNESS` 明写「Selection rule is unchanged: validation only」），
> 所以 test 上的差不是选择偏倚的产物；**但也因此不能反过来用 test 结果重新选配置**（用户禁令）。

## 4. 待核与缺口（不假装已闭合）

| # | 项 | 现状 | 谁去核 |
|---|---|---|---|
| 1 | **`TrainDefaults` 的真实 tag / 运行记录** | 原件 §1 称其为「literal train DEFAULTS（`full/off/none/cycle=True`）」，但**仓库里没有 3 seed 的 engine-level val**；经理称「疑似 `A0`，4 行」 | 实验结果管理员 |
| 2 | **`V0N` 是否真零运行** | `runs.csv` 无 V0C/V0N 记录；`V0-Cycle` 以 `F_full_basic_hi1` 存在 → 命名与 tag 不同源 | 实验结果管理员 |
| 3 | **`A8_full_L70` ≡ `F_full_basic_hi1`** 的去重 | 同一批 12 run 两个 tag，**任何合并表必须显式映射**，否则重复计数 | 台账维护方 |
| 4 | **`MATCHED_COMPARISON_REPORT` 的跨分支可见性** | 已在另一分支存在却被判「不存在」——**检查工具用错分支范围**（见 §5） | 已修正，见恢复件横幅 |
| 5 | **「已有 96 格消融」的确切所指** | `runs.csv` stage C 100 行（93 非 smoke）、`F_` 系列 24 行，都对不上 96 | 用户/经理指定后才能保证不重复执行 |

## 5. 一处方法学：`git log` 不带 `--all` 会把另一分支的文件判成「从未存在」

本次事故的完整形态：

```
git log --oneline -- "*MATCHED_COMPARISON*"        → 空（结论：仓库里没有、历史也没删过）
git log --all --oneline -- "*MATCHED_COMPARISON*"  → d7bd9c17 Complete DSPT-matched public baseline comparison
git branch -a --contains d7bd9c17                  → codex/paper-b-open-source-reproduction（本地 + origin）
```

**`git log -- <path>` 只遍历当前分支的祖先**，所以另一条分支上的文件不但「看不到」，而且**在历史里也不留痕迹**——
它看起来就像「从未被提交过」，而不是「在别处」。**「查过历史」这句话必须带分支范围**，否则它与「查过当前分支」是两件事。

**自查问法**：**「我说『不存在』时，遍历的是哪些 ref？」** 若答案是「默认的」，那结论只覆盖 `HEAD` 的祖先。

## 6. 复算方式

```powershell
cd C:\Users\20562\Desktop\deepseek-rul
# 1. 确认原件在另一分支（不是丢失）
git log --all --oneline -- "*MATCHED_COMPARISON*"
git branch -a --contains d7bd9c17
# 2. 取原件并核对字节
git rev-parse d7bd9c17:docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md
git cat-file -s d7bd9c17:docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md   # 期望 15454
# 3. 三套 DSPT 细胞的 val/test（机器生成，勿手改）
python -m paper_b.experiments.seed_report
# 4. 同名不同义的字段级依据
git show HEAD:docs/paper_b_baselines/RUN_INVENTORY.md        # §5.1/§5.2
git show HEAD:docs/paper_b_baselines/BASELINE_PROTOCOL_AUDIT.md   # §6b/§7
```
