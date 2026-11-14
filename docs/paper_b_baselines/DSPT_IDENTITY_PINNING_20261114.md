# Paper B · Task A 三身份钉定与口径对账（2026-11-14）

> **本文不是 `MATCHED_COMPARISON_REPORT_20261112.md` 的重建。**
> 那份报告**从未丢失**，此前「仓库里没有、历史也没删过」的判断**作废**。准确位置如下（三个 ref，别混）：
>
> | ref | 内容 | blob | 字节 |
> |---|---|---|---:|
> | `codex/paper-b-open-source-reproduction` @ `d7bd9c17` | **原始文件，无任何附加** | `d76c4aef0a4e` | `15454` |
> | `paper-b-contribution-attribution` @ `042a1405`（本分支） | 原件 + 顶部来源横幅 | `4df7b410a1b8` | `16895` |
> | `origin/punet-interval` @ `042a1405` | 同上（本分支已被合入） | `4df7b410a1b8` | `16895` |
>
> **两个 blob 不同，差的就是那 1441 字节横幅；横幅以下正文与 `d76c4aef` 逐字节相同**（`sha256 588dc4232ab59468` 两侧一致，已实核）。**引用「原件」时请指 `d7bd9c17` 那个 blob，不是本分支的。**
> **它比较的是 DSPT 与 5 篇公开基线**，不是本文要做的「`V0-Cycle / TrainDefaults / Selected` 三者身份对账」。
> **后者此前从未作为单一文件存在**，本文是它的首次汇编。
>
> **来源**：`paper_b/reports/SEED_ROBUSTNESS.md`（`seed_report` 机器生成）、
> `docs/paper_b_baselines/RUN_INVENTORY.md` §5、`docs/paper_b_baselines/BASELINE_PROTOCOL_AUDIT.md` §6b/§7、
> `docs/decision_log.md` D039（X2/X3 裁决）、`docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md`（恢复件）。
> **本文不含任何新训练结果**，只做对账与标注。

## 0. 结论摘要

1. **三个身份都能钉死到配置字段**，依据是 D039 X3 的显式表 + `RUN_INVENTORY` §5 的字段级 diff。
2. **但同一个「`DSPT-V0-Cycle`」名下有两个不同的 TEST 数**（`11.512` 与 `11.1725`），**而且它们不是同一个完整配置**：架构相同（`full/basic/hf/cycle=on`），**训练协议不同**——**`hi_mono` 一个是 `correct`、一个是 `none`**（实核见 §2.1），加上精度、选轮准则、聚合方式，**共四个未分离的变量**（见 §2）。
   **没有单变量对照，所以不能把 `0.34` 全归因到「BF16 → FP32」**——本文早先那版正是这样写的，已按军师 2026-11-14 的核对收窄。
   **因此 Task A 固定身份时必须同时固定整套协议（精度 + 选轮准则 + 聚合），不只是精度。**
3. **`V0N`（强骨干 + `use_cycle=False`）在已核查的档案范围内没有找到可复用运行。** **⚠ 这句话的上限就在这里——它不能升级成「历史上从未运行过」**（军师 2026-11-14 指出）：网格定义、目录扫描、台账标签**三者都是「档案里没有」**，而「档案里没有」推不出「没发生过」——可能跑在未归档的机器/目录上、可能跑过但没进任何台账。**所以补跑的依据是「证据缺口」，不是「确证未跑」；发射前必须由燕子按配置字段（`ms=full, sa=basic, hi=hf, use_cycle=False`）在 3090 侧全盘查重，不能只查台账。**
   已核的三条链：
   ① `paper_b/experiments/grids/nocycle.json` 只有 **12 条**，`tag` 全为 **`NC_base_L70`**、`use_cycle` 全 `False`、**`ms/sa/hi` 全为 `off/off/none`** ——**它定义的是 base 骨干的 cycle-off，不是 V0N**；
   ② 盘上无任何 `V0*` 目录（管理员独立扫描）；
   ③ `runs.csv` 无 V0C/V0N 标签（**仅此一条不足以定论**，因为台账不是全量，见 §4 第 5 行）。
   **而 `V0-Cycle` 本身在网格层面就是 `A8_full_L70`**：`ablation8.json` 的 96 条里该 tag 12 条、`ms/sa/hi = full/basic/hf`、`use_cycle=True`。
   所以 Task B 第 1 项「Cycle 的独立贡献」在强骨干上**必须新训练**；base 骨干上已有 12 格可复用（见 §4 第 6 行）。
4. **`DSPT-TrainDefaults` 已闭合为 `A8_ms_L70`**（`full/off/none/cycle=on`，参数 **`82114`**，与 `DSPT(max_span=70)` 默认构造逐位相等）：**12 格 val 齐、val engine-mean `11.3404`；test 只有 `s101` 四格**（`n=1`/子集，均值 `11.3867`）。
   **⚠ 但一个曾差点写错的地方**：`A0` **不是** TrainDefaults。`A0` 是 `cycle_only=True`（只喂年龄、无传感器），4 格全 `s101`、val `30.10/26.29/39.98/37.12`、**无 test**。**把 `A0` 当 TrainDefaults，基线会被写成差 3 倍。**
5. 「base」在本项目里有 **4 个不同含义**，引用前必须写三开关取值 + tag 前缀。

## 1. 三个身份的配置钉定

| 名字 | `ms_mode` | `sa_mode` | `hi_source` | **`hi_mono`** | `use_cycle` | `input_mode` | 可定位的 tag | 依据 |
|---|---|---|---|---|---|---|---|---|
| **`DSPT-V0-Cycle`**（主参考，历史批） | `full` | `basic` | `hf` | **`correct`** | `True` | `raw` | **`F_full_basic_hi1`** ≡ **`A8_full_L70`** | D039 X3 表；`RUN_INVENTORY` §5.2 #5；`ablation8.json` 12 条 `hi_mono=correct` |
| **`DSPT-V0-Cycle-FP32`**（公开基线批，**不同训练协议**） | `full` | `basic` | `hf` | **`none`** | `True` | `raw` | `physical_runs/fixed40/dspt_v0_cycle` | `d7bd9c17` 的 `FD001/s101/config.json`：`hi_mono:"none"`、`precision:"float32"`、`tf32:false`、`checkpoint_selection:"val_engine_mean_rmse"` |
| **`DSPT-TrainDefaults`**（辅助参考） | `full` | `off` | `none` | **`none`** | `True` | `raw` | **`A8_ms_L70`**（`82114` 参数，与 `DSPT(max_span=70)` 默认构造逐位相等） | 原件 §1「literal train DEFAULTS」；D039 X2 记 DEFAULTS 为 `sa_mode='off'`；管理员 2026-11-14 核实 |
| **历史 `Selected`** | `single_proj` | `basic` | `hf` | **`correct`** | `True` | `raw` | **`F_single_proj_basic_hi1`** | `SEED_ROBUSTNESS` 的 `cand` 行；factorial 线 |
| （Task B 需要，**档案内未找到**）`V0-NoCycle` | `full` | `basic` | `hf` | **`correct`** | **`False`** | `raw` | 无 | D039 X3 已定义；**`nocycle.json` 12 条全为 `NC_base_L70`（`off/off/none`），不含本行**；**「未找到」≠「未运行」** |
| （Task B 需要，已存在）`V0-NoHI` | `full` | `basic` | **`none`** | **`none`** | `True` | `raw` | **≡ `A3`**（D039 X3 明写「同配置，只跑一次、双向引用」） | D039 X3 |

> **⚠ `hi_mono` 不是装饰字段（军师 2026-11-14 终审新增）**：`train.py:137–144` 里只有在 `hi_mono == 'correct'` 时才实际施加单调损失；`'none'` 直接跳过。所以 `hi_mono` 从 `correct` 换成 `none` **会改变训练目标本身**，不是记录项。**后果**：`DSPT-V0-Cycle`（历史批，`correct`）与 `DSPT-V0-Cycle-FP32`（公开基线批，`none`）**只能称为「同架构、不同训练协议」，不能当作同一完整配置的两个别名**；两者数字之差也因此不能全部归给「选轮/精度」。
> **而 `hi_mono` 在 `A8` 网格里是跟着 `hi_source` 走的**：`hi_source=hf` 的 4 个臂全为 `correct`，`hi_source=none` 的 4 个臂全为 `none`（实核 8 个 tag 各 12 条）。**所以 A8 内部不存在「同 `hi_source` 但 `hi_mono` 不同」的对照——`hi_mono` 的主效应在这 96 格里不可识别**，这是本轮必须显式标注的不可识别项。

**三个身份共享的 27 个训练/划分字段**（`RUN_INVENTORY` §5.1 已逐字段核对）：
`span=70, epochs=40, patience=12, batch=512, lr=1e-3, weight_decay=1e-4, d_model=64, layers=2, dropout=0.2,
mode=dual, hi_mono_coef=0.1, rul_cap=125, split_seed=7, fold=0, val_frac=0.2, n_cond=6, cuts=[0.4,0.55,0.7,0.85,0.95], …`
**所以三者之间的差不是预算或划分造成的**——**但这不等于「只剩结构/输入表示」**（军师指出，本文初版写得太满，已收窄）：
**`RUN_INVENTORY` §5.1 那张表只覆盖 27 个训练/划分字段，不覆盖 checkpoint 选轮准则、数值精度和运行出处。**
而 `SEED_ROBUSTNESS` 明写它的三个细胞是**在 GPU 主机上按 `pooled` 验证 RMSE 选的轮**（该主机仍是 pre-R2 `metrics.py`），`engine-level` 值是事后从保存的预测重算的。**所以「同 27 字段」只排除了预算与划分，选轮准则这一项仍在两批 run 之间可能不同。**
**引用任何跨批差值前，必须另外核对：选轮准则、精度、`val_pred.csv` 的重算口径、run 出处（哪个根/哪台机）。**

## 2. 口径分裂：同一配置的两个 TEST 数

**这是本文最要紧的一节。**

| 配置 | 口径 | val（engine-level，4 子集均值） | test（4 子集均值） | 逐子集 test | 来源 |
|---|---|---:|---:|---|---|
| `V0-Cycle`（= `full`） | **未记录**（见下「精度是推断」） | **10.981** | **11.512** | `10.379 / 10.988 / 12.317 / 12.362` | `SEED_ROBUSTNESS` test 表 `full` 行 |
| `V0-Cycle-FP32` | **统一 FP32（明写）** | 无记录 | **11.1725** | `10.60 / 11.03 / 10.68 / 12.38` | 原件 §3/§6 主参考行 |
| `Selected`（= `cand`） | **未记录** | **10.896** | **11.134** | `10.854 / 10.960 / 10.799 / 11.922` | `SEED_ROBUSTNESS` test 表 `cand` 行 |
| `Factorial base`（= `base`） | **未记录** | **11.364** | **11.655** | `11.140 / 11.544 / 11.130 / 12.806` | `SEED_ROBUSTNESS` test 表 `base` 行 |
| `TrainDefaults`（= `A8_ms_L70`） | **未记录** | **11.3404**（12 格） | **11.3867**（**仅 `s101`，`n=1`/子集**） | `—`（`n=1`，不报逐子集稳定性） | `RESULTS_AUDIT_20261114_IDENTITIES.md` |
| `TrainDefaults-FP32` | **统一 FP32（明写）** | 无记录 | **11.23**（3 seed） | `10.07 / 11.20 / 11.16 / 12.49` | 原件 §3 辅助参考行 |

（均值均为本文按四个子集等权计算：`V0-Cycle-FP32` = `(10.60+11.03+10.68+12.38)/4 = 11.1725`；`TrainDefaults-FP32` = `11.23`。）

### 2.1 两个数之间至少四个变量，且没有单变量对照

| 变量 | `SEED_ROBUSTNESS` 的 `full`（`11.512`） | 原件的 `V0-Cycle-FP32`（`11.1725`） | 是否已被排除 | 证据强度 |
|---|---|---|---|---|
| **seed 数** | **3**（`runs per cell: base=12, cand=12, full=12` = 3 seed × 4 子集） | **3**（12 逻辑条件 = 3 seed × 4 子集） | **✅ 已排除：两侧都是 3** | 两侧均有明文 |
| **`hi_mono`** | **`correct`**（`ablation8.json` 12 条） | **`none`**（`config.json` 明文） | **❌ 未排除，且是训练目标差异** | **两侧均有明文（本日新增实核）** |
| **选轮准则** | **`pooled` 验证 RMSE**（GPU 主机 pre-R2 `metrics.py`） | **`val_engine_mean_rmse`**（`config.json` 的 `checkpoint_selection` 字段明文） | **❌ 未排除** | **两侧均有明文** |
| **聚合/汇报方式** | `engine-level` 为**事后重算**（不是选轮时用的量） | 与选轮同一口径 | **❌ 未排除** | 有明文 |
| **数值精度** | **未记录**（见下） | **FP32，禁 autocast/TF32**（原件 §1 明写） | **❌ 未排除（但也不敢认定是它）** | 仅一侧有明文 |

**「seed 数已被排除」这一条值得单说**：任务书要求把 seed 数与聚合方式并列列出，**而核下来这一对两侧都是 3 seed × 4 子集**（`SEED_ROBUSTNESS` 自报 `runs per cell = 12`；原件 §1 的主条件也是 12）。**所以 `0.34` 里不含 seed 数的影响——但它含聚合方式的影响**：`pooled` 与 `engine-level` 不是同一统计量（前者先平方后平均、后者先平均后平方，Jensen 下 `pooled ≥ engine-level`，见 D039 的 B5 更正）。**⚠ 这条「已排除」只对 `V0-Cycle` 这一对成立**：`TrainDefaults` 的 test 只有 `s101`（`n=1`/子集），**那里 seed 数是真差异**，不能套用本表。

**⚠ 「`SEED_ROBUSTNESS` 是 BF16」是推断，不是查实。** 本文初版把它写成「默认 BF16」，依据只是原件 §1 那句「不与历史 DSPT 默认 BF16 结果声称数值等价」——**那说的是「历史默认值」，没有指认 `SEED_ROBUSTNESS` 那 12 格**。我在 `paper_b/experiments` 下也没找到带 `precision/autocast/tf32/dtype` 字段的 run config。**所以精度这一列应读作「未记录」，不是「BF16」。**

**三条硬规则**：

1. **`11.512` 与 `11.1725` 不能互相替代**，也不能各自与对方的基线相减。原件的 FP32 条件是为「与 5 篇公开基线同台」而设的，**它明确不声称与历史 DSPT 数值等价**（原件 §1 末句）。
2. **两数之差 `0.34` 与 `Selected` 对 `full` 的效应（`−0.378`）是同一量级——占其 `90%`**（`0.3395 / 0.378 = 0.898`）。**但它的来源未被分离**：**未排除的有 `hi_mono`、选轮准则、聚合方式、精度四项**（seed 数已排除，见上表）。**其中 `hi_mono` 是最硬的一项——它不是记录差异，而是训练目标差异**（`train.py:137–144` 只在 `correct` 时施加单调损失）。**所以只能说「协议不同造成的位移与要检验的效应同量级」，不能说「精度把它推向某一侧」**——**没有单变量对照，任何方向性归因都是猜的**（本文初版那句「换一次精度就能推向任何一侧」已撤回）。
3. **`SEED_ROBUSTNESS` 的 val 是事后重算的 engine-level 值，而选模当时用的是 pooled**。**所以引用它的 val 时必须带这句**，否则会被读成「用 engine-level 选模」。

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
| 1 | **`TrainDefaults` 的真实 tag / 运行记录** | **已闭合 = `A8_ms_L70`**（`82114` 参数）：12 格 val 齐、engine-mean **`11.3404`**；**test 仅 `s101` 四格**（均值 `11.3867`，`n=1`/子集）。**`A0` 是 `cycle_only` 年龄臂，不是它**（`A0` val `30.10/26.29/39.98/37.12`、无 test） | 已核（管理员 `aaa4e8ba`） |
| 2 | **`V0N` 是否真零运行** | **不能闭合为「零运行」，只能闭合为「档案内未找到」**：`nocycle.json` 12 条全为 `NC_base_L70`（`off/off/none`）→ 不含 V0N；盘上无 `V0*` 目录；`runs.csv` 无 V0 标签；经理另扫 6 个网格共 216 格亦无 `full/basic/hf + use_cycle=False`。**四条链都是「档案缺失」型证据** | **按证据缺口补跑**；发射前由燕子按配置字段在 3090 侧全盘查重 |
| 3 | **`A8_full_L70` ≡ `F_full_basic_hi1`** 的去重 | 同一批 12 run 两个 tag，**任何合并表必须显式映射**，否则重复计数 | 台账维护方 |
| 4 | **`MATCHED_COMPARISON_REPORT` 的跨分支可见性** | 已在另一分支存在却被判「不存在」——**检查工具用错分支范围**（见 §5） | 已修正，见恢复件横幅 |
| 5 | **「已有 96 格消融」的确切所指** | **已闭合**：就是 `paper_b/experiments/grids/ablation8.json` 的 **96 条 = 8 臂 × 4 子集 × 3 seed**，`use_cycle` 全 `True`，`A8_full_L70` 即 `V0-Cycle`。**并且它是完整 `MS×SA×HI` 的 `2³` 设计**（8 个组合一个不缺：`off/off/none`、`full/off/none`、`off/basic/none`、`off/off/hf`、`full/basic/none`、`full/off/hf`、`off/basic/hf`、`full/basic/hf`）→ **主效应与二阶/三阶交互都可识别**，不需要再补设计点 | 已核（管理员 `a8_engine_level.csv` 96 行齐；网格 8 个 tag 各 12 条） |
| 6 | **强骨干 cycle-off 的替代证据** | 强骨干档案内无 `V0N`，但 **base 骨干有 12 格 `NC_base_L70`**：`cycle on 11.3118` vs `off 12.3523`，`Δ = +1.0405`（val，4 子集均值） | **仅适用于 base 骨干**：`cycle` 的效应在 `ms/sa/hi` 全关的骨干上测得，**不能外推到强骨干**（外推正是本轮禁止的写法）。这也正是补 12 格 `V0N` 的唯一理由 |
| 7 | **`runs.csv` 不是全量** | `128` 行；`F_*` 每族只登 `s202/s303`（`s101` 全不在），A8 只登冻结清单外 4 格。**「12 run」必须引 `paper_b/results/`，引 `runs.csv` 会得到 8** | 台账维护方 |

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
