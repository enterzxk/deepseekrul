# 阶段四 · 预注册（训练前冻结）

> **状态：骨架已冻结，候选相关项待 L2 裁定后填入。**
> **冻结时刻**：`2026-11-14`，分支 `paper-b-contribution-attribution`，`HEAD = 5e56c94b3e199952cde6d0387f0da2363b8229a5`。
> **冻结范围**：本文第 1–6 节（母体、协议、seed 集、判据、四项对照、三个探针）**全部不依赖候选是什么**，现在就冻结。
> 第 7 节（候选定义与数学差量）**留空待填**——若 L2 裁定「无候选」，第 7 节写「无候选」并直接进 PIVOT，**本阶段不产生任何训练**。
>
> **冻结的意义**：任何在结果产生后修改第 1–6 节的行为都构成违规。**新增内容只能进第 8 节「冻结后的追加」并标明日期与原因。**

## 1. 母体（逐字段，来自产物而非记忆）

**来源**：`paper_b/results/raw/F_full_basic_hi1/FD001/s101/config.json`，**36 个键**，其中 33 个为协议键（`tag`/`seed`/`subset` 为逐格键）。

| 键 | 值 | 键 | 值 |
|---|---|---|---|
| `model` | `dspt` | `mode` | `dual` |
| `ms_mode` | **`full`** | `sa_mode` | **`basic`** |
| `hi_source` | **`hf`** | `hi_mono` | **`correct`** |
| `hi_lambda` | `0.5` | `hi_mono_coef` | `0.1` |
| `use_cycle` | **`true`** | `input_mode` | `raw` |
| `span` | **`70`** | `rul_cap` | **`125`** |
| `epochs` | **`40`** | `patience` | **`12`** |
| `batch` | `512` | `lr` | `0.001` |
| `weight_decay` | `0.0001` | `d_model` | `64` |
| `layers` | `2` | `dropout` | `0.2` |
| `n_cond` | `6` | `split_seed` | **`7`** |
| `fold` | **`0`** | `val_frac` | **`0.2`** |
| `cuts` | `[0.4, 0.55, 0.7, 0.85, 0.95]` | `data_dir` | `data/CMAPSS` |
| `aux_dual` | `true` | `use_pad_mask` | `true` |
| `global_hi` | `false` | `instnorm` | `false` |
| `legacy_ema` | `false` | `legacy_ms_pad` | `false` |
| `legacy_rul` | `false` | | |

**逐格键**：`tag = F_full_basic_hi1`、`seed ∈ {101,202,303}`、`subset ∈ {FD001,FD002,FD003,FD004}`。

**⚠ 一条不能省的身份说明**：`hi_mono = correct` 是**训练目标的一部分**（`train.py:137–144` 只在 `correct` 时施加单调损失）。
它与公开基线批的 `DSPT-V0-Cycle-FP32`（`hi_mono = none`）**同架构、不同训练协议**，**不是同一完整配置的别名**（见 `DSPT_IDENTITY_PINNING_20261114.md` §2.1）。

## 2. 执行级三项（**不在 `config.json` 里，必须写进 run metadata**）

母体 `config.json` 的 36 个键里**没有** `precision`、`autocast`、`cudnn_sdp`、`checkpoint_selection`——这四项是执行级，不能靠读配置证明。本阶段冻结如下，并要求每次 run 把它们写进 metadata：

| 项 | 冻结值 | 来源 | 为什么必须单列 |
|---|---|---|---|
| 数值精度 | **`bf16-autocast`** | D045（3090 `~/rulb` 现存路径） | 配置里无此字段；历史 precision 仍为「未记录」，**当前源哈希不冒充历史执行指纹** |
| `cudnn_sdp` | **`off`** | 同上 | 同上 |
| **选轮准则** | **`val_pooled_rmse`** | D045（legacy pooled 路径） | 与汇报口径**不同**，见 §3 |
| **汇报口径** | **`val_engine_mean_rmse`**，**从 `val_pred.csv` 重算** | 本轮硬规则 | `metrics.json::val_engine_rmse` 在 265 个 run 里装的是 `pooled`（engine-mean 出现 0 次），**字段名不可信** |

**⚠ 选轮用 `pooled`、汇报用 `engine-level`，这是两条不同的量，不是同一个数的两种写法。**
理由：`pooled` 先平方后平均、`engine-level` 先平均后平方，Jensen 下 `pooled ≥ engine-level`（D039 的 B5 更正）。
**任何跨这两个口径的比较都是跨口径相减，禁止。**

## 3. 数据与划分

| 项 | 冻结值 |
|---|---|
| 子集 | `FD001 / FD002 / FD003 / FD004` |
| 划分 | 按 `engine_id`，`split_seed=7`、`fold=0`、`val_frac=0.2` |
| 验证机数 | `FD001 20 / FD002 52 / FD003 20 / FD004 50` |
| 验证截断点 | 每机寿命的 `40/55/70/85/95%`（`cuts` 已冻结） |
| 官方 TEST | **本阶段不使用**；只在决策冻结后跑一次作描述 |

## 4. seed 集与样本量

**冻结 seed 集：`{101, 202, 303}`，每个 `(臂, 子集)` 各 3 颗，不增不减。**

**主判据的样本结构**：同 seed 配对，`n = 3`，`df = n − 1 = 2`。
**⚠ `df = n − 1`，不是 `n`。** 用错会把半宽从 `0.3724` 量级放大到 `0.64` 量级，**方向永远让结果更难通过**。

**预算**：`epochs = 40`、`patience = 12`。**`CosineAnnealingLR(T_max = epochs)`——`T_max=40` 与 `T_max=80` 是两条不同的学习率曲线**，所以本阶段任何结论都只对 `40ep` 协议成立，**不得与 80ep/160ep 的数字混表**。

## 5. 判据（**先冻结，再看结果**）

**主判据：逐子集同 seed 配对差，`n=3`、`df=2`、双侧 95%。**

```
对每个子集 s：
  d_i = metric(候选, s, seed_i) − metric(母体, s, seed_i)      i = 1,2,3
  Δ   = mean(d_i)
  half = t(0.975, df=2) × sd(d_i) / √3 = 4.3027 × sd / √3
判定：
  Δ < −half  且 3/3 同号  → better（候选更好，越低越好）
  Δ > +half  且 3/3 同号  → worse
  否则                     → undecidable
```

**冻结的四个细节**（每一条都是今晚付过代价的）：

1. **`metric` = `val_engine_mean_rmse`，从 `val_pred.csv` 重算**，不读 `metrics.json` 的字段。
2. **`t(0.975, df=2) = 4.3027`**，用精确分位数，**不用 `1.96`**（`n=3` 下差 2.2 倍）。
3. **`half` 由配对差自己的 `sd` 算出**，不借用任何现成门槛值（`single_vs_mean`、`paired` 这些注册表值是给别的样本结构用的，**在 `n=3` 配对上不适用**）。
4. **`3/3 同号` 是独立于 `half` 的一道条件**：`n=3` 的 `sd` 本身很不稳，`half` 可能被一颗离群 seed 撑大或压小，**同号检查是防这个的**。

**GO / PIVOT / STOP 的冻结定义**：

| 结论 | 冻结条件 |
|---|---|
| **GO** | 至少 **1 个子集** `better`，且该子集在四项对照中**全部**优于对照；且机制探针三条全过 |
| **PIVOT** | 候选与对照**无差异**（全 `undecidable`），但 Task C 指出了一个**新的、可证伪的**研究问题 |
| **STOP** | 候选与对照无差异，且 Task C 也没有给出可证伪的新问题 → **保留 DSPT 作强基线，停止堆模块** |

**⚠ 不允许「部分子集 better、部分 undecidable」就宣布 GO**——那必须写成「仅在子集 X 上 `better`」，并说明其余子集为何不成立。

## 6. 四项对照与三个机制探针（冻结）

### 6.1 四项对照（来自 L2 裁定，顺序即优先级）

| # | 对照 | 目的 | 必须相同的量 |
|---|---|---|---|
| **C1** | **当前工况收缩基线**（`data.py:146 _baseline_ref` 原样） | 证明候选**不是**它的等价改写 | 同母体、同预算、同 seed |
| **C2** | **同一前缀的普通中心化**，或 **DAIN 式平移子层** | 证明增量不是「随便减个均值」 | 同上，且输入前缀完全相同 |
| **C3** | **移植到同一母体的 M2** | 排除「M2 早就能做到」 | 同上；**历史无 cycle 的 M2 不是匹配对照** |
| **C4** | **等参数量简单替代** | 排除「涨参数就涨点」 | 参数量差 `≤ 1%` |

**⚠ C2 的前缀必须与候选拿到的前缀逐点相同**——否则是「多给了信息」，不是「换了算法」。

### 6.2 三个机制探针（缺一条则该机制不可主张）

| # | 探针 | 通过条件 |
|---|---|---|
| **P1** | **未来记录改动不影响当前输出** | 把 `t` 之后的记录任意替换，`t` 时刻输出**逐位不变** |
| **P2** | **本机前缀换成工况/长度匹配的其他机前缀** | 输出**发生变化**（否则机制没在用个体前缀） |
| **P3** | **观测恒定偏移干预** | 加常数偏移后输出按预期变化；**若输出不变，说明候选对该通道不敏感** |

**⚠ P2 的边界**：替换前缀会制造分布变化，**所以 P2 只作压力检查，不能单独当自然因果证据**（L2 原文）。自然 validation 改善、个体性检查、泛化误差三者必须**共同**报告。

## 7. 候选定义与数学差量 —— **待填（L2 裁定后）**

**本节在 L2 裁定前保持空白。** 填入时必须包含：

1. **候选相对 `_baseline_ref` 改了哪个估计量或哪个假设**（不是「加了个模块」）；
2. **数学差量**：写出两个估计量的式子，指出差异项；
3. **可证伪判据**：什么观测会让候选**当场失效**；
4. **与 C1–C4 的逐项预期**：候选应当优于哪几个、与哪几个无差异。

**若 L2 裁定「无候选」**：本节写「**无候选**」+ 裁定日期 + 裁定依据，**本阶段不产生任何训练**，直接进 PIVOT。

## 8. 冻结后的追加（只增不改）

| 日期 | 追加内容 | 原因 |
|---|---|---|
| — | — | — |

## 9. 复核方式

```powershell
cd C:\Users\20562\Desktop\deepseek-rul
# 1. 母体 36 键逐位
python -c "import json,io;c=json.load(io.open('paper_b/results/raw/F_full_basic_hi1/FD001/s101/config.json',encoding='utf-8'));print(len(c));[print(k,c[k]) for k in sorted(c)]"
# 2. 执行级四项不在 config 里（应为 0 命中）
python -c "import json,io;c=json.load(io.open('paper_b/results/raw/F_full_basic_hi1/FD001/s101/config.json',encoding='utf-8'));print([k for k in c if any(x in k for x in ('precision','autocast','cudnn','selection'))])"
# 3. 冻结时刻
git rev-parse HEAD
```
