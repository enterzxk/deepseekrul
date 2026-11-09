# Draft: §3 and §4 (boundary paper)

> Target: IEEE TIM. Working title: *Why Engine-Level Coverage Collapses in Conformal RUL Prediction: A Two-Sided Impossibility Bound and Metrological Limits*
> All numbers from `docs/tables/BOUNDARY_TABLES.md (80 cells, calibrated CQR).

---

## 3. First-Moment Limit: Center Bias Dominates, and Removing It Does Not Help

### 3.1 The marginal interval is textbook-correct

Table I establishes the premise on which everything in this section rests. Across
4 subsets x 5 folds x 4 seeds = 80 independently trained cells, split-conformal
CQR attains a marginal coverage of 0.886 / 0.885 / 0.905 / 0.896 on
FD001 / FD002 / FD003 / FD004. The nominal level is 0.90 and the finite-sample
correction of Eq. (2) is applied verbatim. **There is no implementation defect to
explain away.** The interval procedure does exactly what conformal theory promises.

### 3.2 What the same intervals do to individual engines

Decomposing that same coverage per engine yields Table I, right-hand columns.
The mean engine-level PICP is 0.872-0.891 -- unremarkable. But the 5th percentile
of the per-engine distribution falls to 0.656-0.697, and the minimum engine
across all 80 cells reaches **0.2045 (FD001), 0.2375 (FD002), 0.5146 (FD003) and
0.1331 (FD004)**.

For FD004 this means: an interval certified at 90% marginal coverage leaves
**86.7% of one engine's operating windows outside its bounds**. For a
safety-critical asset this is not a statistical curiosity; it is the exact failure
mode that a marginal guarantee was supposed to exclude.

### 3.3 Which moment drives the collapse

Table II answers the causal question. Within each cell we compute, per engine,
the coverage and the mean signed error of the interval centre, then correlate
across engines; the values reported are means over the 20 cells per subset.

- `r(PICP_i, |bias_i|)` = **-0.759 / -0.830 / -0.728 / -0.815**
- `r(PICP_i, halfwidth_i)` = **+0.339 / +0.155 / +0.100 / -0.164**

The first is strong and consistently negative in every subset; the second is weak
and changes sign. Fig. 1 shows the same relation over all 1976 engine records
(pooled correlations -0.739 / -0.822 / -0.756 / -0.740).

**Engine-level coverage collapse is a first-moment phenomenon.** It is produced
by a displaced interval centre, not by an interval that is too narrow. Any
post-hoc procedure that only touches the second moment is therefore aimed at the
wrong component.

### 3.4 The harmful-debias paradox

If the collapse is a centre problem, the natural remedy is to remove the centre
error. We test the *upper bound* of that remedy: an oracle that knows each
engine's true mean bias and subtracts it exactly. No observable predictor is
required -- this is the best any centre-correction scheme could possibly achieve.

The result (Table III) is counter-intuitive:

| subset | worst engine, original | worst engine, oracle-debiased | Delta |
|---|---:|---:|---:|
| FD001 | 0.5990 | 0.5483 | **-0.0507** |
| FD002 | 0.5564 | 0.4890 | **-0.0673** |
| FD003 | 0.6483 | 0.6502 | +0.0019 |
| FD004 | 0.5277 | 0.4642 | **-0.0635** |

**Perfect removal of the per-engine mean bias makes the worst engine worse in
three of four subsets**, by 5.1-6.7 percentage points.

The mechanism is visible in the residual decomposition. Writing the residual at
window t of engine i as `e_{i,t} = b_i + eps_{i,t}`, subtracting the oracle mean
sets the first term to zero and leaves `eps_{i,t}` untouched. When the
within-engine dispersion is heavy-tailed and non-exchangeable -- as it is here,
with `|bias_i| / halfwidth_i` having median 0.25 but 90th percentile 1.09 -- the
per-engine mean is a poor summary of where the interval should sit. Removing it
can move the centre *away* from the region where most windows actually fall.

**Corollary for practice.** A centre-correction module validated on a benign,
single-regime subset can reverse its sign on a multi-regime subset. An engineer
who debugs on FD001-like data would observe an improvement and deploy; on
FD002/FD004-like data the same module degrades coverage. This regime-dependent
sign reversal is a hazard that no marginal-coverage report can expose.

---

## 4. Second-Moment Limit: The Metrological Cost of Width

### 4.1 The frontier

If the centre cannot be trusted, the remaining lever is width. Fig. 2 plots, for
each subset, the median worst-engine PICP as a function of width inflation
`Delta W`, with the 10th-90th percentile envelope across the 20 cells per subset.

Two reference lines are drawn: the safety threshold `PICP = 0.75` and the
metrological budget `Delta W = 15%`.

**No curve enters the admissible region** (upper-left: PICP >= 0.75 at
Delta W <= 15%). At `Delta W = 15%` the median worst-engine PICP is
approximately 0.69 (FD001), 0.62 (FD002), 0.72 (FD003) and 0.58 (FD004).

### 4.2 Required inflation

Table IV reports the inflation needed to lift the worst engine to 0.75:

| subset | median | 90th pct | max |
|---|---:|---:|---:|
| FD001 | **27.5%** | 63.0% | 110% |
| FD002 | **47.5%** | 70.5% | 80% |
| FD003 | **27.5%** | 89.0% | 125% |
| FD004 | **150.0%** | 225.5% | 285% |

Even the most favourable subset requires 27.5% at the median; the 90th percentile
reaches 63%-225%. The 15% budget is exceeded by factors of 1.8 to 10.

### 4.3 Why this is a metrological failure, not a tuning inconvenience

The GUM distinction between *dispersion* and *uncorrected systematic error* is
decisive here. An uncertainty interval is licensed to describe scatter; it is not
licensed to absorb a systematic displacement. When `|bias_i|` exceeds the
half-width for a substantial fraction of engines, inflating the half-width
converts a bias problem into a dispersion number. The reported PINAW then no
longer measures the instrument's resolving power.

Concretely: at `Delta W = 150%` the FD004 interval spans a large fraction of the
observed RUL range. For a maintenance-scheduling instrument, an interval covering
a majority of the possible remaining life carries no decision value -- the
interval has been purchased at the price of the measurement itself.

### 4.4 The two-sided bound

Combining Sections 3 and 4:

- **First moment.** Removing the per-engine mean bias exactly (an oracle) does
  not raise worst-engine coverage; it lowers it on 3 of 4 subsets.
- **Second moment.** Raising the width until worst-engine coverage reaches 0.75
  costs 27.5%-150% (median), 63%-225% (90th percentile).

Neither lever alone, and (from the independence of the two mechanisms) no convex
combination of them, restores unit-level reliability within the metrological
budget. **Engine-level coverage collapse in conformal RUL prediction is not
repairable by any post-hoc calibration of the interval.**

---

## 数值核对清单（写进正文前逐条核对）

| 正文数字 | 来源 | 状态 |
|---|---|---|
| 边际 PICP 0.886/0.885/0.905/0.896 | a2_cells.csv 按子集均值 | OK |
| 引擎均值 0.872-0.891 | 同上 | OK |
| 引擎 Q0.05 0.656-0.697 | 同上 | OK |
| 最差 0.2045/0.2375/0.5146/0.1331 | a2_cells.csv min | OK |
| r_bias -0.759/-0.830/-0.728/-0.815 | 格内均值 | OK |
| r_hw +0.339/+0.155/+0.100/-0.164 | 格内均值 | OK |
| 合并 r -0.739/-0.822/-0.756/-0.740 | 图 1 标注 | OK |
| oracle Delta -0.0507/-0.0673/+0.0019/-0.0635 | a2_cells.csv | OK |
| 宽度代价 27.5/47.5/27.5/150 中位 | a2_cells.csv | OK |
| 宽度代价 p90 63/70.5/89/225.5 | 同上 | OK |
| abs(bias)/hw 中位 0.25, p90 1.09 | **单折旧算，需 80 格重算** | TODO |
| Delta W=150% 时 FD004 区间占比 | **需验证** | TODO |

## 待补

1. 在 80 格上重算 `|bias|/halfwidth` 的分布（当前 0.25/1.09 来自单折）
2. 验证 `Delta W=150%` 时 FD004 的 PINAW 绝对值
3. §3.4 的「sign reversal」需要一个显式的小表或脚注
---

## 4.5 关键更正：那 16% 的达标格是「本来就没坏」

80 格中有 13 格（16%）能在 `Delta W <= 15%` 内把最差引擎拉到 0.75。
逐一检查这 13 格后发现：

| 量 | 达标的 13 格 | 全部 80 格 |
|---|---:|---:|
| 基线最差引擎（加宽前）中位数 | **0.73** | 0.591 |
| 所需 `lambda` | **1.00 – 1.15** | 中位 1.28 – 2.50 |

**这 13 格不是被救回来的，而是本来就没有崩。** 它们的基线最差引擎已达 0.73，
宽度膨胀只做了 0–15% 的象征性推动。

各子集的代价分布（`Delta W` %）：

| 子集 | min | q25 | 中位 | q75 | max | <=15% 格数 |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 0.0 | 18.8 | **27.5** | 45.0 | 110 | 3/20 |
| FD002 | 15.0 | 33.8 | **47.5** | 66.3 | 80 | 1/20 |
| FD003 | 0.0 | 10.0 | **27.5** | 72.5 | 125 | 8/20 |
| FD004 | 5.0 | **75.0** | **150.0** | 215.0 | 285 | 1/20 |

**FD004 即便取最好的四分位也需要 75% 宽度。**

### 修正后的主张（替代 §4.4 的「不可能」表述）

> 后验宽度膨胀**只能锦上添花，不能雪中送炭**。
> 它在基线最差引擎已接近 0.75 的格上「成功」（16%），
> 而对真正失效的格，代价是中位 27.5%–150%（FD004 最好四分位 75%），远超 15% 计量预算。

这个表述比「不可能」更精确，并且**解释了文献中「某方法达标」的报告**：
那些报告很可能抽到了基线本就接近阈值的折划分。

### 由此产生的一个可检验预测

若上述机制成立，则任何声称「后验方法显著提升最差单元覆盖」的论文，
其基线最差单元覆盖应当已经接近其报告的目标值。
**这是一个可以拿去检验已发表工作的具体预言。**