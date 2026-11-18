# 因果时序聚合（CTA）模块：首轮验证记录

> 状态：**进行中（pilot）**。日期 2026-11-18。作者：DSH 主控线。
> 数据：3090 上已冻结的 Stage D B0 checkpoints；**只读 validation**，未打开官方 test。
> 代码：`paper_b/modules/temporal_agg.py`（推理脚本，不改冻结训练代码）。

## 1. 问题与动机

Stage D 的 B0 在每个 cutoff 只用一个窗口预测 RUL。相邻窗口的预测会抖动；而 RUL 是
轨迹属性，同一个发动机在 `c, c-Δ, ...` 的预测应当满足

```
RUL(c) = RUL(c-Δ) - Δ          （未封顶区域）
```

如果模型在相邻窗口上的预测有独立噪声，那么把 K 个**年龄对齐**后的预测聚合起来应当
降方差；如果模型存在随工况/封顶变化的系统偏差，聚合反而会引入偏差。这两种情况在
C-MAPSS 的不同子集上可能同时存在，因此**先测现象，再决定要不要做门控**。

## 2. 方法（推理期，无训练）

对每个 validation cut `c`：

1. 取 K 个因果窗口，结束于 `c, c-Δ, ..., c-(K-1)Δ`（只用 `c` 及以前的数据）；
2. 模型对每个窗口预测 `p_t`；
3. 年龄对齐：`a_t = p_t - (c - (t+1))`；
4. 聚合 `a_t`：mean / median / trimmed mean / recency-weighted（0.7^j）/ 线性拟合；
5. 预测截断到 `[0,125]`，按主指标 engine-mean RMSE 评价。

**不使用任何标签做聚合**；标签只用于评价。`K=1` 与 checkpoint 的 `metrics.json` 子集均值一致
（最大差 `4.9e-5`）；但脚本的 per-cut `parity` 字段因 `_gate` 根没有 `preds.csv` 而为 0，
审计已将其标记为**证据缺口**（`AUDIT_CATA_20261118.md` Check 1/6）。

## 3. 首轮结果（B0 checkpoints）

固定配置：`Δ=5`，`K=3`/`K=5`，median；数据为 `_gate`（FD001 6 seed 80ep、FD004 3 seed
40ep）、`d`（FD002 3 seed 40ep）、`d80`（FD003 3 seed 80ep）。

### 3.1 全窗口聚合（不做门控）

| 子集 | n seed | K=1 均值 | K=3 median 均值 | K=5 median 均值 | K=3 逐 seed Δ(K1−K3) |
|---|---:|---:|---:|---:|---|
| FD001 | 6 | 11.696 | 11.163 | 11.443 | +0.963/+0.550/+0.490/+0.556/+0.317/+0.322 |
| FD002 | 3 | 11.440 | 11.648 | 12.349 | −0.542/+0.055/−0.138 |
| FD003 | 3 | 12.599 | 12.818 | 13.830 | −0.481/−0.286/+0.109 |
| FD004 | 3 | 13.480 | 14.345 | 15.196 | −0.894/−1.100/−0.599 |

**结论：全窗口聚合不是普适模块。** FD001 上 K=3 median 六个 seed **全部为正**（均值
+0.533，sd 0.232，t≈5.6），但 FD003/FD004 明确变差，FD002 方向混合。

### 3.2 机制诊断

- **工况漂移**：FD004 有 6 个工况，旧窗口的预测可能来自不同工况；按当前工况过滤
  （exact match）能把 FD004 的伤害从 −1.72 降到 +0.11（K=3 recency），但 FD001 也会
  被 exact match 伤害（KMeans 标签在单工况子集上仍有噪声）。
- **封顶删失**：FD004 约 20% 的 validation 窗口在 cap=125；旧窗口的 `p_t` 若已在 125，
  减去 Δ 会把预测拉低到 125 以下，造成系统低估。`cap=clamp` 能把 FD004 的伤害
  进一步降到 +0.06（K=3 recency），但会伤害 FD001（FD001 也有 7% cap 窗口）。
- **相对离散度门控（RGTA）**：`stat = MAD(a_t)/|median(a_t)|`；`stat <= τ` 用 K 窗口 median，否则用最新窗口。
  - FD001 s101：τ=0.05，K=5 → 10.233；但 **K=5 的 6-seed 均值 +0.516，CI [−0.108,+1.139]，不过 G-CATA-1**。
  - FD004 s101：τ=0.05，K=3 → 13.918（K=1 13.619，**变差 0.298**）；门控实际聚合 **66.8%**，不是“拒绝所有聚合”。
    “拒绝所有聚合”是**绝对 MAD 门控**（`gm=mad`，τ 以 cycle 为单位）的行为，不是 RGTA 的行为；旧文已按审计 F1 更正。
  - **随机门控对照（同 rate，15 格）**：FD001 K=3 real 10.778 vs random 10.405（random 更好）；FD004 K=3 real 13.918 vs random 14.157（real 更好），但 FD004 AUC=0.502、FD001 AUC=0.447 → 离散度统计量**不预测**“聚合是否有益”。
  - **结论：RGTA 作为“选择器”不成立**；按预注册 §9 退为“年龄对齐 + 固定聚合率”，门控只作实现细节。详见审计报告。

### 3.3 当前候选（审计后修订）

**CATA（Causal Age-aligned Temporal Aggregation）**：只保留“年龄对齐 + median 聚合”，
**不再把相对离散度门控当作独立贡献**。主配置建议 `Δ=5, K=3, median`：

- FD001 6/6 seed，K=3 无门控 +0.533（sd 0.232，t≈5.6）；τ=0.05 门控 +0.615
  （CI [+0.376,+0.853]），但门控未过同 rate 随机对照（G-CATA-4 FAIL）。
- K=5 不通过 G-CATA-1（5/6 正，CI 跨 0），不得作为主配置。
- 方法定位：**条件性部署模块 / 何时该聚合的经验刻画**，不是“通用 RUL 模块”。
  对多工况子集，naive 聚合有害；门控统计量无法可靠识别“何时有益”。

## 4. 待完成（审计后）

1. ~~全 seed × 全子集~~ 已完成（15 格，FD001 6 / 其余 3）。
2. **τ 标定**：在 training engines 的 holdout 上选 τ（未做；当前 τ=0.05 是开发期选择）。
3. **对照**：未对齐已完成（FD001 s101 K=3 12.17 vs 11.56）；随机门控已完成（15 格）；
   跨发动机 placebo、oracle gate 已完成分析（AUC 0.447，oracle 增益占比 36.7%）。
4. **learned aggregator**：agent 实现中；若不能击败 random/median，不保留。
5. **审计整改**：`AUDIT_CATA_20261118.md` 的 F1/F2/F3/F5 已在本文件修订；仍需补 14 格
   `gate_rate`、全 seed 多次随机抽样、training-holdout τ 标定。
6. **test 冻结**：未打开；等 L2 可用与冻结清单。

## 5. 证据路径

- 脚本：`paper_b/modules/temporal_agg.py`、`paper_b/modules/cata_preds.py`
- 结果：3090 `~/sprint24/experiments/hc/temporal_agg/`（base + `gate/` + `gate2/` + `preds/`）
- 本地副本：`tmp/cata_results/`（gate/base/preds）
- 审计：`docs/paper_b_modules/AUDIT_CATA_20261118.md`
- 新颖性：`docs/paper_b_modules/NOVELTY_CATA.md`
- 预注册草案：`docs/paper_b_modules/CATA_PREREG.md`
- 首轮 parity：K=1 与 checkpoint `metrics.json` 的最大差 `4.9e-5`；per-cut parity 未执行
- **审计后补记（2026-11-18 17:2x）**：`preds.csv` 的 `cut` 列是 **0..4 的索引**，不是 cycle；
  脚本已改为同时支持 cycle 与索引。在 `d`/`d80` 根上重跑：
  FD002-s101 `parity_n=260`、`max|diff|=3.1e-3`；FD003-s101 `parity_n=100`、`max|diff|=4.4e-3`。
  这是 bf16 重跑级差异，不是逐位一致；K=1 子集均值与 `metrics.json` 仍为 `4.9e-5`。
  审计的 “parity 门从未执行” 已修复，但不能写成 “逐位复现”。
- 相关历史：`docs/paper_b_edc/sprint_24h/FINAL_24H_DECISION.md`（模块筛选全部未过门槛）、
  `docs/paper_b_hc/ANALYSIS_RULE_GATE_A.md`（口径与功效纪律）
