# 传感器侧 + 结构化 HI 首轮（SMHI-RF / SHI-LTAG）结果

> 日期：2026-11-18。状态：**pilot，mixed / 未过门**。test 未用于本轮；LTAG 的冻结 test
> 结果仍是唯一 test 级证据。代码：`paper_b/modules/smhi.py`、`merge_smhi.py`、
> `learned_gate_shi.py`；文献 gap：`docs/paper_b_modules/SENSOR_HI_GAP_20261118.md`。

## 1. 方法

**SMHI-RF**：冻结 B0 的 `sensor_attn` 输出逐传感器 token `h_tj`（B×L×N×D）。
逐传感器 `δ_tj=softplus(MLP(h_tj))≥0`，`H_tj=cumsum δ_tj`（窗口内单调）；
可靠性 `r_tj=softmax_j(MLP([h_tj, mean_j h_tj]))`；融合
`H_t=Σ r_tj H_tj / Σ r_tj`；零初始化残差头
`pred = reg_head(z) + res_head([z, H_last])`。只训 HI bank + 残差头（8,419 参数），
B0 冻结；training engines 训练，20% training engines 做 early stopping。

**SHI-LTAG**：把 SMHI 的逐 cut 预测 `p_smhi` 及其与 `p_k1/p_med` 的差作为
LTAG 门控的额外特征，其余协议不变（5-fold GroupKFold，training engines only）。

## 2. SMHI-RF 结果（B0 vs SMHI，validation engine-mean RMSE）

| 子集 | seed 101 | seed 202 | seed 303 | 均值 | 符号 | 95% CI |
|---|---:|---:|---:|---:|---|---|
| FD001 | +0.076 | +0.118 | +0.177 | **+0.124** | 3/3 | [−0.003,+0.251] |
| FD002 | **−0.451** | +0.079 | **−0.340** | **−0.238** | 1/3 | [−0.932,+0.457] |
| FD003 | **+1.235** | +0.233 | +0.016 | **+0.495** | 3/3 | [−1.121,+2.111] |
| FD004 | +0.112 | +0.052 | +0.240 | **+0.135** | 3/3 | [−0.104,+0.374] |

- FD001/FD004 小幅正、FD003 均值大但由 s101 单点驱动（sd 0.65），**FD002 明确受伤**。
- 多数格 `best_ep` 在 1–10（FD003 s101 例外 164），说明 early-stopping 的 20% 训练机
  holdout 太 noisy，HI bank 很快过拟合。
- 结论：**SMHI-RF 不是稳健模块**；不能写“四子集改善”。

## 3. SHI-LTAG 结果（validation，pooled across seeds）

| 子集 | LTAG（无 SMHI 特征） | SHI-LTAG（加 SMHI 预测） | 差 |
|---|---:|---:|---:|
| FD001 | +0.764 | **+0.769** | +0.005 |
| FD002 | +0.323 | **+0.240** | −0.083 |
| FD003 | +0.385 | **+0.422** | +0.037 |
| FD004 | +0.055 | **+0.046** | −0.009 |

- 加 SMHI 预测**没有改善 LTAG**；FD002 还变差。
- 原因（诊断）：SMHI 是在**全部 training engines**上训练的，喂给 gate 的 training
  特征是其 **in-sample** 预测，gate 会高估 SMHI 的可靠性；validation 上 SMHI 本身
  在 FD002 变差，gate 无法从 in-sample 特征学到“FD002 不要用”。
- 这不是“传感器+HI 没价值”的最终结论，而是**当前 stacking 设计有信息泄漏式乐观**。

## 4. 与最新文献的关系

最新传感器侧工作（`SENSOR_HI_GAP_20261118.md`）：
- UCD-OCMN 2026：物理约束 Bayesian 动态图 + 传感器耦合不确定性 + Kalman 融合；
- EAAI 2025：GAF + CNN-Transformer 构造单一 CHI + Wiener；
- 多指标动态 HI 2026（海底）、熵权传感器选择 2026、退化感知掩码增强 2026；
- ADFG / 图注意力 + 物理方程等特征级融合。

我们的窄差异是“**逐传感器单调 HI + 动态可靠性 + 结构化融合**”，不是“首次 HI/多传感器融合”。
但本轮 pilot 没有把差异做成稳健收益；需要下一轮修正。

## 5. 下一步（按优先级）

1. **Cross-fitted SMHI（OOF）**：5-fold by engine 训练 HI bank，用 OOF `p_smhi` 作为
   gate 的训练特征；validation 用 5 个 fold 的平均。这样 gate 看到的是诚实的 SMHI，
   才可能学到“FD002 不用、FD001/FD003/FD004 用”。这是当前最直接的修正。
2. **用 H 特征而不是 SMHI 预测**：把 `H_fused`、可靠性熵、top-sensor 作为 gate 特征；
   HI 状态可能比 RUL 预测更稳。
3. **条件鲁棒可靠性**：FD002 的伤害可能来自 6 工况下的传感器可靠性漂移；用
   condition-matched 的可靠性（不是 exact match，而是软权重）或 sensor-coupling
   consistency 替代学习可靠性。
4. 若 1–3 仍不能使 SMHI 在至少 FD002/FD004 上稳定为正，**放弃 sensor+HI 作为主模块**，
   保留 LTAG；把传感器侧写成“为什么在 C-MAPSS 上传感器级 HI 不额外增益”的负结果。

## 6. 判定门（下一轮，预注册）

- 主：cross-fitted SHI-LTAG 在 FD001/FD002/FD003/FD004 的 validation engine-mean RMSE
  相对 LTAG 的平均增益 ≥ 0.10，且 ≥3/4 子集非负；
- 关键：FD002 不得比 LTAG 差超过 0.05；FD003/FD004 至少一个 ≥ +0.10；
- 消融：OOF vs in-sample SMHI 特征；H 特征 vs SMHI 预测；可靠性 vs 均匀；
- test **不再打开**用于本轮变体；test 级主张仍只属于已冻结的 LTAG。
