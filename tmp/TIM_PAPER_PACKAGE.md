# TIM 论文材料包（v3，2026-11-05 21:20）

> 数字以 `RESULTS.csv` / `ABLATIONS.csv` / `observability.json` /
> `MEASUREMENT_CONDITIONS.csv` / `REFERENCE_LOCK.json` 为准，由
> `assemble_delivery.sh` 重新生成，不手写。

## 1. 最终方法（诚实陈述）

**本项目没有产出优于既有方法的新预测机制。** 可交付的"方法"是一个冻结的测量协议
加一套测量学分析：

- 口径 C：5 折发动机分组外层验证，80 轮、全部训练窗口、固定末轮、无轮次选择，
  主干 42,306 参数，RUL 上限 125。
- 四指标：RMSE、MAE、NASA score（逐窗口均值，另给求和）、Bias；发动机等权；
  发动机聚类配对 bootstrap 5000 次。
- 参考臂按预先规则锁定（seed 101、全窗口 pooled RMSE 最低者）。

## 2. 测量链条

传感器观测（14 路 × 50 循环窗口）→ 工况处理（训练折内 6 簇 KMeans + 簇内 Z-score
+ 个体早期参考态）→ 退化信息提取（PE 主干；可选寿命先验注入 / 时间基准替换 /
目标尺度替换）→ 寿命估计（125 截断）→ 误差与适用条件（四指标、发动机聚类区间、
按寿命四分位/观测年龄/工况支持度分层、测量条件扰动）。

变量与单位：RUL 与 μ_T 为循环；c/μ_T 无量纲；参考值为仿真器给出的 T − t（cap 125）。
RMSE 是估计器性能指标，不是仪器测量不确定度。

## 3. 核心证据

### 3.1 可观测性（observability.json）
corr(μ_T, T)：FD002 0.26@50 → 0.75@100 → 0.92@150 → 0.94@190；MAE 35.5 → 22.4 →
11.1 → 9.9。FD004 0.53 → 0.74 → 0.88 → 0.87。

### 3.2 三种使用方式（ABLATIONS.csv / RESULTS.csv）
- 加性注入（FULL）：FD002 13.054 vs M0 13.505（−3.3%），四指标同时最优。
- 时间基准（NORM）：14.513（4 折）vs M0 13.495 → 显著更差。
- 目标尺度（PHASE）：+3.909 [+2.383,+5.421]（2 折）→ 显著更差。
- 差分观测（DIFF）：−0.038 [−0.534,+0.472] → 中性。
- 发动机均衡权重（EQBAL）：−0.092 [−0.873,+0.704] → 中性。
- 传感器图注意力（GRAPH，+40% 参数）：−0.019（1 折）→ 中性。

### 3.3 可辨识性（IDENTIFIABILITY.md）
六类可观测量与"个体读数是否更优"的相关均 |ρ|<0.08；各类门控 RMSE 13.03–13.62，
均劣于固定 0.5 融合 12.89；逐窗口 oracle 10.97（15–16% 空间不可达）。
灵敏度系数与 |误差| 的相关 0.37–0.39，与"相对优劣"的相关 ≈0.02。

### 3.4 协议依赖（ENDPOINT_DEPENDENCE.md）
FD002 三端点三种最优：全窗口 FULL 13.054 / 截断匹配 M0 11.662 / 官方测试 OPS 11.842。
LOFO 融合在外层验证 12.657（−3.0%），官方测试 11.967（劣于 OPS 11.842）。
训练预算改变符号：NORM 轻量 −4.7%，完整 +7.7%。

### 3.5 测量条件（MEASUREMENT_CONDITIONS.csv）
FD002 fold 2，M0 vs FULL：clean 13.462/12.751；MC1（20% 缺测）15.318/14.194；
MC2（3 通道 +1σ）32.561/30.506。FULL 在三种条件下都更好，缺测下相对优势扩大
（−5.3% → −7.3%）；两者在 1σ 偏移下都崩溃。

### 3.6 误差相关（RESULTS_NARRATIVE.md §6）
窗口级 corr(M0,OPS)=0.835、corr(FULL,M0)=0.826、corr(FULL,OPS)=0.909；
发动机级 0.919/0.891/0.946。这解释了融合增益上限约 2.8%。

## 4. 英文题目与摘要

**Title**: *Observability-Limited Individualisation for Remaining-Useful-Life
Measurement of Aero-Engines under Varying Operating Conditions*

**Abstract**（见 `PAPER_DRAFT.md`，含 MC 结果；数字定稿时更新）。

## 5. 论文结构

I Introduction；II Measurement problem and related work（N1–N4）；
III Measurement model；IV Three roles of the lifetime estimate；
V Identifiability of adaptive correction；VI Measurement conditions；
VII Results；VIII Discussion and limitations；IX Conclusion。

## 6. 相关工作

N1 Qu et al., TIM 2026, DOI 10.1109/TIM.2026.3742092（条件感知时空网络）；
N2 Kapoor et al., IEEE Access 2026, DOI 10.1109/ACCESS.2026.3663754（phase-aware
normalization，**公式待全文核实**）；
N3 Piao et al., TIM 2025, DOI 10.1109/TIM.2024.3522678（CRULP 保形区间）；
N4 Gupta, engrXiv 2026, DOI 10.31224/8145（工况归一化协议审计，预印本）。

## 7. 限制（必须随文披露）

1. 仅 C-MAPSS 仿真；无真实台架。TIM 要求对"仅仿真"给出充分理由。
2. 官方测试已被历史使用两次；不是盲测。
3. N2 公式未核实。
4. 区间为描述性发动机聚类 bootstrap，不是覆盖保证；RMSE ≠ 仪器不确定度。
5. **初始化噪声**：运行器在 `set_seed` 之前构建模型，单次运行不可精确复现；
   FD002 fold 2 的 M0 两次运行相差约 3%，与效应量同量级。主结论只用 3 种子均值
   或远大于噪声的效应。详见 `REPRODUCIBILITY.md`。
6. FD004/FD001/FD003 的种子数按算力推进，缺口在 FINAL_DECISION 标注。

## 8. 可以主张 / 不可以主张

可以：可观测性曲线；三种使用方式的受控比较；自适应修正不可辨识；灵敏度系数作为
不确定度代理；端点/预算依赖；FULL 在缺测条件下相对优势扩大。

不可以：新 SOTA 预测器；"测量不确定度 = RMSE"；盲测验证；"个体化先验普遍有效"
（官方测试 FD002 上 FULL 12.432 劣于 OPS 11.842）；把固定融合的 3.0% 当作机制贡献；
把单种子结果写成三种子结论。

## 9. 复算

```bash
PY=/home/user/miniconda3/envs/pmv/bin/python
cd /home/user/rulb/tim_20261105/full_20261105
bash assemble_delivery.sh
```
