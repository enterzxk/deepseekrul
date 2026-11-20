# LTAG v3 决策包（2026-11-21）

> 目的：在 Gate 1 FAIL、RHSE-TSH 删除之后，给出论文还能怎么写、需要什么决策。
> **test 封存；未做任何 test 推理。**

## 1. 已确定的负结果

| 项 | 结果 |
|---|---|
| Gate 1 | **FAIL**：LTAG 被 K1-Calib 4/4、Linear-Stack 4/4 击败 |
| G1.3 | PASS：Ens3-LTAG 4/4 胜 Ens3-K1；但仍不如 LS-ctx |
| RHSE-TSH | **删除**：公平 R5b/R5m 在 bias/drift 追平/超过 R3；只有 stuck 通过 |
| LTAG 机制 | 不成立：多窗口特征去掉后差距 ≤0.17；context 才是主要增益 |
| 原始 B0 LTAG test +0.398 | 不因 `cycle_frac` 泄漏失效（test 特征可观测），但不是最优聚合器 |

## 2. 候选方法：LS-ctx

`pred = Ridge([p_k1, p_med, regime_dist, cap, cut_norm, hist_norm])`，训练在
`cata_gate/train` 训练发动机上，alpha=1.0。

### 2.1 Validation engine-mean

| subset | K1 | LTAG | K1-Calib | **LS-ctx** |
|---|---:|---:|---:|---:|
| FD001 | 12.094 | 11.334 | 10.931 | **10.150** |
| FD002 | 11.527 | 11.210 | 10.970 | **10.604** |
| FD003 | 12.793 | 12.372 | 12.129 | **12.033** |
| FD004 | 13.608 | 13.522 | 12.839 | **13.080** |

### 2.2 LS-ctx vs LTAG（paired engine bootstrap）

| subset | Δ (LS−LTAG) | 95% CI | 显著？ |
|---|---:|---|---|
| FD001 | −1.184 | [−2.488, −0.057] | 是 |
| FD002 | −0.605 | [−1.066, −0.176] | 是 |
| FD003 | −0.339 | [−0.962, +0.238] | 否 |
| FD004 | −0.441 | [−1.057, +0.122] | 边际 |

### 2.3 机制代价

- 增益集中在 90–100% 大误差尾部（Δ|e| −3.3 到 −13.0）；小误差变差 +0.47 到 +1.14。
- FD001 bias 从 +0.88 推到 −2.21；MAE 改善，但不是无代价。
- 这是 **RMSE 导向的尾部后校准**，机制不新。

### 2.4 逐发动机胜率（更关键）

| subset | n engines | LS < LTAG | LS < K1 | mean Δ | median Δ |
|---|---:|---:|---:|---:|---:|
| FD001 | 20 | **10** | 11 | −1.184 | **+0.104** |
| FD002 | 52 | 28 | 33 | −0.605 | −0.139 |
| FD003 | 20 | 10 | 13 | −0.339 | −0.058 |
| FD004 | 50 | 28 | 26 | −0.441 | −0.229 |

- LS-ctx 只在 **50–56%** 的发动机上优于 LTAG；FD001 的中位 Δ 是 **+0.104（更差）**。
- 均值改善由少数高误差发动机驱动，**不是广泛提升**；这与“尾部校准”的机制解释一致。
- 任何“LS-ctx 更好”的主张都必须同时报告胜率/中位数，不能只报 mean。

### 2.5 分 cut：LS 的优势在早期，不在 test 附近

| subset | LS−LTAG @0.4 | @0.55 | @0.70 | @0.85 | @0.95 |
|---|---:|---:|---:|---:|---:|
| FD001 | −4.170 | −0.481 | −0.302 | +0.301 | +0.267 |
| FD002 | −1.548 | −0.563 | −0.708 | −0.199 | **+0.592** |
| FD003 | −1.239 | −0.646 | −0.120 | +0.213 | −0.276 |
| FD004 | −0.904 | −1.097 | −0.163 | −0.450 | **+0.851** |

- 在 0.95（最接近 end-of-life test）：FD002/FD004 **LTAG 最好**；FD001 K1-Calib 最好；
  FD003 LS 最好。
- validation 均值被早期 cut 主导；test 是单窗口 end-of-life，LTAG 的 +0.39 与 late-cut 一致。
- **选项 A 因此更弱**；选项 B 有分 cut 证据支撑。

## 3. 三个选项

### 选项 A：以 LS-ctx 为主方法，冻结后一次性 test

- 论文定位：“推理期尾部后校准”，强调 RMSE/MAE/bias trade-off 与 context 条件化。
- 需要：冻结 `FREEZE_LTAG_V2.json`，一次性 test（Phase 5），5 张表。
- 风险：创新性弱；审稿人会说“linear stacking / calibration”，难以支撑 IEEE TIM 方法贡献。

### 选项 B：以 LTAG test +0.398 为主，LS-ctx 作对照

- 论文定位：“年龄对齐多窗口 + 学习门控”，但必须承认 Gate 1 显示 LS/K1-Calib 更强。
- 风险：主方法被自己的对照击败；审稿人会要求解释为何仍主张 LTAG。
- 更诚实的写法：LTAG 是“有效的推理期门控”，但不是最优；贡献降级为经验发现。

### 选项 C：停止新实验，写负结果/边界报告

- 论文/报告定位：C-MAPSS 上多窗口聚合的系统性对照；LTAG 不是最优，RHSE 是稻草人。
- 风险：不适合 IEEE TIM 方法论文；可转短文/负结果报告。

## 4. 建议

- **选项 A 的证据比表面弱**：LS-ctx 只在 50–56% 发动机上赢，FD001 中位 Δ 为正；
  分 cut 后优势集中在 0.4–0.55，在 0.95 上 FD002/FD004 反被 LTAG 击败。
  它是早期 cut 的尾部校准，不是 end-of-life 主方法。若选 A，必须把胜率/中位数/分 cut 放主表。
- **选项 B 更诚实且更有解释力**：LTAG test +0.39 是 end-of-life 结果；分 cut 表解释
  “validation 均值 LS 更好”的矛盾。贡献定位为“end-of-life 推理期门控 + cut-dependent 证据”。
- **选项 C 可接受**：若期刊要求方法创新，负结果/边界报告更合适。
- **不启动 Phase 2/5**，除非 L2/用户明确选择 A 或 B。
- L2 状态：`mcp__codex__codex` 返回 `[WinError 2]`，Codex 不可用；本决策标 **PENDING-L2**。
- 磁盘仍 98%；test 封存。

## 5. 证据

- `docs/LTAGV2_P0_RECONCILE.md`
- `docs/LTAGV2_P1_BASELINES.md`
- `docs/LTAGV2_P3_MECHANISM.md`
- `docs/LTAGV2_P4_RHSE_DECISION.md`
- `results/ltagv2_p1_FD00*.json`、`results/ltagv2_p1b_FD00*.json`、`results/ltagv2_p3_FD00*.json`
