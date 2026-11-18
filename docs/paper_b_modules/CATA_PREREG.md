# CATA / RGTA 预注册草案（未冻结）

> 状态：**DRAFT / 未冻结**。test 仍封存。冻结前不得打开官方 test，也不得根据 test 改配置。
> 日期：2026-11-18。前置证据：`docs/paper_b_modules/TEMPORAL_AGG_PILOT_20261118.md`、
> `tmp/cata_results/`（15 格 validation）、`paper_b/modules/temporal_agg.py`。

## 1. 假设

**H-CATA（主）**：在冻结的 Stage D B0 checkpoint 上，对每个 validation cut `c` 取
K=3 个因果窗口（结束于 `c, c-5, c-10`），做年龄对齐
`a_t = p_t - (c - (t+1))`，用 `median(a_t)` 作为 RUL 预测（**固定聚合，不依赖门控**）；
在 FD001 上相对 K=1 的 engine-mean RMSE 降低 **≥ 0.40**（约 3.4%），
且 **≥5/6 seed 同号、95% CI 排除 0**。

**H-GATE（已否决，保留记录）**：原假设“仅当
`MAD(a_t)/|median(a_t)| <= 0.05` 时聚合，否则用最新窗口”能作为**选择器**优于
同 rate 随机门控。**该假设未通过**：K=3 FD001 真实门控 10.778 vs 随机 10.405；
oracle AUC=0.447；按 §7 G-CATA-4 退为“年龄对齐 + 固定聚合率”，门控不再是主方法。

**H-NONINF（主，非劣）**：同一策略在 FD002/FD003/FD004 上相对 K=1 的
engine-mean RMSE **不劣化超过 0.50**（单侧 95% 上限 < +0.50）。点估计可以为负，
但不得显著劣化；若显著劣化，则模块只保留为 FD001 条件性方法，不写通用部署主张。

**H-MECH（机制）**：
1. 不做年龄对齐的聚合应显著更差（已实测：FD001 s101 K=3 12.17 vs K=1 11.56）；
2. 同 rate 的随机门控应不优于真实离散度门控（待全 seed 对照）；
3. 门控率在 FD001 应高于 FD004（已实测 s101：0.76 vs 0.67；待全 seed）。

## 2. 配置（冻结候选）

| 项 | 值 |
|---|---|
| backbone | Stage D B0（`DSPT_HC_Structured`，冻结 checkpoint） |
| K | 3 |
| Δ | 5 cycles |
| 聚合 | median of age-aligned predictions（**固定，无门控**） |
| 门控 | **不是主方法**；RGTA 未过 G-CATA-4，只作敏感性/实现细节 |
| fallback | 无（主方法对每个 cut 都聚合） |
| 训练 | 无（推理期，backbone 冻结） |
| 标签 | 不参与聚合/门控；只用于评价 |
| 窗口 | 只用 `c` 及以前的数据（因果） |

**τ 的来源必须写清**：0.05 是在 pilot 看到 FD001/FD004 s101 后选的，属于
**开发期选择**，不是预注册值。冻结前必须做 training-engine 标定（见 §5）；
若标定给出不同 τ，以标定值为准，并重跑全部 validation 后再冻结。

## 3. 评价

**主指标**：engine-mean RMSE，capped label（min(RUL,125)），prediction clip 到 [0,125]，
5 个 protocol cuts（0.4/0.55/0.7/0.85/0.95）。

**次要**：pooled RMSE、NASA score、MAE、bias、`rmse_uncensored`、逐 cut RMSE、
gate rate（聚合比例）。

**统计**：
- 逐 seed 配对差（K=1 − gated；正 = 模块更好），报 mean/sd/t/95% CI/符号计数；
- engine-level paired bootstrap（按发动机重采样，5000 次）给 CI；
- 全 seed、全子集都报，不做 seed 选择，不删子集。

## 4. 对照（缺一不可）

| 对照 | 目的 | 预期 |
|---|---|---|
| K=1 | 基线 | — |
| K=3 全窗口 median | 无门控 | FD001 改善，FD003/FD004 变差 |
| K=3 gated τ=0.05 | 主方法 | FD001 改善，其他不显著劣化 |
| 未对齐聚合 | 证明年龄对齐是机制 | 明显更差 |
| 同 rate 随机门控 | 证明门控选择有价值 | 不优于真实门控 |
| exact 工况匹配 / cap clamp | 机制诊断 | FD004 伤害减小，FD001 可能受伤 |
| learned aggregator | 方法上限（可选） | 不劣于固定 median；若更差则只写固定版 |

## 5. τ 标定（冻结前必做）

在 **training engines** 上做 cross-fit，不用 validation 选 τ：

1. 把训练发动机按 5 折分组；
2. 对每折，用其余折训练一个 B0（或复用已训练 B0 的 in-sample 预测，需注明）；
3. 在 held-out 折上，对每个 τ 计算 K=1 vs gated 的 RUL loss；
4. 选使 held-out loss 最小的 τ（或选一个稳定平台区的中位数）；
5. 把选定的 τ 冻结，再在 validation 上报告；validation 不参与 τ 选择。

若 cross-fit 成本过高，退路：用 **a priori τ=0.05**（相对离散度 5%）并在论文中
声明“未标定，按 5% 先验固定”，同时报 τ 敏感性。不得用 validation 的最优点当 τ。

## 6. Test 协议

1. 冻结代码（`paper_b/modules/temporal_agg.py` 的 sha256）、配置、checkpoint 列表、
   τ、K、Δ、聚合器、对照清单；
2. 写 `FREEZE_CATA.json`，列出每个 (subset, seed, checkpoint, sha256)；
3. 只打开一次官方 test；所有 subset/seed 一次跑完；
4. test 只用于报告，不用于选择；若 test 与 validation 方向不一致，如实报告，
   不改配置、不删子集、不换 τ；
5. test 结果写入 `docs/paper_b_modules/CATA_TEST_2026xxxx.md`，并同步 `results_summary.md`。

## 7. 判定门

| 门 | 通过条件 | 失败处置 |
|---|---|---|
| G-CATA-1 | FD001 6/6 seed 同号，mean ≥ +0.40，95% CI 排除 0 | 不写 FD001 部署收益 |
| G-CATA-2 | FD002/FD003/FD004 单侧 95% 上限 < +0.50 | 只写 FD001 条件性方法 |
| G-CATA-3 | 未对齐对照显著更差 | 机制不成立，撤回 |
| G-CATA-4 | 真实门控 ≥ 随机门控（同 rate） | 门控无独立贡献，退为固定率聚合 |

**G-CATA-4 实测（2026-11-18，15 格）：FAIL（K=3 主配置）。** FD001 K=3 真实门控
10.778 vs 同 rate 随机 10.405/10.506；FD004 K=3 real 13.918 vs random 14.157（real 更好）
但 AUC=0.502，FD001 AUC=0.447。按本表退路，主方法改为“年龄对齐 + 固定聚合（K=3 median）”，
门控不再作为贡献。审计：`AUDIT_CATA_20261118.md` §5/§8。
| G-CATA-5 | test FD001 方向一致且 ≥ +0.20 | 不写 test 收益，只写 validation |

## 8. 当前证据（2026-11-18，validation）

- FD001 K=3 gated τ=0.05：mean **+0.615**，sd 0.227，CI **[+0.376,+0.853]**，6/6 同号。
- FD001 K=3 **无门控** median：mean **+0.533**，6/6 同号（这是主方法的候选）。
- FD002：+0.041（CI 跨 0）；FD003：−0.185（CI 跨 0）；FD004：−0.322（CI 跨 0）。
- 全窗口 K=3 median：FD001 +0.533（6/6 同号）；FD003/FD004 负。
- 未对齐：FD001 s101 K=3 12.17 vs K=1 11.56。
- 随机门控（15 格）：FD001 K=3 real +0.615 vs random +0.630（random 不差于 real）；
  FD004 K=3 real −0.322 vs random −0.482（real 较好）；AUC 0.447/0.502 → 门控统计量不预测收益。
- 工况稳定性门控（新，3 格 s101）：FD001 op 距离≈0，τ≥4 退化为全聚合（+0.966）；
  FD002/FD004 低 τ 只聚合 4–24%，收益 −0.09~+0.08；AUC 0.486–0.589，无稳健门控。
- **结论**：CATA 的 FD001 增益真实且对 τ 稳健，但“门控选择器”两条路线（相对离散度、工况稳定性）
  均未通过；论文只能写**条件性聚合/经验刻画**，不能写通用模块。

## 9. 未决

- L2 独立裁决：Codex CLI 配额到 2026-11-21，MCP 不可用；本文件与模块主张标 **PENDING-L2**。
- 随机门控全 seed 结果；若 gate 不优于 random，主方法退为“年龄对齐 + 固定聚合率”，
  门控只作实现细节。
- learned aggregator 的实现与对照（agent 进行中）。
- test 冻结与打开需在 L2 可用后由主控线执行；本轮不打开。
