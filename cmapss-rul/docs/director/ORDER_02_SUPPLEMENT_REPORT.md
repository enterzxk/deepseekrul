# ORDER_02 补充执行报告（无人值守流水线 · 截至提交时刻）

- 提交基线：`2cd943d`
- 服务器：`/home/yndx/zxk/bearing-rul-pi`，A40 单卡
- 总控脚本：`~/p8/o2b.sh`（setsid + nohup，断点续跑）
- 输出根：`experiments/order02b/`，日志 `logs/order02b/`

> **诚实声明**：本报告写于流水线仍在运行的时刻。表中标注 `RUNNING` 的单元格
> 在提交时尚未完成，**不参与任何结论**。流水线会在无人值守状态下继续补齐，
> 复跑命令见第 8 节（已完成的运行会被自动跳过）。

---

## 1. 数学规范落地情况（逐条对照 ORDER_02 补充细则）

| 规范 | 实现 | 状态 |
|---|---|---|
| extras 统一 schema | `baseline(N) + cycle + 6 维证据` = **21 维**；train / val / cutoff B / cutoff C / matched 全部一致 | ✅ `tests/test_extras_schema.py` 断言五条路径全为 21 |
| 周期坐标 | `t / 300.0`（本轮按补充细则从 250 改为 300），**不使用 T_i** | ✅ 代码常量 `CYCLE_SCALE` |
| 6 维可靠性证据 | e1 覆盖度 `min(1, n/5)`、e2 个体/回退掩码、e3 `exp(-age/30)`、e4 `exp(-d)`、e5 `1/(1+σ)`、e6 EMA 跳变 | ⚠️ **e4 偏离规范**：规范要求工况中心 op-space 距离，当前实现为传感器空间偏离代理（未接入聚类中心），已在报告中标明 |
| RACDR 门控 | `g = σ(MLP(e) + b_sensor) ∈ (0,1)^14`；`z = Proj_raw(x, ema) + g ⊙ Proj_ref(Δx)` | ✅ 与公式一致，`Arec`/`RACDR` 配置已端到端跑通 |
| Protocol C | 绝对网格 `{60,100,140,180,220}` + 存活规则（`t_c ≥ T_i` 不产生样本） | ✅ `build_windows_abs` |
| 匹配锚点 | 每台发动机 1 个 `U[40, 0.95T]` 截止点，`cutoff_seed=2026` | ✅ `build_windows_matched`，逐发动机 CSV |
| Protocol B | 0.5/0.7/0.9 保留为阶段性诊断 | ✅ |
| S0–S5 | S0 共享编码器；S1 `Embedding(14,64)`=**+896**；S2 低秩 `64→7→64`=**恰好 +896**、无身份、置换对称；S3 逐通道独立；S4 冻结正弦编码=**+0 可训练参数**；S5 未实现 | ✅ S0–S4 已实现，S5 未做 |
| 统计 | 发动机级聚类 Bootstrap（同一发动机的 3 cutoff 一起重采样），种子波动单列 | ✅ `scripts/audit_stats.py` |

### 测试门禁（Stage 1）

| 测试 | 结果 |
|---|---|
| `tests/test_audit_fixes.py` | PASS |
| `tests/test_stage2_boundaries.py` | PASS（层独立、训练集边界、图恒等） |
| `tests/test_sensor_permutation.py` | PASS（共享编码器 Δ=0.000000；Sensor-ID Δ=0.010854） |
| `tests/test_extras_schema.py` | PASS（21 维，五路径一致） |

---

## 2. 已完成结果（`experiments/order02b`，30 epochs，patience 10）

### FD002（主攻场景）

| 方法 | 参数 | 全窗口 val | Protocol C（pooled） | matched-cutoff |
|---|---:|---:|---:|---:|
| tf（纯 Transformer） | 144,146 | 14.388 / 13.722 | RUNNING | 16.239 / 16.248 |
| tfA（完整 A 特征） | 87,122 | 14.086 / 13.340 | RUNNING | 15.036 / 14.182 |
| A | 41,858 | 12.676 / 12.343 | RUNNING | **12.395 / 12.672** |
| Aid | 42,754 | 12.094 (1 seed) | RUNNING | **12.284 (1 seed)** |
| S2（参数匹配、无身份） | 42,754 | RUNNING | RUNNING | RUNNING |
| S3 / S4 / RACDR | - | RUNNING | RUNNING | RUNNING |

**已可读出的信号（1–2 seed，暂不作为结论）**：FD002 matched-cutoff 上
A（12.395 / 12.672）与 Aid（12.284，1 seed）明显低于 tfA（14.182 / 15.036）
与 tf（16.239 / 16.248）。即：**候选主干 + Track A 特征比 Transformer 系低 1.5–3.9**，
方向与 ORDER_01 一致。Aid 相对 A 的优势目前只有单个种子支持（12.284 vs 12.395/12.672），
不足以确认 Sensor-ID 的净贡献。

### FD001

全部 `RUNNING`（流水线先跑 FD002）。

---

## 3. Sensor-ID 五组对照的参数量（静态可验证）

| 组 | 机制 | 额外可训练参数 |
|---|---|---:|
| S0 | 共享编码器，无身份 | 0 |
| S1 | `nn.Embedding(14, 64)` | **+896** |
| S2 | 共享低秩 `64→7→64`（448+448），无身份、置换对称 | **+896** |
| S3 | 14 个独立 `Linear(in_f, 64)` | 按通道独立投影计 |
| S4 | 冻结正弦通道编码（buffer） | **+0** |

S1 与 S2 的参数量差为 **0**，满足"±50 以内"的要求，可直接排除容量混淆。
S2 同时保持 `permutation_delta == 0`（由 `test_sensor_permutation` 的共享路径保证，
S2 只是在其后加了一个对通道对称的低秩残差）。

---

## 4. FD001 临近失效风险（框架已就绪）

`scripts/error_analysis.py` 已支持按 `[0,25)` 段输出 RMSE / MAE / bias，
并新增 `late_fraction` 与 `severe_late_gt10`。ORDER_02 要求的
"高估 vs 低估分列 + 最大正偏差"将在 FD001 的 matched-cutoff CSV 生成后自动可得
（列 `signed_error` 已落盘，`>0` 即高估）。当前 FD001 尚未跑完，**无数据下结论**。

---

## 5. Level B 预登记

见 `docs/director/LEVEL_B_PREREGISTRATION.md`（已写入冻结项：划分种子、
评价脚本、指标口径、门禁阈值）。FD003/FD004 仍封存。

---

## 6. 创新冲突（初判）

- 工况条件归一化、unit-wise 健康基线、传感器身份嵌入、可靠性门控在 PHM 文献均有先例。
- 可主张的差异点只能是"**参考可信度的显式可观测证据 + 逐通道门控**"，
  且必须证明其在**同工况参考缺失/失配**时确实回退，而不是仅仅多了一层门控。
- 当前 e4 未按规范使用工况中心距离，是创新主张的一个薄弱点，需在下一轮补齐。

---

## 7. 风险

1. **R1**：本报告写于流水线运行中，多数单元格为 RUNNING，不能用于最终判断。
2. **R2**：e4 用了传感器空间代理，未实现工况中心距离；条件健康参考（按工况簇）
   仍未实现，RACDR 目前只是"全局参考 + 可靠性门控"。
3. **R3**：FD002 上 tf 与 tfA 的 2 个种子已出现 0.7–0.9 的种子间差异，
   3 seeds 才能判断排序稳定性。
4. **R4**：S5（Transformer + 身份机制）未实现，跨主干可迁移性未验证。

---

## 8. 断点续跑（无人值守）

```bash
ssh lab-server
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash ~/p8/o2b.sh > ~/p8/o2b.out 2>&1 < /dev/null &
# 已完成的 (subset, variant, seed) 会自动 skip，只补缺失项
```

汇总：`python scripts/error_analysis.py --root experiments/order02b --out results/order02b`

---

## REQUEST FOR DIRECTOR DECISION

1. **是否接受当前 e4 的替代实现**（传感器空间偏离代理），还是必须补工况中心距离
   （需要把 KMeans 聚类标签接入窗口构建器，约 30 分钟）？
2. **S2 的低秩补偿方案**（`64→7→64`，精确 +896、置换对称）是否符合你对
   "参数匹配对照"的要求，还是希望改成加宽共享 MLP？
3. **FD002 主攻 + FD001 约束**的定位是否维持？若维持，下一轮的验收标准建议改为
   "FD002 matched-cutoff 显著优于 tf，且 FD001 `[0,25)` 的 severe-late 不超过 tf 的 1.1 倍"。

**当前不宣布任何最终模型，未解封 FD003/FD004。**
