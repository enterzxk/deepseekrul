# Phase 1 最终报告：RHSE-TSH、EOL FPR、Gate 1 判定

> 日期：2026-11-19。协议冻结：`results/FREEZE_PHASE1.json`（SHA256 + gamma=1.30、
> mag=3、sensors 0/5/10、onset 0.2/0.5、R0–R7）。**test 全程封存。**
> 结论：**clean 代价与 EOL FPR 两个硬门通过；平均 recovery 与 drift 门不通过。
> Gate 1 正式失败，Phase 2 不启动。**

## 1. 协议与臂

- 故障：bias / drift / noise / stuck；mag=3（z-score 空间；stuck 原始物理空间待补）。
- 检测：RHSE-TSH（S1 Shewhart、S2 CUSUM、S3h innovation 方差、S3l 15 步活动度 run-length、
  2-of-3 确认、因果锁存、kmax=4）。
- 重构：R2 oracle Hθ；R3 TSH+Hθ；R5 TSH+last-value；R6 onset-aware hybrid；
  R7 5 周期因果中值滤波。
- 所有结果 FD001–FD004、s101、5 cuts、3 个传感器（0/5/10）。

## 2. clean 代价（R3c − R0，验证集）

| 子集 | R0 | R3c | 代价 | Gate ≤0.15 |
|---|---:|---:|---:|---|
| FD001 | 12.647 | 12.671 | **+0.024** | 通过 |
| FD002 | 12.732 | 12.864 | **+0.132** | 通过 |
| FD003 | 14.299 | 14.448 | **+0.149** | 通过（贴边） |
| FD004 | 18.442 | 18.406 | **−0.036** | 通过 |

## 3. EOL FPR（clean 末 20%，gamma=1.30）

| 子集 | per-sensor-cycle | any-window |
|---|---:|---:|
| FD001 | **0.32%** | 4.43% |
| FD002 | **1.04%** | 13.02% |
| FD003 | **1.31%** | 9.71% |
| FD004 | **1.17%** | 12.39% |

- 按 per-sensor-cycle（与首轮 simple RHSE 1.8% 同口径）：**全部 ≤2%，通过**。
- any-window 口径仍 4–13%；论文需明确口径，并同时报告两者。

## 4. RUL 恢复率（R3 / R5 / hybrid / filter）

| 子集 | bias R3/R5 | drift R3/R5/hybrid | stuck R3/R5 | noise R3/R5 |
|---|---:|---:|---:|---:|
| FD001 | 0.862 / 0.782 | 0.855 / 0.984 / **0.972**（onset0.2 0.912） | **0.895** / 0.294 | R1<R0；R7 12.424 最好 |
| FD002 | 0.913 / 0.844 | 0.603 / 0.525 / 0.536 | 0.167 / −0.776 | −2.106 / −6.854 |
| FD003 | 0.808 / 0.791 | 0.765 / 0.772 / **0.793** | 0.440 / 0.101 | 0.581 / 0.400 |
| FD004 | 0.801 / 0.502 | 0.518 / 0.322 / 0.307 | **0.702** / −0.758 | 0.338 / −1.173 |

## 5. Gate 1 判定

| 条件 | 结果 |
|---|---|
| clean 代价 ≤0.15 | **通过**（四子集） |
| EOL FPR ≤2%（per-sensor-cycle） | **通过**（0.32–1.31%） |
| 平均 recovery ≥0.65，≥3/4 子集 | **不通过**：FD002/FD004 明显低于 0.65；FD003 贴边 |
| R4 在 stuck 上领先 R5 ≥0.10 | **通过**（FD001 +0.60、FD002 +0.94、FD003 +0.34、FD004 +1.46） |
| R4 在 drift 上领先 R5 ≥0.10 | **不通过**：FD001 晚漂移 R5 更好；FD002/FD003 未达；只有 FD004 +0.20 |
| onset=0.20 领先 R5 ≥0.10 | FD001 drift +0.079（接近）；stuck 大幅通过 |

**Gate 1 正式失败。** 失败集中在：
- **FD002/FD004 的 drift/noise**：多工况下 Hθ 重构/滤波都不够；FD002 noise 的 R2 < R1。
- **FD003 stuck 0.440**：15 步 run-length 的检测延迟对短窗口/晚 onset 太贵。
- **drift 的 last-value 优势**：晚漂移 R5 0.984，Hθ 无法在 +0.10 上超过；这是重构方式问题，
  不是检测问题。

## 6. 可写的单工况结果

| 场景 | 证据 | 可写性 |
|---|---|---|
| FD001/FD003 bias | R3 0.808–0.862，clean +0.024/+0.149 | 条件性模块 |
| FD001 drift（早/晚） | hybrid 0.912/0.972；R5 0.984 | 条件性 + hybrid 机制 |
| FD003 drift | R3 0.765 / hybrid 0.793 | 条件性 |
| FD001/FD004 stuck | R3 0.895/0.702；R5 0.294/−0.758 | 强条件性，EOL FPR 过门 |
| FD001 noise | R7 12.424（比 clean 还好） | 单工况滤波，弱证据 |
| FD002/FD004 多工况 | drift/noise 失败 | **边界/负结果** |

## 7. 结论与下一步

- **RHSE-TSH 不是通用模块**；它是**单工况传感器故障前端**，在 bias/drift/stuck 上有
  可解释的条件性收益，clean 代价和 EOL FPR 过门。
- **LTAG 仍是唯一 test 级验证的主模块**（15/15，+0.398）。
- **Phase 2 端到端不启动**：Gate 1 未过；多工况 noise 的 R2 < R1 说明 Hθ 归纳偏置错误，
  端到端只会放大失败。
- **路线取舍 PENDING-L2**：RHSE 主贡献 vs LTAG 主线；Codex 2026-11-21 后裁决。
- 下一步只做可逆整理：单工况表、边界/限制、论文叙事；test 保持封存。

**一句话：clean 代价与 EOL FPR 已过门，stuck 也过门；Gate 1 死在 FD002/FD004 的
drift/noise 和 FD003 stuck；RHSE 只能写单工况条件性，Phase 2 暂缓。**

---

## 8. Post-audit correction（2026-11-19 23:50，必须读）

Phase 1c 预审计发现 **R5（last-value）用了 clean 反事实值**：`tsh_rul_eval.py:97` 的
`x[max(0, fb-1), jj]` 取的是 clean 输入，而不是故障输入 `x_f`。这使 R5 成为 oracle，
并抬高了 R6（晚漂移走 R5）。修复后重跑 FD001 s101 mag=3：

| 场景 | R3 Hθ | R5 last-value（修复后） | R6 hybrid |
|---|---:|---:|---:|
| FD001 bias | 0.862 | **0.059** | 0.862 |
| FD001 drift | **0.861** | **0.313** | 0.311 |
| FD001 stuck | **0.895** | **−0.012** | 0.895 |

- **撤回**“drift 上 last-value 更好”的结论：修复后 R3 0.861 远高于 R5 0.313。
  §4/§5 中所有 R5、R6 的数字均为修复前 oracle 泄漏版本，**不得再引用**；R3 数字不变。
- clean 代价变为 **−0.0063**（修复后 EOL 规则/NaN 处理也变了）。
- EOL（cut=0.95，sensors 0/5/10）：FD001 per-sensor-cycle **0.24%**、any-window **5.0%**；
  FD003 **0.00%**、any-window **0.00%**。
- **Gate 1 的 drift 条件现在通过**（R3 0.861 vs R5 0.313，+0.548）；但 FD002/FD004 的
  noise/drift 仍可能拉低平均 recovery，Phase 2 仍暂缓，等 Phase 1c 3-seed 结果。
- 其他修复：recovery 在 `R1−R0≤0.05` 时输出 `n/a`；`--gamma` 默认 1.30；
  q 加 floor；`u` 用 `nanmax`；R4 的 hold 起点用真实 onset。
- 代码哈希已变，`results/FREEZE_PHASE1.json` 的旧哈希作废；Phase 1c 使用
  `results/FREEZE_PHASE1C.json`。
