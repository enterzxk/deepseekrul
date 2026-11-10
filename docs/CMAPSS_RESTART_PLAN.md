# C-MAPSS 方向重启：状态恢复与执行计划

> 用户 2026-11-10 决定：主目标回到 NASA C-MAPSS FD001–FD004，目标 IEEE TIM。
> FEMTO 轴承支线保留（`docs/bearing_tim/`、`experiments/bearing_tim3/`、`scripts/run_bearing*.py` 不动）。
> 本文件是恢复点，新会话从这里开始。

## 1. 已确认存在的资产

| 资产 | 路径 | 状态 |
|---|---|---|
| 数据 | `data/CMAPSS/`（train/test/RUL_FD00x.txt，4 子集） | ✅ |
| 骨干 | `scripts/run_ida.py`（IDA-PE + `CandidateModel`） | ✅ |
| CQR 训练 | `scripts/run_e2e_cqr.py` | ✅ |
| 80 格矩阵 | `experiments/punet_matrix/`、`docs/tables/cells_all.csv` | ✅ |
| 冻结协议 | `docs/FINAL_PROTOCOL.md` | ✅ |
| 决策记录 | `docs/decision_log.md`（D001–D035） | ✅ |

## 2. 已知强基线（需在下一轮核对后引用）

- **AST 协议对齐**（git `bf61185`）：单模型均值 **11.85 / 13.07 / 11.87 / 14.19**
  （FD001–FD004），对照 AST **10.43 / 12.02 / 10.49 / 14.11**。
- 该数字来自 commit message，**下一轮必须从实验产物重新核对**，不能直接引用。
- 80 格 CQR 表（`cells_all.csv`）记录的是**区间**指标（const/cqr 等），不是点预测 RMSE，
  两者不可混用。

## 3. 协议（必须沿用，不得重设）

| 项 | 值 |
|---|---|
| 划分 | 发动机级 5 折（`outer_folds`，OUTER_SEED 固定） |
| 窗口 | 50 |
| 标签 | cap-125（`min(T−t, 125)`） |
| 归一化 | FD002/FD004 六簇工况归一化；FD001/FD003 min-max；只用训练机拟合 |
| 早停 | **必须与最终评价机互斥**（ES/TEST 分层，`split_es_test`） |
| 官方测试集 | 只在最终一次性使用，**不参与任何选择** |
| 指标 | 点预测：RMSE/MAE/NASA Score；区间：PICP/Winkler |

## 4. 三种 Transformer 输入（移植思想，不复制 FEMTO 网络）

| 模型 | 输入 | 序列长度 |
|---|---|---|
| **C-T1** 时间步 Transformer | 每时间步 = 当前工况(3) + 14 传感器 | 50 |
| **C-T2** 时间 Patch Transformer | 每 patch = 连续 P 个时间步的传感器+工况展平 | 50/P |
| **C-T3** 传感器—时间双维 Transformer | token = (时间步, 传感器) 对，保留传感器身份嵌入 | 50×14 |

三者共用：窗口 50、同一折、同一预算、同一早停。
必须与 `run_ida.py` 的 IDA-PE 强基线**同协议**比较。

## 5. PDEM 在 C-MAPSS 上的重新定义

核心问题：**区分工况造成的可逆传感器变化与长期退化造成的持续状态变化。**

1. **工况条件创新量**：用历史预测 `t` 时刻的传感器向量，但预测器输入**包含当前工况**，
   使"工况导致的变化"被解释掉，残差才反映退化：
   `e_t = normalize( x_t − f_θ(x_{<t}, oc_t) )`
2. **持续性**：`e_t` 在连续多个时间步上是否保持同向/同结构。
3. **有限容量事件记忆**：K 个槽位，写入/替换/读取规则明确。
4. **查询与输出**：当前状态查询记忆 → RUL。

**必须警惕**：创新量 ≠ 退化。工况变化、传感器噪声、个体差异都会产生残差。
因此必须有受控对照（见 §6），不能靠可视化宣称学到退化。

## 6. 强制机制对照

| 臂 | 内容 |
|---|---|
| B0 | 最佳普通 Transformer（C-T1/T2/T3 中的胜者） |
| B1 | B0 + 等容量普通记忆 |
| B2 | B0 + 随机/RMS 驱动事件记忆 |
| B3 | B0 + **工况条件创新量 + 持续性**的完整 PDEM |
| B4 | 原项目最佳点预测基线（IDA-PE） |

判据：B3 必须优于 B0、B1、B2 **且** 优于 B4，否则机制不成立。

## 7. 执行顺序

1. **核对基线**：从产物重算 AST 协议的 4 子集 RMSE（不是引用 commit message）。
2. **实现 C-T1/C-T2/C-T3**，在 FD002/FD004 上开发（多工况最能检验解耦假设）。
3. **裁决骨干** → 实现 B0–B3。
4. 扩到 FD001–FD004，多 seed，消融。
5. Codex 审查 + 最近邻（工况归一化、退化分解、事件记忆、时序 Transformer、C-MAPSS RUL）。

## 8. 与 FEMTO 的边界

- 不改 FEMTO 的任何实验、代码、文档。
- 不复用 FEMTO 的频带 Token（C-MAPSS 没有原始振动波形）。
- 本轮算力主要给 C-MAPSS。

## 9. 下一会话的第一件事

核对 AST 基线的真实 RMSE（从 `experiments/` 的预测文件重算），
确认后再实现 C-T1。**不要在没有核对基线的情况下开始新模型训练。**
