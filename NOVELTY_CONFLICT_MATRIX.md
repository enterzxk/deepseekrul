# NOVELTY_CONFLICT_MATRIX — 最近邻论文与本文边界

**原则：** 不称“首次”，不称“SOTA”；只记录作者原文报告、本项目实测、以及尚未核实的点。BatteryMFormer、2024 JPS、2025 JPS 三篇读到了 arXiv 全文/主要章节；其余以摘要/元数据为主。2025 Applied Energy 为 closed access，未获得摘要/全文。

| 论文 | 作者明确任务 | 输入截止 | knee/EOL 语义 | 数据/划分 | 与本文重合 | 本文差异/边界 | 未核实 |
|---|---|---|---|---|---|---|---|
| **BatteryMFormer (KDD 2026)** arXiv:2605.27044 | 早期 operational data → 全寿命 SOH 轨迹；aging-condition-aware decoder + meta degradation memory + dual-view encoder | 早期 ≤100 循环的电压/电流/SOC | EOL 80%（CALB 90%）；部分外推补轨迹 | 4 domains；aging-condition-exclusive 6:2:2 或 leave-one-condition-out；MAE/MAPE | 长程 SOH 轨迹；多来源 | 本文 capacity-only、fixed-origin、任意 k0、多源/删失；HOF 是一参数层级基线；**未在 BatteryMFormer 任务上运行** | 13 子数据集是否含本文来源；官方 split/代码；capacity-only 适配 |
| **2024 JPS knee curvature** 10.1016/j.jpowsour.2024.234619 | 曲率近似识别 knee/knee-onset；在线早期预测 | 容量 fade 前缀 | 新 knee 定义；knee 与 EOL 强相关 | 两种化学 + 合成数据 | knee/EOL 语义 | 本文不做 knee 检测；把时间尺度/协议偏移作为主瓶颈 | 真实数据来源/协议；代码 |
| **2025 JES multi-time-scale** 10.1016/j.est.2025.116024 | 前 100 循环多时间尺度特征 + attention/MLP + snapshot ensemble → RUL/knee | 前 100 循环 | RUL + knee | 123 电池；主/副测试集 | 早期长程预测 | 本文 fixed-origin capacity-only；不做 knee 分类 | 数据身份、EOL 语义、代码 |
| **2025 Applied Energy fade+ML** 10.1016/j.apenergy.2025.125703 | 经验容量衰减模型 + ML 早期预测整条容量轨迹 | 早期循环 | 容量轨迹（可能含 EOL） | 未知（closed） | **最接近的“参数化 fade + ML 轨迹”** | HOF 是非参数协议均值+偏移，无参数 fade 模型；但必须拿到全文后才能判断重合 | 全文/摘要/数据/代码均未获得 |
| **2025 JPS transferable physics-informed** 10.1016/j.jpowsour.2025.238028 | 2D histogram + 混合物理模型 + fine-tuning → 诊断/knee-onset/knee | 在线 prefix | knee onset/knee；EOL 相关 | ICL 18 cells；源/目标域 | 在线/迁移；knee | 本文不做 knee 检测；跨来源失败是本文的核心负结果 | 轨迹预测能力；数据重叠 |
| **2026 GEITS multi-fidelity** 10.1016/j.geits.2025.100367 | P2D + SEI/析锂 + 多保真 ML → knee/RUL | 稀疏早期数据 | knee + RUL | 仿真+实验（待核） | 物理先验+长程 | 本文非参数、无 P2D；不声称物理机理 | 数据谱系；代码 |
| **2026 EA-LSTM** 10.1016/j.rineng.2026.109664 | 前 100 循环多循环统计特征 → KO/KP/EOL | 前 100 循环 | knee onset/knee/EOL | MIT MaTR 118 + CLO 32 | 早期 EOL/knee 多任务 | 本文 fixed-origin、capacity-only；不做 KO/KP | 与 Codex MATR 数据重叠风险 |

## 可主张的贡献（谨慎表述）

1. **一个 capacity-only、fixed-origin、cell-level 隔离的多来源实证**：101 颗物理电芯、5 个官方来源、69 颗观察 EOL80；同时报告 SOH horizon-MAE、EOL、删失、per-source/per-cell。
2. **可预测性分解**：协议均值、电芯级近常数偏移、残差动态三者分开评估；显示“协议均值+偏移”已解释大部分可用信号，残差 GRU 没有稳定增益。
3. **强基线**：HOF（`μ_g(k)+r_c(k0)`）在 h=300 cell-equal SOH MAE 0.0224，优于 GRU 0.0262、exp 0.0424、power 0.0477、persistence 0.0785；EOL MAE 165.3、median 48、within-20%（有预测）75.8%。
4. **校准区间基线**：HOF 的 out-of-fold 80%/90% SOH 覆盖率约 0.80/0.90；EOL 80% 覆盖 0.809、90% 覆盖 0.857（不足）。
5. **跨来源负结果**：leave-one-source-out h=300 cell-equal 0.0820；EOL 194/194 无穿越。明确告诉后续方法：没有协议先验时不可用。

## 不能主张的

- “首次长程 SOH 轨迹预测”“首次 knee 检测”“首次不确定性”“首次物理+ML”。
- “优于 BatteryMFormer”或“优于 2025 Applied Energy”：没有在它们的输入/数据/任务上运行，且 2025 Applied Energy 全文未获得。
- “跨来源通用”：留一来源明确失败。
- “数据完全独立于 BatteryMFormer”：其 13 子数据集可能包含本文来源，需核实。
- “HOF 是 SOTA”：只是一个强的最小基线；GRU/深度模型未穷尽调参。

## 论文定位建议

- **若目标是 Applied Energy/JPS/PHM：** 当前证据可支撑“多源 fixed-origin 容量-only 基准 + 协议均值/偏移分解 + 校准基线”的论文；标题/摘要必须把范围限定为 capacity-only、seen-protocol，并报告跨来源失败。
- **若目标是 KDD/NeurIPS：** 当前方法新颖性不足；需要把“协议先验迁移/时间尺度不确定性”做成真正的新机制，并在独立来源确认集上验证。不要把 HOF 包装成复杂模型。
- **与 Codex 的关系：** 不比较、不合并、不共享 test；Codex 是在线 RUL 更新，本文是固定起点轨迹/EOL。
