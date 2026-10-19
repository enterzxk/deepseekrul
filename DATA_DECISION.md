# DATA_DECISION — 固定起点容量-only SOH 轨迹研究

**状态：** 已完成数据审计与主分析；5 折为开发集，不是最终独立确认集。  
**主数据：** Battery Archive 公开标准化 per-cycle 数据（101 颗 SOC 0–100% 电芯）。  
**辅助审计：** CALCE 官方原始 XLSX 12 颗（6×CS2 + 6×CX2），不并入主 101 表。  
**未使用：** Codex `/home/yndx/zxk/rul` 的任何数据/权重/预测/测试；MATR/NASA 不作为本文数据。

## 1. 物理电芯与来源

| 来源 | 电芯数 | 观察 EOL80 | 观察 EOL70 | 循环中位数 | 化学/工况 |
|---|---:|---:|---:|---:|---|
| SNL | 61 | 39 | 10 | 785 | LFP/NCA/NMC；15/25/35°C；1/2/3C；0–100% SOC |
| HNEI | 15 | 15 | 15 | 1108 | NMC-LCO；25°C；0.5C 充 / 1.5C 放；0–100% |
| UL-PUR | 10 | 4 | 0 | 222 | NCA；23°C；0.5C；0–100% |
| Oxford | 8 | 4 | 0 | 7750 | LCO；40°C；2C 充 / 1.84C 放；每 100 循环表征 |
| CALCE | 7 | 7 | 7 | 1815 | LCO；25°C；0.5C；0–100% |
| 合计 | **101** | **69** | **32** | — | 5 来源、5 化学体系 |

- 电芯身份：使用 Battery Archive 的 `cell_id`，包含来源/化学/温度/SOC 窗口/C-rate/序号；不会把同一物理电芯的多个循环当成多个独立电芯。
- 主 split：5 折按物理电芯隔离，按 `(source, chemistry)` 分层；测试电芯从不进入训练/验证。
- 跨来源压力测试：leave-one-source-out（训练 4 个来源，测试剩余 1 个）。
- 未做 cycle-level random split；未按 EOL/knee 后验位置选样本。

## 2. 观测与标签语义

- 原始列：`ah_c`（充电容量）、`ah_d`（放电容量）、`e_c/e_d`（能量）、`cycle_index`、`test_time`。
- 有效循环过滤：
  - `0.2·nominal < ah_d < 1.4·nominal`
  - `0.2·nominal < ah_c < 1.4·nominal`
  - `0.8 < ah_d/ah_c < 1.2`
  - 目的：剔除双循环、半循环、异常 RPT/设备伪影；不按验证误差事后删点。
- 初始容量：前 5 个有效循环 `ah_d` 的中位数。
- `SOH = ah_d / initial_capacity`（相对初始容量，不是相对额定容量）；这样跨来源的初始差异不被误读为退化。
- 目标平滑：居中 9 点中位数（仅标签）；模型输入使用因果 5 点中位数或原始历史，不使用未来。
- `EOL80`：平滑 SOH 首次 ≤0.80 的循环；`EOL70` 同理。未到阈值为右删失，不填伪 EOL。
- 预测起点：`k0 ∈ {50,100,200,300,400}`；主表共同集 `k0≥100`，horizon `h ∈ {50,100,200,300}`；`k0+h` 不得超过最后有效循环。
- EOL 评估：仅用观察到 EOL80 的电芯；`k0 < 0.9·EOL`；预测轨迹 10,000 循环内未穿越记为未决/删失。

## 3. 数据质量与已知限制

1. **Battery Archive 标准化**：不是原始测试文件；可能已平滑/协议归一。不能与原始 CALCE 逐循环完全同分布。
2. **双循环/部分循环**：部分 `cycle_index` 含两次充放或部分充放（如 SNL early cycle 4 ≈2×nominal）；已按容量/库仑效率过滤，仍可能残留。
3. **Oxford 稀疏**：46–78 个表征点覆盖 5,000–8,200 循环；EOL 标签稀疏。HOF 在 Oxford 的 EOL 预测对 3/4 电芯不穿越；EOL 误差 2,485 循环只基于 1 颗电芯，不能外推为 Oxford 平均。
4. **SNL LFP 删失**：21 颗 LFP 中只有 5 颗观察到 EOL80；EOL 指标是条件子集，不是全样本无偏。
5. **HNEI 短寿命**：1.5C 下 EOL80 约 196–249 循环；同协议内形状很稳定，HOF EOL MAE 5.6 循环，不能把这当作跨来源容易。
6. **UL-PUR 短轨迹**：10 颗中只有 4 颗观察到 EOL80；主表共同窗口可能较少。
7. **CALCE 原始审计**：CS2-8/21、CX2-31 是 CADEX txt，未解析；CX2-16 下载不完整；主分析只用 12 颗 XLSX。原始 zip 的 SHA256/URL 在 `data/calce_manifest.json`。

## 4. 数据来源与获取

- Battery Archive：`database.batteryarchive.org` 公开 `Cycle Test Data` dashboard；query 29（cell metadata）+ query 26（per-cycle capacity/energy）。检索日期 2026-10-19。101 个 query-26 原始 JSON 打包在 `data/ba_raw2.tar.gz`；抓取脚本 `code/scrape_full.py`（本地 Edge + Playwright）。
- CALCE：`https://calce.umd.edu/data`；zip URL 与 SHA256 见 `data/calce_manifest.json`；逐循环 CSV 在 `data/calce_parsed/`。
- 文献原件：`tools/fetch_original_literature.py` 已本地运行并通过 Git blob SHA 校验；原件在 `literature/original_from_github/`。未上传论文 PDF。

## 5. 与 Codex / BatteryMFormer 的独立性

- **Codex：** 未使用 MATR/NASA、DataV1、`/home/yndx/zxk/rul` 文件、checkpoint、调度器或 test。
- **BatteryMFormer：** 本文数据源可能与其 13 个子数据集重叠（SNL/HNEI/CALCE/Oxford 均可能被其使用）。本文**不能声称**对 BatteryMFormer 完全数据独立；投稿前必须核对其数据列表/划分。
- 本文的 split 是 cell-level 5 折 + leave-one-source-out；不是 BatteryMFormer 的 aging-condition-exclusive split，也不是其任务输入（电压/电流早期 100 循环）。

## 6. 决策

- **GO（有条件）：** 101 颗物理电芯、69 颗观察 EOL80，足以做 fixed-origin 容量-only 的诊断与最小基线；主表 cell-level 5 折 + 逐电芯指标可复现。
- **NO-GO（当前）：** 不能把当前 HOF 作为跨来源通用方法；leave-one-source-out 明确失败（h=300 cell-equal 0.0820；EOL 无穿越）。需要协议先验迁移/独立来源确认后才能主张通用。
- **不得偷偷重置：** 后续若换数据/换 split/换阈值，必须新建协议并保留本文件的旧结论，不得用新结果覆盖旧失败。
