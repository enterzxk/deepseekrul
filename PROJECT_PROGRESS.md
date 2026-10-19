# PROJECT_PROGRESS — DeepSeek Battery Trajectory Research

**最后更新：** 2026-10-19  
**工作目录：** `/home/yndx/zxk/deepseek-reaserch`  
**状态：** 已完成多源数据审计、主基线、HOF 最小方法、EOL/区间/跨来源压力测试；尚未冻结最终独立确认集。

## 已完成事实

1. **环境与边界**
   - 本地 `lab-server` SSH 连接成功；服务器目标目录已创建。
   - 服务器无外网；数据/文献在本地联网获取后通过 SSH 传输。
   - 未使用服务器 GPU；未操作 Codex `/home/yndx/zxk/rul`；未杀进程；未请求私钥。
   - 当前执行端无子代理接口，全部串行完成；未声称启动子代理。

2. **数据**
   - Battery Archive 101 颗 SOC 0–100% 电芯，5 来源（SNL/HNEI/UL-PUR/Oxford/CALCE），5 化学体系；69 颗观察 EOL80。
   - 抓取 query 29（metadata）+ query 26（per-cycle capacity/energy），保存 101 个原始 JSON；清洗为 `data/ba_clean.csv`、`data/ba_cell_summary.csv`。
   - CALCE 官方原始 12 颗 XLSX（6 CS2 + 6 CX2）解析为 `data/calce_parsed/`；原始 zip 的 URL/SHA256 在 `data/calce_manifest.json`。
   - 未使用 Codex/MATR/NASA 数据。

3. **实验**
   - 5 折按物理电芯隔离，按 `(source, chemistry)` 分层；共同评估 88 电芯/58,080 窗口。
   - 基线：persistence、linear-100、exp-100、power-law、protocol mean、GRU、residual GRU。
   - 最小方法：HOF = 协议组中位数轨迹 + 当前电芯偏移；EOL 由轨迹首次 ≤0.8 得到。
   - 主结果：h=300 cell-equal SOH MAE HOF 0.0224、GRU 0.0262、exp 0.0424、power 0.0477、persistence 0.0785；HOF EOL MAE 165.3、median 48、within-20%（有预测）75.8%；37 个案例未穿越。
   - 失败：leave-one-source-out h=300 cell-equal 0.0820；EOL 194/194 无穿越；Oxford 3/4 电芯未预测 EOL；CALCE/SNL NCA 形状变化时 HOF 失效。
   - 区间：SOH 80%/90% 覆盖约 0.80/0.90；EOL 80% 覆盖 0.809，90% 覆盖 0.857（不足）。

4. **文献**
   - 阅读 BatteryMFormer（arXiv 2605.27044v2）、2024 JPS knee（arXiv 2304.11671v3）、2025 JPS transferable physics-informed（arXiv 2501.14573v2）全文/主要章节。
   - 2025 Applied Energy 为 closed access，未获得摘要/全文；2025 JES/2026 GEITS/2026 EA-LSTM 仅摘要/元数据。
   - 已写 `NOVELTY_CONFLICT_MATRIX.md`，明确不能声称 SOTA/首次。

5. **交付**
   - `reports/DEEPSEEK_RESEARCH_REPORT_20261019_ZH.md`（中文主报告）
   - `DATA_DECISION.md`、`BASELINE_RESULTS.md`、`NOVELTY_CONFLICT_MATRIX.md`
   - `results/` 逐窗口/逐电芯/EOL/区间/跨来源 CSV 与 `final_metrics.json`
   - `figures/` 3 张图（MAE vs horizon、EOL error by source、cross-source）
   - `code/` 抓取/清洗/评估/训练脚本

## 关键判断

- **GO（有条件）：** 同协议 seen-protocol 的 capacity-only 长程预测，HOF 是强、简单、可解释的最小基线；可写成基准/现实核查论文。
- **NO-GO（当前）：** 跨来源通用方法；leave-one-source-out 明确失败，EOL 无穿越。
- **未解决：** BatteryMFormer 数据重叠、2025 Applied Energy 全文、独立来源确认、多 seed 稳健性、EOL 90% 区间覆盖不足（cell-level paired bootstrap 已做，见报告 §6.7）。

## 资源/锁

- 无 GPU 锁；未占用服务器 GPU。
- 未使用 Codex 的调度器/owner/test。
- 服务器目录：`/home/yndx/zxk/deepseek-reaserch`；本地交付包：`E:/桌面/rul-deepseek/deliverable`。

## 下一个原子动作

1. 在服务器上核对上传文件哈希与路径；运行 `code/final_metrics.py` 复算 `results/final_metrics.json` 的关键数字。
2. 冻结 leave-one-source-out 协议 + 协议先验迁移的最小方案（不是再调 GRU）。
3. 获取 BatteryMFormer 官方数据列表/代码，确认 13 子数据集是否含本文来源；获取 2025 Applied Energy 全文。
