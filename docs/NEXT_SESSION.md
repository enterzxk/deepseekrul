# NEXT_SESSION — 恢复用最小上下文

**日期：** 2026-10-19  
**先读：** `PROJECT_PROGRESS.md`、`reports/DEEPSEEK_RESEARCH_REPORT_20261019_ZH.md`、`BASELINE_RESULTS.md`。不要重新通读 41 篇文献；最近邻只看 `NOVELTY_CONFLICT_MATRIX.md`。

## 当前有效文件/目录

- 数据：`data/ba_clean.csv`、`data/ba_cell_summary.csv`、`data/celllist.json`、`data/ba_raw2.tar.gz`、`data/calce_manifest.json`、`data/calce_parsed/`
- 结果：`results/final_metrics.json`、`results/hof_eval_clean.csv`、`results/hof_eol_clean.csv`、`results/eol_compare_clean.csv`、`results/unified_percell.csv`、`results/gru_pred.csv`、`results/resgru_pred.csv`、`results/losource_hof_pred.csv`、`results/per_cell_h300.csv`
- 代码：`code/final_metrics.py`、`code/hof_eval_clean.py`、`code/eol_compare_clean.py`、`code/train_gru.py`、`code/train_resgru.py`、`code/losource_hof.py`、`code/clean_ba.py`、`code/parse_ba.py`、`code/scrape_full.py`
- 图：`figures/fig1_mae_vs_horizon.png`、`figures/fig2_eol_error_by_source.png`、`figures/fig3_cross_source.png`

## 已验证的关键数字（不要重跑成功 smoke）

- 共同评估：88 电芯 / 58,080 窗口；5 折 cell-level。
- h=300 cell-equal SOH MAE：HOF 0.0224、GRU 0.0262、exp 0.0424、power 0.0477、persistence 0.0785。
- EOL80 HOF：157 个有预测案例，MAE 165.3、median 48、within-20% 75.8%（有预测）/61.3%（全部 194，含未穿越=失败）；37 个未穿越。
- leave-one-source-out HOF h=300 cell-equal 0.0820；EOL 194/194 无穿越。
- SOH conformal：80% 覆盖 0.797–0.802，90% 0.897–0.899；EOL 80% 0.809，90% 0.857。

## 阻塞/风险

1. **跨来源通用性未解决**：HOF 的协议均值不能迁移到新来源；当前 leave-one-source-out 失败。不要把它写成通用方法。
2. **BatteryMFormer 数据重叠未核**：其 13 子数据集可能含 SNL/HNEI/CALCE/Oxford；投稿前必须核对。
3. **2025 Applied Energy 全文未获得**：这是最接近的“fade model + ML 轨迹”工作；不能声称新颖性。
4. **Oxford EOL 稀疏**：3/4 电芯未预测穿越；不要把 2,485 循环误差当作 Oxford 平均。
5. **没有独立确认集**：5 折是开发集；没有冻结的 test。

## 下一个原子动作（按顺序）

1. 在服务器 `/home/yndx/zxk/deepseek-reaserch` 下运行：
   ```bash
   python3 --version  # 建议用已有 conda 环境，不要改 base
   # 服务器已实测：a40_env 可跑；base python3.8 无 pandas
   cd /home/yndx/zxk/deepseek-reaserch
   PYTHON=/home/yndx/miniconda3/envs/a40_env/bin/python bash code/run_all.sh
   # 或只复算已有结果：
   cd work && /home/yndx/miniconda3/envs/a40_env/bin/python ../code/final_metrics.py
   ```
   需要 `numpy/pandas/scipy/scikit-learn`；如果服务器 base 缺包，使用独立 venv/conda env，不要 pip 改 base。
2. 复算 `results/final_metrics.json` 的 h=300 HOF/GRU/exp/power/persistence 与 EOL 数字；如果路径不一致，只改脚本路径，不改数据。
3. 设计 **协议先验迁移** 的最小实验：
   - 目标：在完全留出的来源上，用训练来源学习 `(chemistry,temperature,C-rate) -> 时间尺度/形状` 的层级先验；
   - 不允许用目标来源的 EOL 标签；
   - 评价 h=300 SOH MAE 与 EOL 穿越率/误差；
   - 如果仍失败，明确把论文范围收缩为 seen-protocol。
4. 获取 BatteryMFormer 官方 repo 的数据列表/split；确认 13 子数据集是否包含本文来源。
5. 获取 2025 Applied Energy 全文；写一页重合矩阵；若其已报告协议均值/偏移基线，重新定位本文贡献。
6. cell-level paired bootstrap CI 已补（报告 §6.7：HOF−GRU h300 −0.0048 [−0.0091,−0.0006]；HOF−Power −0.0263 [−0.0404,−0.0161]）。下一步补多 seed/独立确认，以及 EOL 90% 覆盖不足的修正（按来源/化学分层 conformal 或生存模型）。

## 不要做

- 不要操作 `/home/yndx/zxk/rul`、Codex checkpoint/调度器/owner/test。
- 不要在未确认 GPU 空窗时占用服务器 GPU；当前无需 GPU。
- 不要重复下载/重跑已验证的 smoke；不要用多 seed/数十组模型凑结果。
- 不要把“计划/目录/单元测试成功”写成实验结果。
- 不要声称自动续作；当前客户端未验证定时/后台续作能力。
