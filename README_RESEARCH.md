> **Lean GitHub snapshot:** the large row-level files (`data/ba_clean.csv`, `results/hof_eval_clean.csv`, etc.) are distributed in `release/full_data_bundle.tar.gz`; extract it at repo root before running `code/run_all.sh`. The server also has the full snapshot.

# README_RESEARCH — 复现与目录说明

**项目：** 固定起点、容量-only 的电池长程 SOH 轨迹外推（DeepSeek 独立线）  
**主报告：** `reports/DEEPSEEK_RESEARCH_REPORT_20261019_ZH.md`  
**边界：** 未使用 Codex `/home/yndx/zxk/rul` 的任何数据/权重/预测/测试；未使用 GPU。

## 目录

```
data/
  ba_clean.csv            101 电芯逐循环清洗 SOH/元数据
  ba_cell_summary.csv     逐电芯 EOL/循环/化学/工况
  celllist.json           Battery Archive query-29 原始 metadata
  ba_raw2.tar.gz          101 个 query-26 原始 JSON（公开仪表盘抓取）
  calce_manifest.json     CALCE zip URL/SHA256/解析状态
  calce_parsed/           12 颗 CALCE XLSX 的逐循环 CSV
results/
  final_metrics.json      主指标（机器可读）
  hof_eval_clean.csv      HOF 逐窗口预测/真值/pmean/fold
  hof_eol_clean.csv       HOF EOL 预测/误差/删失
  eol_compare_clean.csv   HOF/pmean/linear/exp/power EOL 对比
  unified_percell.csv     per-cell 曲线基线逐窗口
  gru_pred.csv            GRU 逐窗口
  resgru_pred.csv         residual GRU 逐窗口
  losource_hof_pred.csv   leave-one-source-out HOF
  per_cell_h300.csv       逐电芯 h=300 指标
code/
  clean_ba.py             Battery Archive JSON -> 清洗 CSV
  parse_ba.py             query-26 JSON -> 逐循环表
  scrape_full.py          本地 Edge/Playwright 抓取公开仪表盘（需本地联网）
  hof_eval_clean.py       HOF 主评估
  eol_compare_clean.py    EOL 对比
  train_gru.py            GRU 基线
  train_resgru.py         residual GRU
  losource_hof.py         leave-one-source-out
  final_metrics.py        汇总主指标
  parse_calce2.py / diag.py / diag2.py / compare_models.py  CALCE 原始审计
  run_all.sh              从 data/results 建工作目录并复算
figures/                  3 张图
reports/                  中文主报告
docs/NEXT_SESSION.md      恢复用最小上下文
```

## 环境

- Python 3.10+，CPU 即可：`numpy pandas scipy scikit-learn matplotlib torch`。
- `scrape_full.py` 额外需要 `playwright` 和本地 Edge/Chrome；服务器无外网，不能在服务器直接抓取。
- 不要 pip 改共享 base 环境；建议独立 venv/conda env。

## 一键复算（从项目根目录）

```bash
# 服务器 base python3.8 没有 pandas；使用独立环境（已在 a40_env 实测）
PYTHON=/home/yndx/miniconda3/envs/a40_env/bin/python bash code/run_all.sh
# 本机若有完整环境，也可直接：
# bash code/run_all.sh
```

`run_all.sh` 会创建 `work/`，把 `data/` 和 `results/` 的必要 CSV 链接进去，然后依次运行 HOF/EOL/汇总。GRU 训练可选（CPU 分钟级）：

```bash
cd work && python ../code/train_gru.py
cd work && python ../code/train_resgru.py
```

如果只想复算已有结果而不重训：

```bash
cd work && python ../code/final_metrics.py
# 服务器：
# cd work && /home/yndx/miniconda3/envs/a40_env/bin/python ../code/final_metrics.py
```

## 数据获取

- Battery Archive：公开 `Cycle Test Data` dashboard，query 29（cell list）+ query 26（capacity/energy decay）；`code/scrape_full.py` 用 Playwright + Edge 抓取。原始 JSON 已打包，避免未来 dashboard schema 变化导致不可复现。
- CALCE：`https://calce.umd.edu/data`；`data/calce_manifest.json` 有每个 zip 的 URL/SHA256。原始 zip 未上传（~900 MB），按 URL 重新下载即可。
- 文献：`tools/fetch_original_literature.py` 已本地运行并 SHA 校验；原件在 `literature/original_from_github/`。没有上传论文 PDF。

## 重要限制

- 101 颗电芯、69 颗观察 EOL80；EOL 指标条件于观察到 EOL，SNL LFP 大量删失。
- 5 折是开发集；没有冻结的独立确认集。
- leave-one-source-out 明确失败；当前 HOF 不是跨来源通用方法。
- BatteryMFormer 数据可能重叠；2025 Applied Energy 全文未获得。
