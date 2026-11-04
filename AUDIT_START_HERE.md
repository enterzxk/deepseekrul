# 审计入口 — IDA-PE / IEEE TIM（2026-11-04 23:25）

本目录是 3090 计算工作区（非 git 仓库；git 主仓库在 GitHub enterzxk/deepseekrul，
本地 Windows 侧 `C:\Users\20562\Desktop\deepseek-rul\`）。审计从这里开始。

## 1. 读这三份（按序）

1. `docs/tim_idc/HANDOFF.md` —— 全项目状态交接（判决数字/下一步/红线）
2. `gates.json` —— 预注册冻结判据原文（**审计基准**；与 GitHub 同步）
3. `docs/FINAL_PROTOCOL.md` —— 早期冻结协议（P 线/R3.1 历史纪律）

## 2. 判决 JSON（机器可读，全部由脚本生成、非手写）

| 判决 | 文件 |
|---|---|
| Gate 0 六种子 | `experiments/ida/gate0/gate0_verdict_s{101,202,303,42,67,123}.json` |
| Gate 1 | `experiments/ida/gate1/gate1_verdict.json` + `h2_weights.json` |
| Module B/C | `experiments/ida/moduleBC/modulebc_verdict.json` |
| FD003/004 五臂 | `experiments/ida/gate0/fd34_summary.json` |
| E22a 对齐 | `experiments/ida/e22a_align/align_summary.json` |

人类可读报告在 `docs/tim_idc/`：gate0_report / gate0_multiseed /
gate1_report / modulebc_report / four_subset_results / protocol_calibers。

## 3. 复算验证命令（只读，随时可跑）

```bash
PY=/home/user/miniconda3/envs/pmv/bin/python
# Gate 0 任一种子判决（引擎级 bootstrap 5000）
cd /home/user/rulb && $PY scripts/summarize_gate0.py --seed 101 --out /tmp/verify_g0.json
# Gate 1
$PY scripts/summarize_gate1.py
# Module B/C
$PY scripts/summarize_modulebc.py
# FD003/004 与 E22a 对齐
$PY scripts/summarize_subsets.py --base experiments/ida/gate0 --subsets FD003 FD004 \
  --arms M0 M1 M1p M1_shuf M1_const --contrasts 'M1,M0' 'M1_shuf,M1' 'M1_const,M1'
```

## 4. 代码完整性（2026-11-04 核验）

本地仓库与远程工作区的关键脚本 sha256 **归一化（LF）后 9/9 一致**
（run_ida / run_moduleBC / run_e22a_align / summarize_gate0 / summarize_gate1 /
summarize_modulebc / summarize_subsets / deg_modules.py / nested_prior.py）。
本地侧因 Windows 检出为 CRLF，整文件 hash 不同属行尾差异，非代码漂移。
原始运行均在远程本目录执行，`experiments/ida/**/metrics.json` 内含
`command` 字段可追溯每次运行的完整命令行。

## 5. 审计红线

- 判据冻结在 gates.json：不得以"换个口径更好看"为由修改或重判
- 禁止窗口级 bootstrap（一律引擎聚类 5000 次）
- 官方 test 尚未解封（历史上 E21/E22a 用过一次旧口径，见
  `docs/tim_idc/protocol_calibers.md`，两套数字严禁混表）
- 原始预测数据：`experiments/ida/gate0/FD00X/{ARM}_f{k}_s{seed}/val_predictions.csv`
  （含 mu_T/sigma_T 列，可直接复算 pooled RMSE）

## 6. 当前硬件状态

双卡空闲，无运行中队列。队列日志：`logs/{gate0B,gate1,gate0C,gate3fd,moduleBC,e22a}/`。
