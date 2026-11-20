# LTAG v3 证据索引（2026-11-21）

> 本文件是入口；详细内容见各链接。**test 未重跑；Phase 2/5 未启动。**

## 1. 计划与决策

| 文档 | 内容 |
|---|---|
| `ltag_v3_execution_plan.md` | 用户提供的 v3 计划（Downloads） |
| [PAPER_PLAN.md](../PAPER_PLAN.md) | 论文计划（选项 B，8 节，Claims C1–C6） |
| [LTAGV2_DECISION_PACKAGE.md](LTAGV2_DECISION_PACKAGE.md) | A/B/C 选项、LS-ctx 证据、L2 状态 |
| [LTAGV2_TEST_EVIDENCE.md](LTAGV2_TEST_EVIDENCE.md) | 冻结 test 审计与 validation 对照 |
| [LTAGV2_CUTWISE.md](LTAGV2_CUTWISE.md) | 分 cut 与 cut0.95 bootstrap |
| [LTAGV2_P4_RHSE_DECISION.md](LTAGV2_P4_RHSE_DECISION.md) | RHSE-TSH 删除决定 |

## 2. Phase 0–4 证据

| 阶段 | 文档/数据 |
|---|---|
| Phase 0 口径 | [LTAGV2_P0_RECONCILE.md](LTAGV2_P0_RECONCILE.md)、`results/SHARED_R0.json`、`paper_b/eval/metrics_v2.py` |
| Phase 1 强对照 | [LTAGV2_P1_BASELINES.md](LTAGV2_P1_BASELINES.md)、`results/ltagv2_p1_FD00*.json`、`results/ltagv2_p1_*.preds.csv` |
| Phase 1b 集成/消融 | `results/ltagv2_p1b_FD00*.json`、`paper_b/eval/ltagv2_phase1b.py` |
| Phase 3 机制 | [LTAGV2_P3_MECHANISM.md](LTAGV2_P3_MECHANISM.md)、`results/ltagv2_p3_FD00*.json` |
| Phase 4 RHSE | `experiments/phase4/logs/`、`results/phase4.json`、`paper_b/modules/phase4_r5b.py` |

## 3. 论文资产

| 资产 | 路径 |
|---|---|
| 初稿 | [paper/main.tex](../paper/main.tex)、`paper/sections/0–8`、`A_appendix.tex` |
| 表格 | `tables/tab1_test.tex` … `tab5_gates.tex` |
| 图 | `figures/fig1_cutwise.pdf`、`fig2_test.pdf`、`fig3_winrate.pdf`；`figures/latex_includes.tex` |
| Claim audit | [paper/PAPER_CLAIM_AUDIT.md](../paper/PAPER_CLAIM_AUDIT.md)、`.json`（WARN，本地 15 组 PASS） |
| 投稿清单 | [paper/SUBMISSION_CHECKLIST.md](../paper/SUBMISSION_CHECKLIST.md) |

## 4. 关键数字

- **Test**：LTAG 15/15，mean **+0.391**；FD001 +0.426、FD002 +0.162、FD003 +0.656、FD004 +0.321。
- **Validation 均值**：K1-Calib/LS 4/4 胜 LTAG；LS-ctx 只在 50–56% 发动机上赢。
- **Cut 0.95**：LTAG 在 FD002/FD004 最好；FD001 K1-Calib、FD003 LS 略好；LTAG 距最好 ≤0.28。
- **RHSE**：只有 stuck 过门；bias/drift 上 R5b/R5m ≥ R3 → 模块删除。
- **Gate 1**：FAIL（G1.1 0/4，mean −0.393；G1.2 LS 4/4 胜；G1.3 PASS；G1.4 2/4）。
- **Gate 2**：未启动。

## 5. 未完成 / blockers

- PDF 编译：本机与 lab3090 均无 `pdflatex`/`latexmk`。
- 独立 claim audit / 科学评审：Codex `WinError 2`，**PENDING-L2**。
- 投稿：需 L3 授权。
- 磁盘：`/home` 98%（96G 可用）；不做新大实验。
- 路线：A/B/C 未定；当前论文实现 B。
