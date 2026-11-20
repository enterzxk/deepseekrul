# LTAG v3 项目最终报告（截至 2026-11-21）

> 本报告汇总 2026-11-20 至 11-21 的 LTAG v3 执行结果。
> **test 未重跑；Phase 2/5 未启动。**

## 1. 一句话结论

在冻结的 C-MAPSS B0 骨干上，**LTAG 在 end-of-life test 上有效（15/15，mean +0.391）**，
但**不是所有 cut 上的最优聚合器**：简单 context 校准（K1-Calib / LS-ctx）在 5-cut
validation 均值上 4/4 胜出，其优势集中在早期 cut；到 0.85–0.95，LTAG 追平或反超。
**RHSE-TSH 被公平时序基线否决，已从论文删除。**

## 2. 关键结果

### 2.1 官方 test（冻结，15/15）

| subset | seeds | K1 | LTAG | Δ(K1−LTAG) | wins |
|---|---:|---:|---:|---:|---:|
| FD001 | 6 | 13.421 | 12.995 | +0.426 | 6/6 |
| FD002 | 3 | 11.514 | 11.352 | +0.162 | 3/3 |
| FD003 | 3 | 12.979 | 12.323 | +0.656 | 3/3 |
| FD004 | 3 | 13.031 | 12.710 | +0.321 | 3/3 |
| mean | — | — | — | **+0.391** | **15/15** |

### 2.2 Validation 强对照（5-cut engine-mean）

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| K1 | 12.094 | 11.527 | 12.793 | 13.608 |
| LTAG | 11.334 | 11.210 | 12.372 | 13.522 |
| K1-Calib | 10.931 | 10.970 | 12.129 | 12.839 |
| LS-ctx | 10.150 | 10.604 | 12.033 | 13.080 |

- K1-Calib/LS 4/4 胜 LTAG；LS 只在 50–56% 发动机上赢，FD001 中位 Δ 为正。
- 优势集中在 0.40–0.55；0.95 上 LTAG 在 FD002/FD004 最好，FD001/FD003 第二且距最好 ≤0.28。

### 2.3 RHSE-TSH 公平基线（3-seed mean recovery）

| subset | fault | R3 | R5b | R5m | 结论 |
|---|---|---:|---:|---:|---|
| FD001 | bias | 0.799 | 0.793 | 0.821 | R5m 更好 |
| FD001 | drift | 0.856 | 0.868 | 0.925 | R5m 更好 |
| FD001 | stuck | 0.737 | −0.695 | −0.202 | R3 更好 |
| FD003 | bias | 0.908 | 0.927 | 0.939 | R5m 更好 |
| FD003 | drift | 0.916 | 0.853 | 0.921 | 持平 |
| FD003 | stuck | 0.605 | −0.179 | −0.288 | R3 更好 |

- 只有 stuck 过门（≥0.10，2/3 faults）；bias/drift 不通过 → **RHSE-TSH 删除**。

## 3. Gate 结果

| Gate | 结果 |
|---|---|
| Gate 1：LTAG vs K1-Calib / Global-g / Linear-Stack | **FAIL**：G1.1 0/4，mean −0.393；G1.2 LS 4/4 胜；G1.3 PASS；G1.4 2/4 |
| Gate 2：跨骨干 | **未启动**（Gate 1 失败） |
| RHSE 公平基线门 | **FAIL**（只有 stuck） |

## 4. 论文状态

- 7 页 IEEE journal 初稿：`paper/main.pdf`。
- 8 节 + 附录 A；5 表；3 图；9 条已核实引用。
- Claim audit：本地 15 组 PASS，verdict **WARN**（无独立 cross-model 评审）。
- 未投稿（L3 未授权）。

## 5. 未完成 / blockers

1. **独立评审**：Codex 不可用（`mcp__codex__codex` WinError 2；`codex` 不在 PATH）。
   `PENDING-L2`。
2. **投稿**：需 L3 授权。
3. **Phase 2/5**：Gate 1 失败，不启动；test 封存。
4. **磁盘**：`/home` 98%（96G）；不做新大实验。

## 6. 决策选项（A/B/C）

| 选项 | 内容 | 评价 |
|---|---|---|
| A | LS-ctx 主方法 + 一次性 test | 证据弱：胜率 50–56%，早期 cut 优势，test-like cut 不占优 |
| **B** | LTAG test 主 + 分 cut/后校准对照 | **当前证据最支持；论文已按 B 写** |
| C | 停止写负结果/边界报告 | 可接受；适合方法创新要求低的 venue |

## 7. 产物索引

- [LTAGV2_INDEX.md](LTAGV2_INDEX.md)
- [LTAGV2_REVIEW_FAQ.md](LTAGV2_REVIEW_FAQ.md)
- [PAPER_PLAN.md](../PAPER_PLAN.md)
- [paper/main.pdf](../paper/main.pdf)
- [PAPER_CLAIM_AUDIT.md](../paper/PAPER_CLAIM_AUDIT.md)
- [SUBMISSION_CHECKLIST.md](../paper/SUBMISSION_CHECKLIST.md)

**最终判断**：LTAG 是一个 **end-of-life 有效的推理期门控**，不是通用 SOTA；
RHSE 是负结果。论文已按这一边界写完；下一步只缺独立评审与投稿授权。
