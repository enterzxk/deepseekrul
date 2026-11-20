# LTAG v2 论文：审稿人问题预案（2026-11-21）

> 用途：投稿前自检/回复审稿人。所有回答只引用已落盘证据；**test 未重跑**。

## Q1. “LTAG 不是 SOTA，validation 上 K1-Calib/LS 更强。”

**答**：论文不 claim SOTA。Table 2 明确报告 K1-Calib/LS 在 5-cut 均值上 4/4 胜
LTAG；Table 3 显示优势集中在 0.40–0.55。cut 0.95 上 LTAG 在 FD002/FD004 最好，
FD001/FD003 第二且距最好 ≤0.28。论文的贡献是 **cut-dependent 证据**，不是“最强聚合器”。

## Q2. “为什么 test 上 LTAG +0.39，但 validation 均值上输？”

**答**：test 是 end-of-life 单窗口，接近 cut 0.95；validation 均值被早期 cut
（误差大）主导。Table 3 + Figure 2 显示 0.85–0.95 的 crossover。两者不矛盾。

## Q3. “LS-ctx 只在 50–56% 发动机上赢，凭什么说它强？”

**答**：论文没有说它“广泛更强”。Figure 3 明确报告 10/20、28/52、10/20、28/50，
FD001 中位 Δ 为正；它的均值增益来自少数高误差发动机，是尾部校准。论文把它作为
“早期 cut 的强对照”，不是主方法。

## Q4. “test 是不是被看过？”

**答**：是历史观察过的 B0 test；论文 Limitations 明确写“one-shot confirmatory,
not a pristine blind test”。没有用 test 做任何选择；配置在 test 前冻结。

## Q5. “只有一个 backbone，怎么推广？”

**答**：论文不 claim cross-backbone。所有结论限定在冻结 B0；跨骨干是 future work，
也是原计划 Gate 2 未启动的原因（Gate 1 失败）。

## Q6. “为什么不在 test 上跑 LS-ctx？”

**答**：test 封存；Gate 1 失败后不启动 Phase 2/5。LS-ctx 只有 validation 证据，
论文明确标注；没有用 test 补跑。

## Q7. “RHSE-TSH 为什么删了？”

**答**：公平 R5b/R5m 在 bias/drift 上追平/超过 R3；只有 stuck 过门（Table 4）。
旧 R5 是 last-value 稻草人。删除是负结果，不是实现失败。

## Q8. “15/15 是不是只报赢的 seed？”

**答**：15/15 是全部 seed-subset 组合：FD001 6/6、FD002 3/3、FD003 3/3、FD004 3/3；
Table 1 同时给 mean 和 per-subset Δ。没有挑 seed。

## Q9. “engine-mean 和 pooled 是否混用？”

**答**：没有。Table 1 test 一台一个窗口，二者重合；validation 的 Table 2/3 用
engine-mean，pooled 在 `SHARED_R0.json` 单独报告；论文 Methods 明确禁止相减。

## Q10. “为什么只报 0.95 cut 的 bootstrap？”

**答**：0.95 是最接近 test 的 validation cut；完整 5-cut 表在 Table 3。bootstrap
是配对 engine-level，10,000 次；多 seed 先在 engine 内平均。

## Q11. “论文的创新点到底是什么？”

**答**：三点：(1) cut-controlled 评估协议；(2) 学习门控在 end-of-life 有效、但在
all-cut 均值上会被简单校准击败的实证边界；(3) 公平时序基线删除 RHSE 负结果。
不是新网络结构，也不是 SOTA。

## Q12. “为什么 Related Work 只有 9 篇？”

**答**：当前只保留已核实、与论点直接相关的条目；投稿前会补 C-MAPSS 骨干与
calibration 的完整引用（需逐条核实 DOI，不能用记忆生成）。

## 证据索引

- [LTAGV2_INDEX.md](LTAGV2_INDEX.md)
- [LTAGV2_TABLES.md](LTAGV2_TABLES.md)
- [LTAGV2_CUTWISE.md](LTAGV2_CUTWISE.md)
- [LTAGV2_P4_RHSE_DECISION.md](LTAGV2_P4_RHSE_DECISION.md)
- [paper/PAPER_CLAIM_AUDIT.md](../paper/PAPER_CLAIM_AUDIT.md)
