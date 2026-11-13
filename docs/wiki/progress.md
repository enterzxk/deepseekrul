# 进度与下一步

> 权威源：docs/research_state.md（状态）、docs/decision_log.md（决策）。
> 本页只做阶段视图，不写未验证推测。

## 阶段视图

| 阶段 | 状态 | 说明 |
|---|---|---|
| A 方向探索 | 完成 | docs/idea_report.md Part 1 |
| B Idea 深化 | 完成 | PU-Net：冻结 IDA-PE + 尺度头 + 零初始化偏置头 |
| C 实验设计 | 完成 | Part 3 |
| D 实现设计 | 完成 | docs/implementation.md |
| E 编码 | 完成 | src/punet/、scripts/ |
| F 迭代 | 进行中 | 主矩阵 79/80；D001 关闭宽度路线，待新训练侧假设 |
| G 论文 | 进行中 | 稿件 + PDF 已编译；参考文献仍为占位，需 BibTeX |

## 下一步（按优先级）

1. 补 FD004 f4 s13，凑满 40 格（E001 补齐行）。
2. 由 L2（Codex）就"训练侧尺度-残差耦合"新假设给出取舍，写入 decision_log 后立项 H004。
3. 参考文献去占位：按 literature.md 登记表补 BibTeX，逐条核实。
4. 每完成一项，更新 research_state.md 与本页。

## 阻塞项

- reasoningEffort 实测 high（约定 max），属用户决定，未擅自改。

## 2026-11-09 02:xx · 本轮进展（主控线）

| 项 | 状态 | 产物 |
|---|---|---|
| D016 第 2 项：基线冻结 | ✅ 完成 | `docs/BASELINE_FREEZE.md`、`scripts/freeze_audit.py`（50/50）、D020 |
| D016 第 5、6 项：六件套草案 | ✅ 草案 v2 完成，**待 L2 立项门** | `docs/PROPOSAL_unit_coverage_identifiability.md` |
| 第一版「识别边界」框架 | ❌ 作废（文献占位） | `failures.md` F001 |
| 单位级失效跨 seed 稳定性 | ✅ 实测 | ICC 0.780、ρ +0.762、Jaccard 0.599（`scripts/engine_coverage_stability.py`） |
| 最近邻核实 | ✅ 10 篇 DOI 级核实 | `literature.md` 新增节 |

**下一步**：~~等 L2 立项门裁决~~ → **L2 已裁决：否决（D021）**。
- 不再沿 D018 立项；不启动 G5、不重跑实验。
- 已按 D016 第 8 项产出 `docs/INTERNAL_REPORT_baseline_audit.md`（内部技术报告 + 基线）。
- 若要做应用稿：先做具体期刊的贡献匹配审查，结论写 `decision_log.md`；不得当作投稿就绪认定。
- 失败登记：`failures.md` F001（第一版框架被占位）、F002（第二版计数对象混淆）。


- 2026-11-13：新N-CMAPSS机制原型已独立隔离，四缓存完成，作者代码来源保全；尚无正式训练收益证据。恢复见 docs/paper_b_ncmapss_response_20261113/。
