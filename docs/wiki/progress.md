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
