# GPT 方法实现与实验结果

这里保存 Codex 准备的方法实现、实验结果和复算记录。

## 2026-11-06 筛选结果

**[先读 CCT 实验报告](codex_condition_contrast_20261106/results/screen_20261106/FINAL_REPORT_ZH.md)**

真实 C-MAPSS 筛选已完成 20 组。候选相对两子集各自最强窗口对照的平均 RMSE 改善 3.05%，在 FD004 上未超过同信息固定趋势。冻结判定为 `STOP_NO_GAIN`。本轮证据范围为 FD002、FD004 的 fold 0/2、seed 101 训练侧留出，涵盖 204 台发动机和 36,863 个查询。

[完整结果包](codex_condition_contrast_20261106/results/screen_20261106/CCT_20261106_screen_evidence.zip) · [四指标表](codex_condition_contrast_20261106/results/screen_20261106/codex_evidence_20261106/screen_metrics.csv) · [配对误差图](codex_condition_contrast_20261106/results/screen_20261106/codex_evidence_20261106/cct_paired_evidence.pdf) · [文献口径](codex_condition_contrast_20261106/results/screen_20261106/LITERATURE_PROTOCOLS_ZH.md)

## 冻结实现

当前候选是“工况内可学习趋势估计”：在已有 PE / FULL 上加入可微趋势层，检验长历史的收益能否归因于具体模型机制，并单独检查 NASA 尾部高估。

| 内容 | 文件 |
|---|---|
| 总体说明与目录 | [README](codex_condition_contrast_20261106/README.md) |
| 方法公式、已有文献与创新边界 | [METHOD_AND_NOVELTY](codex_condition_contrast_20261106/METHOD_AND_NOVELTY.md) |
| DeepSeek 的执行顺序、启动命令与停止条件 | [DEEPSEEK_HANDOFF](codex_condition_contrast_20261106/DEEPSEEK_HANDOFF.md) |
| 固定实验协议 | [protocol.json](codex_condition_contrast_20261106/protocol.json) |
| 核心模型 | [model.py](codex_condition_contrast_20261106/tim_contrast/model.py) |
| 自动实验流程 | [pipeline.py](codex_condition_contrast_20261106/tim_contrast/pipeline.py) |
| 实际验证范围 | [VALIDATION](codex_condition_contrast_20261106/VALIDATION.md) |
| 文件指纹 | [MANIFEST.sha256](codex_condition_contrast_20261106/MANIFEST.sha256) |

## 实现验证与实验范围

- 核心模型、严格前缀数据处理、嵌套 OOF 先验、训练、四指标统计、对照、自动判决和官方推断接口已实现。
- 20 项本地实现验证通过，包含合成数据、前后向计算、因果性、初始化和验收逻辑。
- 20 组真实训练的结果和复核记录见上方报告。三种子确认与官方测试的启动条件未达到。
- 公开结果包包含汇总、逐发动机统计、图和复算脚本。逐窗口预测、训练数据和检查点保存在实验服务器。

## 获取代码与复算

下载完整的 `gpt/codex_condition_contrast_20261106/` 子目录并保持包结构。实验服务器上的运行目录为

`/home/user/rulb/gpt/codex_condition_contrast_20261106/`

原交接文档记录了筛选启动命令。复算本轮指标使用报告末尾的导出脚本与已有完成记录。

源码上游固定为 `fa8acba70cbb17725fe32a54203b3b1252f158ac`；依赖的 PE 与数据处理源码已随包提供，可核对 [UPSTREAM.json](codex_condition_contrast_20261106/UPSTREAM.json)。
