# GPT 方法实现与 DeepSeek 实验入口

这里保存 Codex 准备的独立方法实现和实验交接，供 DeepSeek 直接从 GitHub 获取。

## 当前入口

**[先读 DeepSeek 执行交接](codex_condition_contrast_20261106/DEEPSEEK_HANDOFF.md)**

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

## 当前证据状态

- 核心模型、严格前缀数据处理、嵌套 OOF 先验、训练、四指标统计、对照、自动判决和官方推断接口已实现。
- 20 项本地实现验证通过，包含合成数据、前后向计算、因果性、初始化和验收逻辑。
- **尚未执行这套候选的真实 C-MAPSS 训练；性能增益、新颖性和 TIM 投稿就绪均未确认。**
- 本目录不包含训练数据、检查点、服务器任务日志或凭据。

## DeepSeek 获取与运行

下载完整的 `gpt/codex_condition_contrast_20261106/` 子目录，保持包结构；仅下载单个模型文件无法运行。若现有服务器研究目录不是 Git 仓库，将这个子目录单独同步到：

`/home/user/rulb/gpt/codex_condition_contrast_20261106/`

然后按照交接文档使用现有环境与 `/home/user/rulb/data/CMAPSS` 数据，一次启动自动流程。沿用原截止时间与预算，按机器生成的继续/停止判决执行，不重新发散设计一批模型，不把实现验证写成性能成功。

源码上游固定为 `fa8acba70cbb17725fe32a54203b3b1252f158ac`；依赖的 PE 与数据处理源码已随包提供，可核对 [UPSTREAM.json](codex_condition_contrast_20261106/UPSTREAM.json)。
