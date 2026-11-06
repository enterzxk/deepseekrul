# 工况内可学习趋势估计：交给 DeepSeek 的可执行实验包

状态：核心模型、训练、评价、消融和交接已实现；尚未在 C-MAPSS 上训练，不能宣称性能提升或创新已成立。

这是接在现有 PE / FULL 先验后的候选结构。从 GitHub 获取完整的 `gpt/codex_condition_contrast_20261106/` 子目录，例如同步到：

`/home/user/rulb/gpt/codex_condition_contrast_20261106/`

使用已有科研环境和已有 C-MAPSS 数据，不需要重新克隆基线、重新寻找模型、安装新骨干或修改全局环境。上游两个依赖源码已随包固定，版本为 `fa8acba70cbb17725fe32a54203b3b1252f158ac`。来源及 SHA256 在 `UPSTREAM.json`。

## 已完成与接下来要做的事

Codex 已完成：最近邻方法核查；候选方法公式与边界；核心可训练模型；严格前缀数据处理；FULL 嵌套交叉拟合；初始化修复；四指标、逐发动机与尾部统计；有界队列；自动筛选与确认；冻结后官方推断和评分入口；科学性质与接口测试。

DeepSeek 接手：运行队列 → 根据 JSON 判决继续或停止 → 导出实际数据和图表 → 整理同口径文献比较。详细顺序在 `DEEPSEEK_HANDOFF.md`，方法论在 `METHOD_AND_NOVELTY.md`。不要重新设计一批网络，不要为失败臂加种子或改阈值。

## 结构

- `tim_contrast/model.py`：工况内加权趋势层、PE 接入、单独的尾部损失控制。
- `tim_contrast/data.py`：训练侧划分、严格因果窗口、嵌套 FULL 先验、可复用数据缓存。
- `tim_contrast/run.py`：单次 80 轮完整训练，固定末轮，先设随机种子再建模型。
- `tim_contrast/queue.py`：有预算的多卡分工队列，完成文件去重，错误即停。
- `tim_contrast/analyze.py`：同键同标签比较，发动机级区间，机器可读继续/停止判决。
- `tim_contrast/official.py`：通过开发验收后，先封存预测，再单独读取官方标签评分。
- `tests/test_contract.py`：因果性、工况偏置抵消、已知斜率恢复、初始化一致性、反向传播、评分和短轨迹覆盖。
- `protocol.json`：冻结预算、指标与阈值。

## 启动第一阶段

推荐直接按 `DEEPSEEK_HANDOFF.md` 的单条 pipeline 命令启动，它会管理两张卡、结果判决和后续阶段，避免频繁人工查询。以下命令仅用于需要分阶段运行时。

在包目录内分别启动下面两个长任务，各自保留进程号与日志；不要反复提交：

```bash
python -m tim_contrast.queue --phase screen --gpu 0 --shard 0 --data-root /home/user/rulb/data/CMAPSS --wall-hours 8
python -m tim_contrast.queue --phase screen --gpu 1 --shard 1 --data-root /home/user/rulb/data/CMAPSS --wall-hours 8
```

两条命令分别占用一张可用卡，每张卡串行训练自己的子集。当前项目已有训练时先等其完成，不停止其他项目的任务。队列日志在 `outputs/queue_logs/`，模型及预测在 `outputs/FD00x/`。运行入口不会读取官方 test/RUL 文件。

两卡结束后执行一次：

```bash
python -m tim_contrast.analyze --stage screen
```

按返回的 `next` 继续，具体命令见交接文档。第一阶段固定为 FD002/FD004、fold 0/2、seed 101、5 个臂，共 20 次完整预算训练。它是开发筛选，不是性能确认。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

本地验证只检验数学性质、数据与训练接口，不是 C-MAPSS 实验；`VALIDATION.md` 记录实际结果。GPU 运行器未在本机执行真实训练。开发数据和官方测试都已被历史项目使用，不能声称新的独立盲测。

工况偏置抵消只对趋势分支及其指定假设成立，不适用于整网，也不代表能区分传感器线性漂移与真实退化。回归残差只是描述量，不是校准后的预测不确定度。尾部损失是训练控制，不能独立包装成创新。

`outputs/`、`cache/`、检查点和原始数据均不应直接公开上传。论文用结果应另外导出汇总、代码、协议和必要的匿名预测。
