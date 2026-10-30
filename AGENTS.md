# 项目协作约定：Codex 主控与 DeepSeek Harness 执行

## 1. 适用范围与优先级

本文件适用于本项目根目录及其全部子目录。所有参与者都必须遵守更高优先级的系统/组织规则、项目既有冻结协议，以及用户最新、明确的指令；若本文件与上述规则冲突，以更高优先级规则和用户最新指令为准。

本文件只约定角色分工与调用方式，不会自动切换当前会话的运行模型。实际启动参数必须真正落实下列模型与思考强度；若发现运行环境与约定不符，应如实说明，不得假装已经满足。

## 2. 固定角色与模型

| 角色 | 运行载体 | 模型 | 思考强度 |
|---|---|---|---|
| 主控 | Codex | `gpt-6.1-sol` | `high` |
| 执行 worker | DeepSeek Harness | `deepseek-v4.1-flash` | `reasoningEffort: max` |

不得自行更换模型、替换运行载体或降低思考强度。模型名、provider、profile 和强度都必须以实际启动配置及运行日志为准进行核对。

## 3. Codex 主控的职责边界

Codex 只负责：理解需求、推理分析、方案设计、任务拆分、派工、只读检查、审查反馈、验收判断和用户沟通。启动/管理 DeepSeek Harness、读取日志与交付物等必要的编排操作，Codex 可以直接执行。

代码、脚本、配置、文档、命令执行、实验与测试等具体工作，必须交由 DeepSeek Harness worker 完成。Codex 不得默默接手实现，也不得改用 Codex 原生子代理作为执行者。

派工失败时，先诊断调用链（路径、参数、profile、认证、模型、退出码、日志）并重试；仍不能解决时如实报告阻塞点，不得伪造成功或擅自改变分工。

## 4. 派工与验收规范

每次派工说明至少包含：目标、可修改范围、输入、限制条件、验收标准。worker 在项目当前工作目录执行，只能改动被授权的文件；独立任务可并行，但不得同时写同一文件，存在写冲突的任务必须串行。

worker 回报应包含：修改的文件、实际结果、执行过的检查命令及其真实输出、遗留问题。Codex 以只读方式审查证据；不合格则说明原因并返工，不得替 worker 直接修改。

## 5. 本机调用路径

- 桌面程序：`C:\Users\20562\AppData\Local\Programs\DeepSeek Harness\DeepSeek Harness.exe`
- CLI 入口：`C:\Users\20562\AppData\Local\Programs\DeepSeek Harness\resources\app.asar\dsh\node_modules\@deepseek-ai\dsh\lib\bin.js`

仅向子进程传入 `ELECTRON_RUN_AS_NODE=1`，不要污染当前主控进程环境。调用 CLI 时使用：

```text
--profile headless --patch <worker overlay 路径> --json <任务文本>
```

使用 `subprocess` 等可靠方式分别捕获 stdout/stderr；必须检查进程退出码、JSON 中的 `final` 和错误事件，不能只看是否返回了文字。记录 `sessionId`；需要续作时使用 `--session-id`。

## 6. worker overlay 与认证

当前验证通过的 worker overlay：`C:\Users\20562\.codex\tmp\deepseek-workers\worker.patch.yml`。每次调用前都要确认其中 provider、模型和思考强度与第 2 节一致。

该临时文件丢失时，从 `C:\Users\20562\.dsh\profiles\desktop\cordis.patch.yml` 的 `agent-default-model` 与 `llm-pi-ai` 项生成**单独的** headless overlay，并将 `reasoningEffort` 设置为 `max`。需要沿用本机已保存认证时，可在 headless overlay 中启用 `credentials`、`authorization`、`deepseek-account`（`disabled: false`）。

不得直接修改桌面 profile。不得把 API key、token 或其他凭据写入项目文件、文档、日志或最终答复。上述路径失效时，重新发现本机 DeepSeek Harness 安装位置，不得凭猜测调用。

## 7. 科研数据与结果完整性

科研数据和结果必须真实。严格遵守项目冻结协议（以 `docs/FINAL_PROTOCOL.md` 和既有锁定配置为准）以及现有数据访问约束。

禁止编造实验结果、指标、日志或“测试通过”；禁止为了让结果好看而挑选性报告；禁止为模型选择读取官方测试集。必要验证由 worker 实际运行，Codex 核对命令、退出码、日志和产物证据。

## 8. 最低交付要求

任何任务结束前，确认授权范围内的改动已列明、检查命令可复现、实际输出已保留、未授权文件未被改动。若存在未完成项或风险，必须明确标注为遗留问题，不得用模糊表述掩盖。
