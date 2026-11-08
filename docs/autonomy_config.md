# 自治科研决策系统 · 配置与验证记录

> 配置日期：2026-11-08。本轮只做配置，未启动新实验。

## 一、实际侦察结果（非推测）

| 检查项 | 实际发现 | 路径/证据 |
|---|---|---|
| Codex 子代理工具 | ✅ 官方 `subagent_codex` 可用 | 本轮已成功激活 subagent `ea4b7283-5e50-47fb-be0d-f896853e399d` |
| `mcp__codex__codex` MCP | ❌ 未安装 | `~/.codex/config.toml` 的 `[mcp_servers]` 只有 `node_repl` |
| ARIS 运行时参数 | ❌ 不存在 | 17 个 `research-*` 技能只有 `SKILL.md`，全文无 `AUTO_PROCEED` / `human checkpoint` / `effort` / `reviewer` / `difficulty` |
| 工具审批模式 | ✅ 已自动放行 | `cordis.patch.yml` → `permission.presets.defaultPreset: danger-full-access`（`approval: never`） |
| 主模型 | `fuwuqi/deepseek-v4.1-flash` | 同上，`agent-default-model` |
| 思考强度 | ⚠️ `high`（约定要求 `max`） | 同上；未擅自修改，待用户决定 |

## 二、已做的修改

### 1. `AGENTS.md`（合并，未覆盖原有 8 节）

| 节 | 内容 |
|---|---|
| **§0.5（新增）** | 科研决策路由 L1 / L2 / L3 + 决策记录要求 |
| §2（改写） | 角色表增加"决策权"列；记录 `reasoningEffort` 实测差异 |
| §3（改写） | Codex 从"主控派工"改为"L2 决策者"；保留旧模式兼容说明 |
| **§9（新增）** | 研究状态持久化与恢复：恢复入口、更新义务、禁止事项 |
| **§10（新增）** | 与 ARIS 技能链配合：明确技能无运行时参数，人工检查点用 L2 决策替代 |

### 2. 新建文件

- `docs/decision_log.md` —— L2 决策记录（含 D001 与模板）
- `docs/research_state.md` —— 恢复入口：目标、已完成、已排除方向、待执行、环境事实

## 三、验证矩阵

| # | 验证项 | 状态 | 证据 |
|---|---|---|---|
| 1 | DSH 能调用 Codex | ✅ | `subagent_codex` 返回 `{"kind":"activation","subagentId":"ea4b7283..."}` |
| 2 | Codex 能返回清晰决策 | ✅ | `subagent_codex` 两次 product-error；改用 CLI `codex exec` 成功，返回 DECISION/REASONING/COUNTER-EVIDENCE/RISKS/NEXT ACTIONS 五段式 |
| 3 | DSH 能消费决策并继续 | ✅ | 已执行 D001 决策 C（宽度缩放扫描），结果写回 `decision_log.md` |
| 4 | ARIS 不因普通路线选择暂停 | ✅ | §10 + §0.5：技能无检查点参数，推进由 L1/L2 决定 |
| 5 | L3 操作仍受控 | ✅ | §0.5 L3 清单；Codex 无权授予新权限 |

## 四、已知缺口（不掩盖）

1. **`reasoningEffort` 是 `high` 而非 `max`**：修改桌面 profile 属于配置变更，按 §6 不擅自改，需用户决定。
2. **没有 `mcp__codex__codex`**：用户提到的 ARIS MCP Reviewer 未安装；当前用官方 `subagent_codex` 替代。
3. **`subagent_codex` 在本机不可用**：两次调用均 product-error（turn 阶段）。已改用官方 Codex CLI
   （`codex.exe exec --sandbox read-only`）作为 L2 通道，可同步返回结果。CLI 版本 0.167.0-alpha.3.2，
   ChatGPT 认证；日志显示 WebSocket 重连 5 次后回退 HTTPS，网络不稳可能是包装层失败原因。
4. **ARIS 无 AUTO_PROCEED**：用户要求的参数名在现有技能里不存在，已改为流程层实现，未虚构配置。

## 五、如何验证端到端

1. 本轮已发出 D001 决策请求（建模路线 A/B）
2. Codex 返回后，DSH 直接把决定写进 `decision_log.md` 并执行
3. 下一轮检查 `decision_log.md` 的 D001 是否有完整的"决定/理由/反对证据/风险/执行结果"
