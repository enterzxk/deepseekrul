# Paper B · DSPT-HC 研究目录

结构化健康状态（HI）与运行周期（Cycle）协同建模。**独立于 Paper A（区间预测/保形）线，互不覆盖。**

## 入口

| 文件 | 作用 | 谁维护 |
|---|---|---|
| [`TASK_BOOK.md`](TASK_BOOK.md) | 唯一执行契约：V0–V3、阶段 A–E、Gate A–D、边界、环境事实 | 郑总 |
| [`RESULTS_REGISTRY.md`](RESULTS_REGISTRY.md) | 实验登记规范：run_id、launch/close、审计清单 | 郑总 → 实验结果管理员执行 |
| [`RESULTS_INDEX.md`](RESULTS_INDEX.md) | 给人看的实验索引（Gate → run 表） | 实验结果管理员 |
| `registry/runs.csv` | 唯一台账 | 实验结果管理员 |
| `registry/launch/` `registry/close/` | tmh 写的实验声明 | tmh |
| 8 份交付文档 | 见 TASK_BOOK §10 | tmh |

## 当前状态（2026-11-11）

- 环境已核实：a40 可连（A40 49GB 空闲、torch 2.5.1+cu121）。
- **工作目录已迁到 2TB 盘**：`/media/sangfor/vdb/user/zxk/deepseek-rul`（可用 1.9T），
  系统盘 `/home/yndx/deepseek-rul` 降为只读参考。环境脚本 `/media/sangfor/vdb/user/zxk/env.sh`。
- 冒烟已通过：`tools_smoke.py` → SMOKE OK（params 117955、前向 403 ms、显存 43 MB）。
- 阶段：**A（文献）与 B（冻结骨干）准备中**，尚未启动任何训练。
- 已登记 run：0。

## 铁律

1. 不碰 `paper_b/models/`、`paper_b/training/`、`paper_b/configs/protocol.json`、`src/candidates/`、Paper A 任何文件。
2. 不用官方 TEST 做任何设计选择；冻结前不产出 `test_*` 指标。
3. 没有 run_id 的结果不算结果；文档数字必须能从 `experiments/hc/<run_id>/` 复算。
4. 不编造数据、引用、日志。没跑就写"未跑"。
