# Gate A 第一波：24 格清单与并行方案

状态：待发车（未启动任何训练）
最后更新：2026-11-12 00:35 (+08)
基线：v9（代码 `d1e0527`，锁 `6c405ad`）
依据：任务书 §5.3（6 变体 × 4 子集 × seed 101 = 24 run，只判方向）

## 1. 变体集

`{A0, A1, A2, A3p, A4, A5}`。**A3p ≡ V0-NoHI**（同配置），只跑一次、V0 表双向引用。
A3a 是敏感性臂，不进第一波（已有 0030Z）；A3b/A3c 不在第一波。

## 2. 初始化策略（公平性的关键）

6 个变体的骨干必须同权重，否则跨变体差里混入初始化噪声（牢莫在 A3a/A3p 上已证）。
做法：以 `init_v9_A3p_s101.pt`（sha256 `336074a6…`）为**骨干基准**，为每个变体生成一份
自己的 init 文件：骨干键从 A3p 复制，变体特有的循环模块保留自身初始化（形状不同，无法共享）。

| 变体 | 循环模块 | init 文件 |
|---|---|---|
| A3p | `pe_proj` Linear(8,64) | `experiments/hc/tmp/init_v9_A3p_s101.pt`（现成，sha `336074a6…`） |
| A1 | 无 | `wave1_init_A1_s101.pt`（仅骨干） |
| A2 | `ScalarCycle` Linear(1,64) | `wave1_init_A2_s101.pt`（骨干 + 自身 Linear(1,64)） |
| A4 | `ScalarCycle(log)` Linear(1,64) | `wave1_init_A4_s101.pt` |
| A5 | `SplineAge` Linear(9,64) | `wave1_init_A5_s101.pt` |
| A0 | `CycleOnly`（无骨干） | 无 init（自身 seed 初始化，参考线） |

生成器放 `experiments/hc/tmp/make_wave1_inits.py`（不在 15-blob 锁内，不碰 `paper_b/hc/`），
产物 sha256 写入各自 manifest 的 `init_sha256`。

## 3. 24 格清单

run_id 规则：`HC-C-{variant}-{subset}-s101-W1`（确定性、可预登记、天然防重名）。

| # | variant | subset | seed | init | run_id |
|---|---|---|---|---|---|
| 1 | A0 | FD001 | 101 | — | HC-C-A0-FD001-s101-W1 |
| 2 | A0 | FD002 | 101 | — | HC-C-A0-FD002-s101-W1 |
| 3 | A0 | FD003 | 101 | — | HC-C-A0-FD003-s101-W1 |
| 4 | A0 | FD004 | 101 | — | HC-C-A0-FD004-s101-W1 |
| 5 | A1 | FD001 | 101 | wave1_init_A1 | HC-C-A1-FD001-s101-W1 |
| 6 | A1 | FD002 | 101 | wave1_init_A1 | HC-C-A1-FD002-s101-W1 |
| 7 | A1 | FD003 | 101 | wave1_init_A1 | HC-C-A1-FD003-s101-W1 |
| 8 | A1 | FD004 | 101 | wave1_init_A1 | HC-C-A1-FD004-s101-W1 |
| 9 | A2 | FD001 | 101 | wave1_init_A2 | HC-C-A2-FD001-s101-W1 |
| 10 | A2 | FD002 | 101 | wave1_init_A2 | HC-C-A2-FD002-s101-W1 |
| 11 | A2 | FD003 | 101 | wave1_init_A2 | HC-C-A2-FD003-s101-W1 |
| 12 | A2 | FD004 | 101 | wave1_init_A2 | HC-C-A2-FD004-s101-W1 |
| 13 | A3p | FD001 | 101 | init_v9_A3p | HC-C-A3p-FD001-s101-W1 |
| 14 | A3p | FD002 | 101 | init_v9_A3p | HC-C-A3p-FD002-s101-W1 |
| 15 | A3p | FD003 | 101 | init_v9_A3p | HC-C-A3p-FD003-s101-W1 |
| 16 | A3p | FD004 | 101 | init_v9_A3p | HC-C-A3p-FD004-s101-W1 |
| 17 | A4 | FD001 | 101 | wave1_init_A4 | HC-C-A4-FD001-s101-W1 |
| 18 | A4 | FD002 | 101 | wave1_init_A4 | HC-C-A4-FD002-s101-W1 |
| 19 | A4 | FD003 | 101 | wave1_init_A4 | HC-C-A4-FD003-s101-W1 |
| 20 | A4 | FD004 | 101 | wave1_init_A4 | HC-C-A4-FD004-s101-W1 |
| 21 | A5 | FD001 | 101 | wave1_init_A5 | HC-C-A5-FD001-s101-W1 |
| 22 | A5 | FD002 | 101 | wave1_init_A5 | HC-C-A5-FD002-s101-W1 |
| 23 | A5 | FD003 | 101 | wave1_init_A5 | HC-C-A5-FD003-s101-W1 |
| 24 | A5 | FD004 | 101 | wave1_init_A5 | HC-C-A5-FD004-s101-W1 |

## 4. 并行方案

- 显存：A40 49140 MiB。实测单 run 约 6.9 GB，双 run 并行时 14.4 GB（≈7.2 GB/run）。
  按郑总给的保守值 14 GB/run，3 路 = 42 GB < 49 GB，余 ~7 GB。
- **并发度 3**（`cuda:0`）。每路独立进程，`cudnn.benchmark=True` 保持默认。
- 墙钟：单 run 实测 200–440 s（并发时变慢）。3 路并行按 6–9 min/run 估，
  24 格 / 3 = 8 波 × ~7 min ≈ **60–75 min**，加启动开销约 80 min。
- 调度：`experiments/hc/tmp/run_wave1.sh` 用 3 个槽位的信号量（`flock` 或 FIFO），
  逐格调用 `run_hc`，每格独立日志 `logs/wave1/<run_id>.log`。

## 5. 失败与重试（不污染台账）

1. `run_hc` 在异常路径也会写 `close.json`（`status=failed`），launch/close 成对，**不删**。
2. 失败格重跑一律**换新 run_id**：`…-W1r2`、`…-W1r3`，绝不复用旧 id（`O_EXCL` 会拒）。
3. 台账唯一键 = `(baseline_version, variant, subset, seed, init_sha256)`；同一格出现多个完成 run 时，
   保留第一个 `ok`，其余标 `duplicate`。
4. 重试只针对失败格；成功格不重跑。若某格连续 2 次失败，停该格并报错，不无限重试。
5. 发车前预检：`code_lock --check` 绿、`git status -- paper_b/{hc,training,models,configs}` 空、
   GPU 空闲、`init` 文件 sha256 已记录。

## 6. 未做

- 未生成 init 文件（等发车指令）。
- 未启动任何训练。
