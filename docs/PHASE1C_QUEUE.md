# Phase 1c 队列：RHSE-TSH 3-seed 单工况恢复 + EOL FPR

> 日期：2026-11-19（23:55 修订）。脚本：`scripts/queue_phase1c.sh`。
> 协议：`results/FREEZE_PHASE1C.json`。**test 全程封存。**

## 1. 修订要点

- **R5 clean 泄漏已修**：last-value 现在用观测到的 `x_f`，不再用 clean `x`。
  修复后 FD001 drift 的 R5 从 0.984 掉到 **0.313**，R3 为 **0.861**；
  “last-value 在晚漂移更好”的旧结论**撤回**。
- **EOL FPR** 改为 cut=0.95、sensors 0/5/10，同时报告 per-sensor-cycle、
  any-sensor-cycle、any-window。FD001 为 0.24% / 0.71% / 5.0%；FD003 为 0% / 0% / 0%。
- **stuck 只跑 mag=3**：z-space stuck 与 mag 无关，mag=2/5 是重复格。
- **60 个 RUL 作业 + 2 个 EOL 作业 = 62**（不是 78）。
- recovery 在 `R1−R0 ≤ 0.05` 时输出 `n/a`，不再出现 1e9 爆炸。
- `--gamma` 默认 1.30；q 加 floor；`u` 用 `nanmax`；R4 的 hold 起点用真实 onset。

## 2. 网格

| 轴 | 取值 | 数量 |
|---|---|---:|
| subset | FD001、FD003 | 2 |
| fault | bias、drift、stuck、noise | 4 |
| mag | 2、3、5（stuck 仅 3） | 3 / 1 |
| seed | 101、202、303 | 3 |
| sensors | 0,5,10 | 固定 |
| gamma | 1.30 | 固定 |
| onset | 0.5 | 固定 |

- RUL：2 × [(3 faults × 3 mags + 1 stuck × 1 mag) × 3 seeds] = **60**
- EOL：FD001/FD003 × seed 101 = **2**
- 合计 **62**，严格串行，cuda:0。

## 3. 启动与监控

```bash
cd ~/sprint24
bash scripts/queue_phase1c.sh start      # nohup 后台启动
bash scripts/queue_phase1c.sh status     # 进度
bash scripts/queue_phase1c.sh collect    # 关键行
bash scripts/queue_phase1c.sh tsv        # kind/subset/fault/mag/seed/metric/value
bash scripts/queue_phase1c.sh mean       # 3-seed mean/min/max
bash scripts/queue_phase1c.sh stop       # 当前 job 后停止
```

- 日志：`$HOME/sprint24/experiments/phase1c/<job_id>.log`
- 锁：`$HOME/sprint24/experiments/phase1c/.queue.lock`；重复启动会退出 3。
- 预检：6 个 checkpoint（FD001 `_gate`、FD003 `d80`，s101/202/303）必须存在；
  `SKIP_PREFLIGHT=1` 可绕过（不建议）。

## 4. 验收门

| 指标 | 门 |
|---|---|
| clean 代价 | 每 (subset, seed) ≤ **0.15** |
| EOL FPR | `EOL_cycle_flag_rate` ≤ **0.02**（2/2） |
| recovery | 3-seed 均值 ≥ **0.65**（按 cell 报告 mean/min/max） |
| R3 vs R5 | stuck/drift 上 R3 领先 R5 ≥ **0.10**（按 seed 配对） |
| 唯一性 | 60 unique RUL cells；2 unique EOL cells |
| 统计 | 3 seeds 是同一划分/注入的 3 个 checkpoint，不是独立数据重复 |

## 5. 运行时间

- RUL 每 job 约 40–90 s；60 个约 **40–90 min**。
- EOL 每 job 15–40 s；2 个约 **0.5–1.5 min**。
- 合计约 **45–95 min**（单卡串行）。

## 6. 注意

- mag=3 是 confirmatory；mag=2/5 是 exploratory，不得混池重算 Phase 1 结论。
- stuck 仍是 z-space；raw-space stuck 未实现，论文中必须写明。
- 3 seeds 共享 `split_seed=7` 与固定注入 RNG，只能测 checkpoint 方差。
- FD002/FD004 与官方 test 不在本队列内。
