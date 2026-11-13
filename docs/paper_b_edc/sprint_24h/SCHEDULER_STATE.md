# SCHEDULER_STATE — 发射器（燕子）可恢复状态

写于 2026-11-13 16:26Z（远端时钟）。**这份文件的唯一目的**：如果 3090 的工作树再次被同步覆盖，
下一个人不必靠对话记录重建队列。

> 为什么写在 git 里而不是只留在 3090：今晚有 10 行台账因为「补丁只在工作树里、没进 git」
> 变成了 `tree_only`，其中 8 行永远补不回来。**调度状态是同一种东西，所以放同一个地方。**

## 1. 唯一的发射器

```
脚本   /home/user/sprint24/experiments/hc/d80/logs/watch.sh      v34
日志   /home/user/sprint24/experiments/hc/d80/logs/watch.log
停止   touch .../logs/FAILED   → watcher 见到即 HALT 并退出（不删标记不会自愈）
启动   nohup bash experiments/hc/d80/logs/watch.sh >/dev/null 2>&1 &
```

环境（缺一条数值就不可比）：

```
export PYTHONPATH=/home/user/sprint24_libs:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT   experiments/hc/tmp/wave1_init_A1_s101.pt
DATA   /home/user/rulb/data/CMAPSS
```

并发策略：每卡最多 2 格（`PERCELL=8000`，`CAP=23000`）。

## 2. 队列（19 条，v34 原文）

| # | arm | subset | seed | epochs | root | 状态 |
|---|---|---|---|---|---|---|
| 1 | B3 | FD004 | 101 | 40 | `_gate` | done |
| 2 | B3 | FD004 | 202 | 40 | `_gate` | done |
| 3 | B3 | FD004 | 303 | 40 | `_gate` | done |
| 4 | M6 | FD004 | 101 | 40 | `_gate` | done |
| 5 | B0 | FD004 | 101 | 40 | `_gate` | done |
| 6 | B0 | FD004 | 202 | 40 | `_gate` | done |
| 7 | B0 | FD004 | 303 | 40 | `_gate` | done（16:27Z） |
| 8 | M2 | FD004 | 101 | 40 | `_gate` | done（16:30Z） |
| 9 | M4 | FD004 | 101 | 40 | `_gate` | **running** |
| 10 | M3B | FD004 | 101 | 40 | `_gate` | **running** |
| 11 | M3C | FD004 | 101 | 40 | `_gate` | **running** |
| 12 | M3C | FD004 | 202 | 40 | `_gate` | **running** |
| 13 | M7 | FD004 | 101 | 40 | `_gate` | pending |
| 14 | M6 | FD004 | 202 | 40 | `_gate` | pending |
| 15 | M6 | FD004 | 303 | 40 | `_gate` | pending |
| 16 | B3 | FD004 | 101 | 80 | `_gate80` | pending |
| 17 | B3 | FD004 | 202 | 80 | `_gate80` | pending |
| 18 | B3 | FD004 | 303 | 80 | `_gate80` | pending |
| 19 | M6 | FD004 | 101 | 80 | `_gate80` | pending |

优先级：**#14/#15（`M6` s202/s303）不要降**——它是今晚唯一还有机会把 `M6 vs B3`
从 `undecidable` 翻成 `better` 的比较（`margin 0.128`，补到 3v3 后门槛会明显变窄）。

## 3. 已完成的门格（`_gate`，FD004）

```
HC-D-B0-FD004-s101-D1   13.619217174167332
HC-D-B0-FD004-s202-D1   13.032263494576259
HC-D-B3-FD004-s101-D1   17.5704
HC-D-B3-FD004-s202-D1   18.1007
HC-D-B3-FD004-s303-D1   18.0968
HC-D-M6-FD004-s101-D1   16.5350
```

`B0` 那两格的 `preds.csv` 与 `history.json` 的 sha256 与母体（`d` 根）**逐字节相同**：

```
s101  preds cffb39d903e04294  history cac7988bbbf12891
s202  preds 4afbc0246d900330  history 69650f0466462cd5
```

## 4. 两条会咬人的操作事实

**① 运行目录只在收尾时才出现。** 一个跑着的格在磁盘上**完全不可见**——`ls _gate/` 看不到它，
`find -name '*FD004*'` 也找不到它。今晚我因此误判 `B0-FD004-s202` 静默失败过一次，
实际它只是还没收尾。

> **进度只能看 `logs/HC-D-*.log` 里的 `epNNN` 行，或 `ps` 的 `--arm/--seed`；不能看目录在不在。**
> 而「完成」只能看 `metrics.json` 在不在。

**② `run_id` 不是主键。** `HC-D-B0-FD004-s101-D1` 这个字符串同时存在于 `d`、`d80`、`_gate`
三个根。**所有计数必须按 `(root, run_id)` 或 `run_dir`**，按 `run_id` 去重会静默合并不同轮次。
`_gate80` 的目录名与 40ep 门格**完全相同**，这正是它必须落独立根的原因。

**③ `watch.log` 是状态变更日志，不是心跳。** 它只在 `launch` / `OK` / `SKIP` / `WAIT` / `HALT` 时写一行。
**四格满载、没有新格落盘时，它会长时间一行不写——那是正常态，不是卡死。**

> 我今晚因此误报过一次「watcher 停了 12 分钟」：真实情况是 `00:30:51` 之后四个槽位全满，
> `00:42:52` `M4` 落盘才产生第一条新状态，`00:43:00` 立刻起了 `M7`。**日志静默 12 分钟，队列一格没丢。**

**判 liveness 要看进程，不要看日志 mtime**：

```bash
ps -o pid,stat,etime,wchan:24,args -p $(pgrep -f 'watch.sh' | head -1)
#   STAT=Ss / WCHAN=do_wait  +  子进程是 sleep 30 或 sleep 120   => 活着，在正常节拍里
#   子进程长时间是 arm_build_ok.py            => 在做构建断言（每个约 2 s，19 条队列最多约 40 s）
#   进程不存在                                  => 真的停了
```

**另一个同源的坑**：`running()` 用 `pgrep -f "train_d train --arm $1 ..."` 判活，
**`pgrep -f` 会匹配到「命令行里含有这个模式」的进程，包括包装它的 shell**。
在交互式排查时我用同样模式做检查，`M7` 显示 `RUNNING` 而当时并没有 `M7` 进程——
**那是我的检查脚本自己被匹配到了**。watcher 内部用函数调用不受影响，但**手工排查时不要照抄那个模式**。


## 5. 恢复步骤（照做即可）

```bash
ssh lab3090
cd /home/user/sprint24
date -u                                     # 先对时：远端比本地慢约 9 分钟
tail -n 20 experiments/hc/d80/logs/watch.log
ls experiments/hc/d80/logs/FAILED 2>/dev/null && cat it   # 有标记就是停了
ps -eo pid,etimes,args | grep 'train_d train' | grep -v grep
for d in experiments/hc/_gate/*FD004*; do [ -f "$d/metrics.json" ] && basename $d; done
```

若 watcher 已死：去掉 `FAILED` 标记后按第 1 节重启，它按 `metrics.json` 存在与否自动 `SKIP` 已完成的格，
**幂等，重复启动不会重跑已完成的格**（这是它能安全重启的唯一原因）。

## 6. GPU 使用账（三个数，回答三个不同的问题）

**不要把它们并成一个「GPU 利用率」。** 它们的分母不同、数据来源不同，混用会得到互相矛盾的结论。

### 6.1 占用时间轴 —— `done-when #6` 的判据：**不达标**

数据源：`SEED_RESULTS.csv` 的 `created_utc` 与 `elapsed_s` 两列，区间取 `[created_utc − elapsed_s, created_utc]`，
按时间轴求并集。**不依赖日志**，所以日志轮转不影响这个数（见 6.3）。

```
窗口          07:00Z → 16:29:39Z   = 9.494 h
有训练覆盖    7.008 h              = 73.8%
```

**四条 > 15 分钟的空转**：

```
07:49Z → 08:47Z   58.4 min
07:00Z → 07:41Z   42.0 min   ← 首格 07:41:57Z 才起，冲刺起点后 42 分钟
11:12Z → 11:32Z   19.8 min
15:18Z → 15:35Z   17.1 min
```

**结论：`done-when #6`（非训练空转不超过 15 分钟）不达标，四条超线，最长 58.4 分钟。**
这是过程指标、事后不可补救，只能如实记录。**最贵的一条是开头那 42 分钟**——冲刺起点到首格之间，
两卡全空，而那时代码已经就绪。

### 6.2 槽位占用率 —— 两张卡压满了没有

```
cell_hours（metrics 口径，100 格）   22.734 h
槽位 = 2 卡 × 每卡 2 格 = 4
可用槽位·小时 = 9.51 × 4           38.04
占用率                              59.8%
```

**这是「槽位压满没有」，不是「卡在忙没有」。** 59.8% 意味着平均有 2.39 个槽位在跑，
**每卡约 1.2 个格并发**——卡一直在忙，但没有稳定压满 2 路。

### 6.3 `cell_hours` 的两个来源，差 3.04 h，差在哪

```
日志口径（grep '^DONE ' 累加）   81 格   19.69 h   ← 下界
metrics 口径（SEED_RESULTS）    100 格   22.73 h   ← 准
差                                     19 格    3.04 h
```

**差的 19 格是日志被轮转掉的格，不是没跑的格。** 所以：

> **凡是用日志算的量都是下界；凡是用 `metrics.json` / `SEED_RESULTS` 算的量不是。**
> 这就是为什么 6.1 的占用时间轴可信、而 6.3 的日志口径只能当下界报。

**报数时的正确写法**：`cell_hours = 22.73`（metrics 口径）；如果只拿到日志，写 `≥ 19.69` 并注明是下界。

### 6.4 三个数的分母各是什么

| 数 | 分子 | 分母 | 回答 |
|---|---|---|---|
| 73.8% | 有任一格在跑的挂钟 | 9.494 h | **有没有空转**（#6 用这个） |
| 59.8% | `cell_hours` | 9.51 h × 4 槽位 | **槽位压满没有** |
| 19.69 | 日志里还有 `DONE` 行的格 | — | **下界，不是利用率** |

