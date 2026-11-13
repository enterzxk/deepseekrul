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
| 7 | B0 | FD004 | 303 | 40 | `_gate` | **running** |
| 8 | M2 | FD004 | 101 | 40 | `_gate` | **running** |
| 9 | M4 | FD004 | 101 | 40 | `_gate` | **running** |
| 10 | M3B | FD004 | 101 | 40 | `_gate` | **running** |
| 11 | M3C | FD004 | 101 | 40 | `_gate` | pending |
| 12 | M3C | FD004 | 202 | 40 | `_gate` | pending |
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
