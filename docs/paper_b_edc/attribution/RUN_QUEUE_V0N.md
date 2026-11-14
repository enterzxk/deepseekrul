# V0N 队列规格（强骨干 cycle-off，12 格）

状态：已写入 `RUN_QUEUE.csv`（12 行，`status=queued`，`blocker=none`）。等待发射器（燕子）按本规格发射。
用途：任务 B-1「Cycle 的独立贡献」在**强骨干**上的那一半。base 骨干那一半已由 `NC_base_L70` 覆盖（+1.0405）。

---

## 1. 为什么恰好是这 12 格（三条证据链）

| 证据 | 实测 |
|---|---|
| 网格定义 | `paper_b/experiments/grids/nocycle.json` 12 条，tag 全 `NC_base_L70`，`ms/sa/hi = off/off/none`（**base 骨干**，不是 V0N） |
| 盘上目录 | lab3090 `paper_b/results/raw/` 无 `V0*` 目录 |
| 台账标签 | `runs.csv` 128 行无 `V0N`（**此条单独不足以定论**，台账不是全量） |
| 全库扫描 | 经理扫 `ablation8/attribution/factorial/hi_mech/ms_mech/sa_mech` 共 216 格，`full/basic/hf + use_cycle=False` **0 格** |

结论：**已核查的档案中（网格定义、盘上目录、台账标签、216 格扫描）没有找到任何可复用运行。**
按军师的收紧，这句话**不等于"历史上从未运行"**，只等于"现有记录里没有"——所以它是**证据缺口**，不是否定性事实。
缺口按 12 格补；发射前由燕子按配置字段再查一次重（`ms/sa/hi/use_cycle` 四元组 + 目录名）。
**不补的代价**：cycle 在强骨干上的贡献只能拿 base 骨干的 `+1.0405` 外推——外推不是证据。

---

## 2. run_id 规则（冻结，发射器照此生成）

```
主格：  ATTR-V0N-<subset>-s<seed>           例 ATTR-V0N-FD004-s202
重跑：  ATTR-V0N-<subset>-s<seed>-R2        再次 -R3 …（永不复用、永不覆盖）
```

规则要点：

1. **确定性**：主格 run_id 不含时间戳，12 个 id 在训练前就能全部写死——这样"队列 12 行 ↔ 盘上 12 目录"可以逐一对账，不会因为时间戳不同而漂移。
2. **重跑必须新 id**：同一格重跑时 `-R2`，**原行不改**；`RUN_QUEUE.csv` 追加一行并把原行 `status` 标 `superseded`（原行的 `val_eng` 保留，不删）。
3. **目录**：`paper_b/results/raw/V0N_L70/<subset>/s<seed>/`（沿用 `run_grid` 的 `<tag>/<subset>/s<seed>` 布局，这样既有的 engine-level 重算与 seed 汇总脚本可直接读）。
4. **id 与 tag 分工**：`run_id` 标识"这一格这一次"，`tag` 标识"这一族的配置"。`V0N_L70` 是 tag（配置族），`ATTR-V0N-FD001-s101` 是 run（个体）。两者不可互相替代。
5. **与 `NC_base_L70` 的区分（经理点名要防的混淆）**：两者都 `use_cycle=False`，但**骨干不同**——
   `NC_base_L70` 是 `ms=off/sa=off/hi=none`（base 骨干，12 格、已跑、无 test）；
   `V0N_L70` 是 `ms=full/sa=basic/hi=hf`（强骨干，本队列 12 格）。
   **tag、run_id、params 三重可分**：`V0N_L70` 期望 `115075`，`NC_base_L70` 实测 `76482`。
   两者**禁止合并统计**；若要同时报，必须写成"base 骨干 +1.0405 / 强骨干 X"两行，不能加总或平均。
6. **唯一性断言**（发射前跑一次）：
   `python -c "import csv,collections;r=list(csv.DictReader(open('docs/paper_b_edc/attribution/RUN_QUEUE.csv')));c=collections.Counter(x['run_id'] for x in r);assert len(r)==12 and all(v==1 for v in c.values());print('ok 12 unique')"`

---

## 3. 每格冻结字段（12 行完全一致，只有 subset/seed 不同）

| 字段 | 值 | 来源 |
|---|---|---|
| model | dspt | 母体同格 config |
| ms_mode / sa_mode / hi_source / hi_mono | full / basic / hf / correct | 母体同格 config |
| **use_cycle** | **False** | 本队列唯一改动的语义字段 |
| input_mode | raw | 母体 |
| epochs / patience / span | 40 / 12 / 70 | `protocol.json` + 冻结 |
| batch / lr / weight_decay | 512 / 0.001 / 0.0001 | 同上 |
| d_model / layers / dropout | 64 / 2 / 0.2 | 同上 |
| n_cond / split_seed / fold / val_frac | 6 / 7 / 0 / 0.2 | 同上 |
| rul_cap | 125 | 同上 |
| mode / aux_dual / use_pad_mask / instnorm | dual / true / true / false | 母体 |
| **expected_params** | **115075** | `115651 − 576`，576 = `pe_proj`（`2×4×64 + 64`） |

**唯一变量证明**：母体 `F_full_basic_hi1/FD001/s101/config.json` 共 **36 键**，与 V0N 期望 config 相比
**只有 `use_cycle` 与 `tag` 两个键不同，其余 34 键逐字段相同**（2026-11-14 在 lab3090 实测）。

---

## 4. grid 文件生成规则（不要手写数值）

`paper_b/experiments/grids/v0n.json`，12 条，生成方式**必须**是"读母体 config、改两个键"：

```python
import json, pathlib
src = pathlib.Path('paper_b/results/raw/F_full_basic_hi1/FD001/s101/config.json')
base = json.loads(src.read_text())
runs = []
for sub in ('FD001', 'FD002', 'FD003', 'FD004'):
    for seed in (101, 202, 303):
        r = dict(base)
        r.update(subset=sub, seed=seed, tag='V0N_L70', use_cycle=False)
        r.pop('data_dir', None)          # run_grid 用 --data-dir 注入
        runs.append(r)
pathlib.Path('paper_b/experiments/grids/v0n.json').write_text(json.dumps(runs, indent=1))
```

手写这 12 条会有 12 次抄错的机会；从母体拷贝只有 2 个键会变，**抄错的可能性被结构性排除**。

---

## 5. 发射命令（`run_grid` 幂等，已有 `metrics.json` 的格自动 skip）

```bash
# 单卡（GPU1，消融口径）
python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/v0n.json \
  --device cuda:1 --data-dir ~/rulb/data/CMAPSS

# 两卡对半（仅当 GPU0 也空；同卡串行、控显存）
python -m paper_b.experiments.run_grid --grid paper_b/experiments/grids/v0n.json \
  --shard 0 --nshards 2 --device cuda:0 --data-dir ~/rulb/data/CMAPSS
python -m paper_b.experiments.run_grid --grid paper_b/experiments/grids/v0n.json \
  --shard 1 --nshards 2 --device cuda:1 --data-dir ~/rulb/data/CMAPSS
```

单格参考耗时：FD001/FD003 约 2–5 分钟（20 引擎），FD002/FD004 约 25–50 分钟（52/50 引擎）。
**12 格串行约 2.5–4 小时；两卡对半约 1.5–2 小时。**

---

## 6. 落盘验收（五条，缺一不可）

1. **12 个目录**：`V0N_L70/FD001|FD002|FD003|FD004/s101|s202|s303` 各含 `metrics.json` + `val_pred.csv`。
2. **config 逐键**：每格 config 与母体同格 diff **恰好 2 个键**（`use_cycle`、`tag`）。
3. **参数**：`metrics.json.params == 115075`（等于 115651 说明 `use_cycle` 没生效——**这条能抓住"配置写了但没传进模型"**）。
4. **预测规模**：`val_pred.csv` 行数 = `引擎数 × 5`（FD001/3 20×5=100；FD002 52×5=260；FD004 50×5=250）。
5. **口径**：`metrics.json` 里的 `val_engine_rmse` **不可信**（legacy 存的是 pooled），engine-level 必须从 `val_pred.csv` 重算：
   `rmse_per_engine = sqrt(mean((y-p)^2))` → `engine_mean = mean(rmse_per_engine)`。

---

## 7. 判定用法（只报 validation）

对每个子集，配对差 `d_i = V0N_i − V0C_i`（i = 3 个 seed，同 split、同引擎集）：

```
mean_d  = mean(d)
sd_d    = sd(d)                      # ddof=1，df=2
half    = 4.302653 * sd_d / sqrt(3)   # t(0.975, 2)
判定    : |mean_d| > half → cycle 在该子集上"可区分"（方向按符号）
          否则            → undecidable（不是"无贡献"）
```

**必须一起报的三件事**：`mean_d`、`sd_d`、`half`；以及每格的 `epochs_run` / `best_ep`
（长度不齐会让"谁更差"混入预算效应）。

**不做的事**：不跑 test（本轮决策只看 validation）；不重跑母体 12 格；不改协议；
不因为结果方向不合预期而改门槛或换 seed。

---

## 8. 失败处理

| 现象 | 处理 |
|---|---|
| 某格 `params != 115075` | 立即停，查 `use_cycle` 是否传入 `DSPT(...)`；该格作废、`-R2` 重跑 |
| `val_pred.csv` 行数不符 | 同上作废重跑；不得手改预测 |
| 某格 OOM | 降 batch 会破坏协议——**不改协议**，改为同卡串行/清显存后重跑该格 |
| 单格 `epochs_run` 触顶 40 未收敛 | 照常入档，标 `budget_limited`；判定时并列报 `epochs_run` |
