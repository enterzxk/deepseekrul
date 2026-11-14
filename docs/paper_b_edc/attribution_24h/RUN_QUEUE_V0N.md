# V0N 队列规格（强骨干 cycle-off，12 格）

**状态：`blocked` / `paired-protocol-unresolved`，不发射。** 队列里另有 12 行 `V0C_R2`（`status=proposed`），是方案 A 的配对参照，同样未发射。
原因：2026-11-14 军师复审指出两处实质问题（配对缺口、attempt 目录覆盖），两处均成立，已按代码行核实，见 §2 与 §4。

---

## 1. 为什么是这 12 格（证据链不变）

| 证据 | 实测 |
|---|---|
| 网格定义 | `paper_b/experiments/grids/nocycle.json` 12 条，tag 全 `NC_base_L70`，`ms/sa/hi = off/off/none`（**base 骨干**，不是 V0N） |
| 盘上目录 | lab3090 `paper_b/results/raw/` 无 `V0*` 目录 |
| 台账标签 | `runs.csv` 128 行无 `V0N`（**此条单独不足以定论**，台账不是全量） |
| 全库扫描 | 经理扫 6 个网格共 216 格，`full/basic/hf + use_cycle=False` 0 格 |

结论：**已核查的档案中（网格定义、盘上目录、台账标签、216 格扫描）没有找到任何可复用运行。**
按军师的收紧，这句话**不等于"历史上从未运行"**，只等于"现有记录里没有"——所以它是**证据缺口**，不是否定性事实。
**不补的代价**：cycle 在强骨干上的贡献只能拿 base 骨干的 `+1.0405` 外推——外推不是证据。

---

## 2. 配对缺口：直接发车会同时动三个变量（军师 2026-11-14 复审，成立）

| 变量 | 历史母体 `F_full_basic_hi1`（= `A8_full_L70` = V0C） | 当前 `paper_b/training/train.py` |
|---|---|---|
| **cycle** | on | **off（本队列要改的那一个）** |
| **选轮指标** | **pooled**（`A8_ABLATION_96CELL.md` §6 明写；`metrics.best_val_rmse = 11.4584` 就是 pooled 值） | **engine_mean**（`train.py:104-115` `val_metrics()` 返回 `(pooled, engine_mean)`；`L150 pooled, vr, _ = ...`；`L156 if vr < best - 1e-4` 选的是 `vr`=engine_mean） |
| **数值精度** | **未记录**（`SEED_ROBUSTNESS` 无此列；恢复的公开基线是 FP32 且禁 autocast/TF32） | **bf16 硬编码**（`train.py:124` `with torch.autocast(device_type='cuda', dtype=torch.bfloat16)`） |

**所以"36 键配置相同"只排除了结构、预算与划分，排不除选轮与精度。** 这与 IDENTITY 文档里那条作用域是同一句话：
同配置 ≠ 同测量。直接发 12 格，`V0N − V0C` 的差里就混着"选轮换了"和"精度换了"。

**一个必须一起说的副作用**：历史母体的 `val_eng = 10.9814` 是**事后从 `val_pred.csv` 重算**的，
它的 `best_ep` 却由 pooled 选出——**选轮与评估本来就不是同一个量**（A8 报告 §6 已记录，不修正）。

---

## 3. 三个可选方案（待军师裁定；我推荐 A）

| 方案 | 做法 | 新增格 | 能否单独分离 cycle | 代价 |
|---|---|---:|---|---|
| **A（推荐）配对重训** | `V0N_L70` 12 + `V0C_R2` 12，**同一 train.py、同选轮、同精度** | **24** | **能**（只差 `use_cycle`） | 两卡约 5–8 小时 |
| B 复刻历史测量 | 给 `train.py` 加 `--select pooled`、`--precision fp32` | 12 | 近似能 | 改代码=改协议；且母体精度"未记录"，`fp32` 是**猜的** |
| C 事后重选 | 用 `history.json` 把母体按 engine-mean 重选 | 0 | 不能 | **不可行**：`history.json` 只有 `val_rmse`/`val_pooled_rmse` 两列，没有逐 engine 的逐 epoch 预测，engine-mean 无法回溯 |

**推荐 A 的第二个理由**：`V0C_R2` 顺带给出「同代码同精度下 V0C 的可复现性」。
`V0C_R2 − F_full` 的差 = 选轮 + 精度 + 代码漂移的**合计效应**——这本身是本轮要报的一条归因（"测量口径值多少 RMSE"），
而且它必须在同一批 seed 上算，不能拿别的臂凑。

---

## 4. 目录隔离（修 `-R2` 覆盖矛盾，军师指出的第二处）

原规则「`-R2` 只改 run_id、目录不变」与「永不覆盖」自相矛盾：`run_grid` 写 `<tag>/<subset>/s<seed>`，
run_id 不参与路径，所以第二次 attempt 会写进第一次的目录。

**修正：attempt 由 `attempt_tag` 决定，run_id 与目录同步加后缀。**

```
第 1 次   attempt_tag = V0N_L70        run_id = ATTR-V0N-FD001-s101        目录 .../V0N_L70/FD001/s101/
第 2 次   attempt_tag = V0N_L70_R2     run_id = ATTR-V0N-FD001-s101-R2     目录 .../V0N_L70_R2/FD001/s101/
```

`run_grid` 不需要改：tag 不同 → 目录不同 → 物理上不可能覆盖。队列里 `attempt_tag` 是必填列。

---

## 5. 查重字段在哪（答燕子）

`ms_mode / sa_mode / hi_source / use_cycle` **只写在每个 run 目录的 `config.json` 里（36 键），`metrics.json` 没有这四个键**，也不能用 `params` 反推（`params` 同时受 `max_span` 影响：`84034` vs `82114` 的差就是 `30×64`）。

发射前查重必须遍历**所有** raw 目录的 `config.json`，而不是只查 `V0N*` 或只查台账：

```bash
# 在 3090 上：找出 full/basic/hf 且 cycle 关闭的全部 run（不论目录叫什么）
python - <<'PY'
import json, glob, os
root = os.path.expanduser('~/rulb/paper_b/results/raw')
hits = []
for p in glob.glob(root + '/*/*/s*/config.json'):
    c = json.load(open(p))
    if (c.get('ms_mode') == 'full' and c.get('sa_mode') == 'basic'
            and c.get('hi_source') == 'hf' and c.get('use_cycle') is False):
        hits.append(p)
print('hits', len(hits)); print('\n'.join(hits[:20]))
PY
```

预期 `hits 0`；**若非 0，则那几格就是可复用运行，本队列对应格改 `status=superseded` 并直接引用，不重训。**

---

## 6. 每格冻结字段（12 行完全一致，只有 subset/seed 不同）

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
| selection_metric / precision | engine_mean / bf16 | **当前 train.py 的实际行为**（与母体不同，见 §2） |

**唯一变量证明（限定版）**：母体 `F_full_basic_hi1/FD001/s101/config.json` 共 **36 键**，与 V0N 期望 config 相比
**只有 `use_cycle` 与 `tag` 两个键不同，其余 34 键逐字段相同**（2026-11-14 在 lab3090 实测）。
**但这只证明"配置单变量"，不证明"测量单变量"**——选轮与精度不在 config 里。

---

## 7. grid 文件生成规则（不要手写数值）

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
（军师已核：本地 `v0n.json` 目前**不存在**，所以"可执行 grid"这一条也还没闭合。）

---

## 8. 发射命令（待 §3 裁定后由燕子唯一发射）

```bash
# 方案 A：两个 attempt_tag 各跑一遍，同卡串行
python -m paper_b.experiments.run_grid --grid paper_b/experiments/grids/v0n.json \
  --device cuda:1 --data-dir ~/rulb/data/CMAPSS
python -m paper_b.experiments.run_grid --grid paper_b/experiments/grids/v0c_r2.json \
  --device cuda:1 --data-dir ~/rulb/data/CMAPSS
```

单格参考耗时：FD001/FD003 约 2–5 分钟（20 引擎），FD002/FD004 约 25–50 分钟（52/50 引擎）。
12 格串行约 2.5–4 小时；方案 A 两臂共 24 格，两卡对半约 2–3 小时。

---

## 9. 落盘验收（六条）

1. **12 个目录**：`V0N_L70/FD001|FD002|FD003|FD004/s101|s202|s303` 各含 `metrics.json` + `val_pred.csv`。
2. **config 逐键**：每格 config 与母体同格 diff **恰好 2 个键**（`use_cycle`、`tag`）。
3. **参数**：`metrics.json.params == 115075`（等于 115651 说明 `use_cycle` 没生效——**这条能抓住"配置写了但没传进模型"**）。
4. **预测规模**：`val_pred.csv` 行数 = `引擎数 × 5`（FD001/3 20×5=100；FD002 52×5=260；FD004 50×5=250）。
5. **口径**：`metrics.json` 的 `val_engine_rmse` **不可信**（历史存 pooled；当前代码又把它写成 `vr`），engine-level 一律从 `val_pred.csv` 重算。
6. **attempt 一致**：`attempt_tag` 与 `run_id` 后缀必须同进（`V0N_L70_R2` ↔ `...-R2`），且目录与 attempt_tag 同名。

---

## 10. 判定用法（只报 validation；方案 A 下才成立）

对每个子集，配对差 `d_i = V0N_i − V0C_R2_i`（i = 3 个 seed，同 split、同引擎集、同选轮、同精度）：

```
mean_d = mean(d)   sd_d = sd(d)   half = 4.302653 * sd_d / sqrt(3)     # t(0.975, 2)
|mean_d| > half → cycle 在该子集上可区分（方向按符号）；否则 undecidable（不是"无贡献"）
```

必须并列报 `mean_d / sd_d / half` 与每格 `epochs_run / best_ep`。

**若走方案 B**，判定式同上但必须在文档里写明「母体精度是推定值」，并且这一条要进失败清单：
**用一个"猜的精度"去对齐，得到的差不能当机制证据。**

---

## 11. 失败处理

| 现象 | 处理 |
|---|---|
| 某格 `params != 115075` | 停，查 `use_cycle` 是否传入 `DSPT(...)`；该格作废，用 `_R2` attempt 重跑 |
| `val_pred.csv` 行数不符 | 同上作废重跑；不得手改预测 |
| 某格 OOM | 不改 batch（改协议），同卡串行/清显存后重跑该格 |
| 单格 `epochs_run` 触顶 40 未收敛 | 照常入档，标 `budget_limited`，判定时并列报 `epochs_run` |
| 查重命中已有 run | 该格改 `superseded`，直接引用已有结果，**不重训** |
