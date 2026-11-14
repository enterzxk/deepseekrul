# V0N 队列规格（强骨干 cycle-off，12 格）

**状态：`queued` / `blocker=none`（grid 已落两端并锁 sha），等燕子按已有授权发射。**
协议按军师 L2 终审统一（2026-11-14）：走 3090 已核旧发射树，**pooled 选轮 + bf16 autocast**，engine-mean 只用于报告。
`V0C_R2`（原方案 A 的配对参照）**已撤回**，理由见 §3。

---

## 1. 为什么是这 12 格（证据链）

| 证据 | 实测 |
|---|---|
| 网格定义 | `nocycle.json` 12 条，tag 全 `NC_base_L70`，`off/off/none`（**base 骨干**，不是 V0N） |
| 盘上目录 | 3090 `paper_b/results/raw/` 无 `V0*` 目录 |
| 配置扫描 | 83 个 `config.json`：`use_cycle=False` 仅 3 格（`VAN_raw_L70`，骨干不同）；四元组 `full/basic/hf/False` **0 格** |
| 归档复核（军师独立代理） | 本地 301 元数据对象 / 20 个 grid；3090 `~/rulb/paper_b/results` 530、`experiments` 395、`~/deepseek-rul` 2844、`~/rulb/results` 28；a40 只读 `~/deepseek-rul` 70；**强骨干 cycle-off 与 variant `V0N` 均 0 命中** |

**结论限这些已核归档**：写「已核查档案中未找到可复用运行」，**不写「全历史从未运行」**。
不补的代价：cycle 在强骨干上的贡献只能拿 base 骨干的 `+1.0405` 外推——外推不是证据。

---

## 2. L2 终审：协议统一（这是本轮唯一有效的口径）

| 项 | 取值 | 依据 |
|---|---|---|
| 发射树 | `~/rulb`（**不是**本地新版 `paper-b-contribution-attribution`） | 3090 三文件 sha256 与 `09076b81` 逐字节一致 |
| 选轮指标 | **`val_pooled_rmse`** | 旧树 `history.json` 每 epoch 只有 `val_rmse`；`F_full/FD001/s101` 的 `best_ep=18` 落在 pooled `11.4584`，不是 engine `10.7175` |
| 报告指标 | **`val_engine_mean_rmse`**（从 `val_pred.csv` 重算） | 与母体同法重算，两侧可比 |
| 精度 | **bf16-autocast** | `train_d.py:353/359` autocast；损伤递推跑 fp32 |
| 母体 | `F_full_basic_hi1`（`hi_mono=correct`） | 冻结 `train_one` 产物 |
| 历史精度 | **未记录** | 不能拿当前 hash 冒充历史执行指纹 |

**三个源锁 sha256（我在 3090 上逐项复核，与军师给出的逐位一致）**

```
paper_b/training/train.py        14eb3eb972359bb1b8c48193b7619fccdc7ea3e29aedb5e5f6799d70efbeab42
paper_b/training/metrics.py      c1c4bb0ceea2c3191bcf9467cbe959d8858611ef4a011241b2fb53ac6423d1c7
paper_b/experiments/run_grid.py  e999be4e96be894595772ddc9c691c2d70957df305bd4e9ea085e68fa3074bdd
```

**这条为什么让配对成立**：母体与 V0N **选轮规则相同（pooled）**、**精度相同（bf16-autocast）**、**代码相同（同一 sha）**，
所以两侧的 `best_ep` 由同一把尺子选出，engine-mean 只是把两侧用同一把新尺子重算一遍。**变量只剩 cycle。**

**一个必须写清的用词**：V0N 不是「母体加一个布尔位」——`use_cycle=False` 会让 `pe_proj` 不再实例化，
参数从 `115651` 降到 `115075`（差 `576 = 2×4×64 + 64`）。准确说法是「**cycle 及其连带参数**」，不是「仅 `use_cycle`」。

**禁止事项**：不要把本地新版 `train.py` / `metrics.py` 同步到 `~/rulb` 覆盖发射树——那会让本批与母体分家，
且历史精度无法验证。

---

## 3. 撤回 `V0C_R2`（原方案 A 的配对参照）

| 项 | 内容 |
|---|---|
| 原提案 | 24 格：`V0N` 12 + `V0C_R2` 12，用当前 trainer 重训 cycle-on |
| 撤回理由 | 前提是「本地新版 trainer 与母体不同口径」。L2 现场核源码后确认：**3090 发射树与 `09076b81` 字节一致，本来就是 pooled 选轮 + bf16**，与母体同口径。前提不成立 → 不需要重训 cycle-on |
| 状态 | 队列中的 12 行 `V0C_R2` **已删除**（不是改成 withdrawn 留着），本表即撤回记录 |
| 保留的教训 | 「换一个 trainer 重训参照」只有在**确认两侧口径不同**时才成立；确认同口径后，重训是浪费 12 格且引入新变量（不同训练实例） |

---

## 4. attempt 目录隔离

`run_grid` 写 `<tag>/<subset>/s<seed>`，`run_id` 不参与路径。所以 attempt 必须由 **`attempt_tag`** 决定：

```
第 1 次  attempt_tag=V0N_L70      run_id=ATTR-V0N-FD001-s101      → .../V0N_L70/FD001/s101/
第 2 次  attempt_tag=V0N_L70_R2   run_id=ATTR-V0N-FD001-s101-R2   → .../V0N_L70_R2/FD001/s101/
```

tag 不同 → 目录不同 → 物理上不可能覆盖。**不改 `run_grid`。**

---

## 5. grid（已生成，两端同 sha）

```
本地   paper_b/experiments/grids/v0n.json
3090   ~/rulb/paper_b/experiments/grids/v0n.json
sha256 1759feb864e07428ae7a70f8d135b0840c22e33096c59b69c5f0e285e4a11a66   （两端一致）
entries 12    keys 35    tag V0N_L70    use_cycle False(12/12)    hi_mono correct(12/12)
```

生成方式**不是手写**：读母体 `F_full_basic_hi1/FD001/s101/config.json`（36 键）→ 改 `subset/seed/tag/use_cycle`
→ 删 `data_dir`（`run_grid` 用 `--data-dir` 注入）→ 写 12 条。这样 12 次抄错的机会被结构性排除。
（若 grid 丢失需重生成：`python /tmp/mkgrid.py`，脚本逻辑即上述四步，重生成后 sha 应逐位相同。）

---

## 6. 发射命令（燕子唯一发射）

```bash
# 单卡
cd ~/rulb && ~/miniconda3/envs/ncmapss/bin/python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/v0n.json --device cuda:1 --data-dir ~/rulb/data/CMAPSS

# 两卡对半（两卡均空闲时）
cd ~/rulb && ~/miniconda3/envs/ncmapss/bin/python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/v0n.json --shard 0 --nshards 2 --device cuda:0 --data-dir ~/rulb/data/CMAPSS
cd ~/rulb && ~/miniconda3/envs/ncmapss/bin/python -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/v0n.json --shard 1 --nshards 2 --device cuda:1 --data-dir ~/rulb/data/CMAPSS
```

发射前记录环境（写进 run note / 台账）：`torch.__version__`、`torch.version.cuda`、CUDA driver、以及 §2 三个源码 sha。
**记录的是"本次执行"的环境，不是"历史执行"的指纹**——历史精度仍是「未记录」。

---

## 7. 落盘验收（七条）

1. **12 个目录**：`V0N_L70/FD001|FD002|FD003|FD004/s101|s202|s303`，各含 `metrics.json` + `val_pred.csv`。
2. **config 逐键**：每格 config 与母体同格 diff **恰好 2 个键**（`use_cycle`、`tag`）。
3. **参数**：`params == 115075`（等于 115651 说明 `use_cycle` 没生效）。
4. **预测规模**：`val_pred.csv` 行数 = 引擎数 × 5（100 / 260 / 100 / 250）。
5. **口径**：`metrics.json` 的 `val_engine_rmse` 不可信（旧树存 pooled）；engine-mean 一律从 `val_pred.csv` 重算。
6. **attempt 一致**：`attempt_tag` 与 `run_id` 后缀同进，目录与 attempt_tag 同名。
7. **源锁**：落盘后重算 `train.py` / `metrics.py` / `run_grid.py` 的 sha256，必须仍是 §2 三个值；变了则本批作废。

---

## 8. 判定用法（只报 validation）

对每个子集，配对差 `d_i = V0N_i − V0C_i`（i = 3 seed，同 split、同引擎集、**同选轮、同精度、同代码**）：

```
mean_d = mean(d)   sd_d = sd(d)   half = 4.302653 · sd_d / √3     # t(0.975, 2)
|mean_d| > half → cycle 在该子集上可区分（方向按符号）；否则 undecidable（不是"无贡献"）
```

两侧都用 **engine-mean（从 `val_pred.csv` 重算）** 比较；`epochs_run` / `best_ep` 并列报。
**`best_ep` 是按 pooled 选的，两侧同规则**——这是本批能配对的前提，不是缺陷。

---

## 9. 失败处理

| 现象 | 处理 |
|---|---|
| 某格 `params != 115075` | 停，查 `use_cycle` 是否传入 `DSPT(...)`；该格作废，`_R2` attempt 重跑 |
| `val_pred.csv` 行数不符 | 同上作废重跑；不得手改预测 |
| 源码 sha 变化 | **整批作废**（与母体分家），恢复 sha 后重跑 |
| 某格 OOM | 不改 batch（改协议），同卡串行/清显存后重跑该格 |
| `epochs_run` 触顶 40 未收敛 | 照常入档，标 `budget_limited`，判定时并列报 `epochs_run` |
| 查重命中已有 run | 该格改 `superseded`，直接引用，不重训 |
