# TEST_EVAL_PLAN — 官方 TEST 事后稳健性检查（协议冻结件）

**状态**：冻结版 v1.0，写于**任何 TEST 格启动之前**。
**提交顺序不可颠倒**：本文件与 `freeze` manifest 的 sha256 必须先落进 git，然后才允许第一次跑 TEST。
`run_sensor_aug.py` 的 `test_rules` 写着 `opened_once=True`——**TEST 只开一次**，跑完不能改协议重跑。

**这一节的全部结论只作事后报告，不回改任何判定。** 三档判定（E1a/E1b = ENGINEERING-ONLY、
方法创新 STOP、低秩路线关闭）是在验证集上按预注册判据作出的，冻结于 `11c57bc0`。
无论 TEST 结果如何，**判定不变**；若 TEST 与验证集不一致，如实记为限制并收窄「可写」句的作用域。

---

## 1. 范围

| 项 | 内容 |
|---|---|
| 臂 | `E0`（配对基线）、`E1a`、`E1b`、`E2a` |
| 划分 | `split_seed=17`（3 seed）与 `split_seed=29`（5 seed） |
| 子集 | FD001 / FD002 / FD003 / FD004 |
| 格数 | 4 臂 × 4 子集 × (3+5 seed) = **128** |
| 冒烟 | 先只跑 `E0-FD001-s101`，通过后再开整批 |

**显式不包含（记录为决定，不是遗漏）**

- **E4**：既有开关、与 Xu et al. 2026 式(6) 精确重叠，已判不构成新颖性，只作控制臂。
- **E5 四臂**：属另一棵代码树（h8 `574ac4a06ae7`），且判定发生在**效应层**（`lr4` 对 `none`
  +0.6714），TEST 不可能改变它。若要测需单独决定，会把这批翻倍。本协议不覆盖。

---

## 2. 指标口径

### 2.1 主指标：pooled RMSE（C-MAPSS 标准）

TEST 上每台测试机**只有一个预测**（末窗），标签取自 `RUL_FD00x` 并封顶 125：

```
e_u = |ŷ_u − y_u|
pooled RMSE = sqrt( mean_u e_u² )
MAE         = mean_u e_u
NASA score  = Σ_u [ exp(−e_u/13) − 1  (e_u < 13) ; exp(e_u/10) − 1 (e_u ≥ 13) ]
nasa_mean   = NASA score / n_u
bias        = mean_u (ŷ_u − y_u)
```

**口径变化必须写进报告**：验证集主指标 `val_engine_mean_roots_rmse` 是「逐机 5 个截断各算
RMSE、再逐机平均」。TEST 每机只有一个点，**这个量在 TEST 上没有对应物**。因此 TEST 的主指标
只能是 pooled RMSE——即我们在验证集上**降为辅助**的那个口径。

### 2.2 绝对数不可与验证集互比

- 验证集：截点为各机自身寿命的 40%/55%/70%/85%/95%，每机 5 个，在 20–52 台**验证机**上做发动机级平均。
- TEST：每台测试机末窗一个点，100/259/100/248 台，标签来自 `RUL_FD00x`。

E0 在 FD001 验证集是 8.3786，而上一轮 `A8_base_L70` 在 FD001 TEST 是 10.76–11.55。**这个差距是
结构性的，不是 bug。** 两个绝对数不可比，能比的只有**同一划分内的配对 Δ**。

### 2.3 汇总方式：四子集等权，不按发动机数合并

四子集发动机数为 100/259/100/248，若把所有测试机拼在一起算一个数，会被 FD002/FD004 主导。
故沿用全研究的约定：**先逐子集算，再四子集等权平均**。逐子集必须全部报出。

---

## 3. 配对统计量

### 3.1 主配对统计量：逐测试机 ΔMAE

```
d_u = |ŷ_u^arm − y_u| − |ŷ_u^E0 − y_u|      （同一台测试机配对）
```

配对键 = **同一 `(subset, seed)` 下的同一台测试机**。统计单位是发动机，不把同机多窗当独立样本
（TEST 本就一机一点，无此问题）。

### 3.2 区间：发动机级 bootstrap

- 重采样单位：**发动机**（在子集内、按引擎有放回抽 n_u 台）。
- 每次抽样：对每个子集抽引擎 → 算该子集的逐机差分均值 → 四子集等权平均。
- 次数 10000；`boot_seed=20261115`（split 17）/ `20261116`（split 29）。
- **每个统计量独立随机流**——`rng` 必须建在 `boot()` 内部。我们已因「共用一条流」栽过一次
  （排第二的统计量端点随调用顺序漂移），本协议不允许重犯。

### 3.3 次配对统计量：配对 Δ pooled RMSE

pooled RMSE 非逐机可加，因此**在每次 bootstrap 抽样内同时重算两臂的 pooled RMSE 再相减**：

```
draw_b = pooledRMSE_arm(engines_b) − pooledRMSE_E0(engines_b)
```

同一 `engines_b` 喂给两臂即为配对。报点估计与 95% CI，作为「主指标自己的区间」。

### 3.4 seed 的处理

先在每个 `(subset, seed)` 内算逐机差分，再**对 seed 取平均**（与验证集一致），最后才做引擎级
重采样。不得先把不同 seed 的发动机拼起来（那会把 n 放大 3–5 倍、把 CI 压窄）。

同时逐 seed 报出合并 Δ，供读者看 seed 间一致性。

---

## 4. 判读规则（跑之前写死）

### 4.1 什么算「与验证集一致」

对 `E1a`、`E1b` 分别判，**两个划分各判一次**：

| 结果 | 判读 |
|---|---|
| 合并 ΔMAE < 0 且 Δ pooled RMSE < 0，且至少一个划分的 CI 不含 0 | **一致**：TEST 支持验证集方向 |
| 点估计同号但两个划分 CI 都含 0 | **方向一致、功效不足**：写「未在 TEST 上检出」，不写「无效」 |
| 任一划分点估计反号 | **不一致**：记为限制，并收窄 §7 可写句的作用域 |

`E2a` 在 TEST 上只作对照报出，不参与「一致/不一致」判定——它在验证集上已判不复现。

### 4.2 不允许做的事

- 不允许因为 TEST 结果不好而改判定、改档位、改判据、改候选。
- 不允许挑 seed、挑子集、挑指标口径来制造正向结论。
- 不允许在看过 TEST 结果后重跑 TEST（`opened_once`）。
- 不允许把 TEST 的 pooled RMSE 与验证集的 mean-of-roots 相减、相除或并列成一张趋势图。
- 不允许把 `−0.36`（验证集 mean-of-roots）与 TEST 的 Δ 直接比较——**两者没有共同单位**。

### 4.3 冒烟门（先于整批）

跑 `E0-FD001-s101` 一格，三条同时满足才开整批：

1. `test_metrics.json` 的键与 `A8_base_L70` 一致：`rmse / mae / nasa_sum / nasa_mean / bias / n`；
2. `n == 100`；
3. pooled RMSE 落在 **10–13**。锚点：仓库里上一轮 **83 份** TEST 的 `test_metrics.json` 均值为
   FD001 **11.37**（n=35）、FD002 **11.13**（n=16）、FD003 **11.24**（n=16）、FD004 **12.41**（n=16）。
   **若落在 8.4 附近，说明 loader 取错（拿到验证集或截点），立即停。**

---

## 5. 产物与留痕

- 每格新增 `test_metrics.json`、`test_predictions.csv`（逐测试机：`unit / y_true / y_pred / abs_err`）。
- **`test_*` 是事后新增产物，写在 `provenance.json` 之后**，文档里要标清这一点。
- **不得改动 `provenance.json`**。
- `freeze` 文件需手写并写明：`frozen:true`、`post_hoc:true`、判定冻结于 `11c57bc0`、
  `cannot_change_verdict:true`、臂清单、以及「TEST 仅作事后报告」。
  （现成 `freeze()` 只认 `group in ('E0', cand)` 单个候选，装不下两个 ENGINEERING-ONLY 臂。）

---

## 6. 交付

1. `TEST_RESULTS.csv`：逐格一行，`run_id / arm / subset / seed / split_seed / rmse / mae /
   nasa_mean / bias / n`。
2. `TEST_PAIRED.csv`：逐格配对，`arm / subset / seed / ΔMAE / ΔpooledRMSE / 配对 E0 的 run_id`。
3. `FINAL_RESEARCH_DECISION.md` §9「事后 TEST 稳健性检查」：按 §4.1 逐臂给一致/不一致，
   写清触发了哪一条；若不一致，同时写出收窄后的可写句。

---

## 7. 版本与冻结

| 项 | 值 |
|---|---|
| 本文件 | **v1.1**（v1.0 跑前冻结 → v1.1 加附录 A 字段映射，冒烟后、整批前） |
| 判定冻结点 | `11c57bc0` |
| 允许改本文件的条件 | **判据 / 指标定义 / 统计量 / 聚合方式 / 判读规则一律不得改**。只有「产物字段名映射」可以补记，且必须重新哈希、并在整批之前提交。 |

---

## 8. 附录 A：产物字段映射（冒烟后补记，判据零改动）

冒烟格 `E0-FD001-s101` 暴露：新 TEST 路径写出的 `test_metrics.json` 键名与上一轮锚点
（`A8_base_L70`）**不同**。这是**字段名差异，不是口径差异**——本节只做映射，不改 §2 的指标
定义、不改 §3 的统计量、不改 §4 的判读规则。

| 锚点键 | 新键 | 说明 |
|---|---|---|
| `rmse` | `pooled_rmse` | TEST 每机一预测，pooled 与 engine-equal 必然重合。冒烟实测 `pooled_rmse == engine_equal_rmse == 11.105118319605069`，**逐位相同** |
| `mae` | `pooled_mae` | 同上恒等 |
| `nasa_mean` | `pooled_nasa_mean` | 同上恒等 |
| `nasa_sum` | `pooled_nasa_mean × n` | 冒烟：2.012391993367951 × 100 = 201.239，与打印的 201.2 一致 |
| `n` | `n` | 冒烟 `n = 100` ✓ |
| `bias` | **需从 `test_predictions.csv` 重算** | `bias = mean_u (ŷ_u − y_u)`；`test_metrics.json` **不含**此键 |
| `late_frac` / `severe_late` / `max_late` / `params` / `subset` / `seed` / `tag` | 新路径未产出 | 本协议不使用；`subset` / `seed` 从 `run_id` 与 manifest 取 |

**硬要求**：分析脚本必须读 `pooled_rmse` / `pooled_mae` / `pooled_nasa_mean`。
读 `rmse` 会 `KeyError`；若有人用 `.get('rmse', ...)` 兜底，会静默拿到错误值——**禁止兜底**。

**加入时点与理由**：冒烟之后、127 格整批之前。冒烟只暴露了 **`E0` 基线一格**的字段名，
未暴露任何臂的 Δ；因此这次补记不可能影响判读，而缺少它会让分析脚本直接读错键。

**产物时序（证明 `test_*` 是事后写入）**：`best.pt` 18:15:15 → `provenance.json` 20:45:53 →
`test_metrics.json` / `robustness_test.json` 23:13:16。**`provenance.json` 未被改动。**
