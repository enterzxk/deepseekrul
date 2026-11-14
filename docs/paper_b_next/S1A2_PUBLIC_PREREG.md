# S1-A 第二半预注册：把默认位置套到五个公开方法

状态：**冻结中**（v1）。冻结凭据以本文件首次提交的 commit 为准，提交后把 SHA 贴到群里才开始任何训练。
上游依据：`docs/paper_b_next/S1A_CYCLE_MATCHED.md`（commit `b8e14e67`）与
`docs/paper_b_next/S1A_POSITION_PREREG.md`（commit `29f337f9`）。
本文件在**任何 V1 训练开始之前**写成，阈值与判据不参考任何 V1 结果。

---

## 0. 一句话

第一半判「无唯一胜出」，`pre_norm` 只是**默认位置**。第二半在这个默认位置上，把同一套
Fourier cycle 编码以最小加法注入接到五个公开方法的回归头输入上，问的是：**在方法已有输入之上，
这段重编码能不能稳定带来增量**。五个方法不合并成一个「Cycle 效应」。

---

## 1. 决策对象与格数

| 臂 | 方法 | 子集 | 种子 | 格数 |
|---|---|---|---|---:|
| V0（复用） | 5 | 4 | 3 | 0（不重跑） |
| V1 | 5 | 4 | 3 | **60** |
| sham | `two_phase` / `aidgn` / `dual_mixer` | 4 | 3 | **36** |
| 合计新训 | | | | **96** |

种子 `101/202/303`。V0 直接复用 `d7bd9c17` 的 FP32 结果（60 个 unified run，每格都有
`validation_predictions.csv`），**不重跑**。

ETA：按 V0 同预算实测训练时间，V1 60 格 25,972 s + sham 36 格 19,085 s = **45,057 s ≈ 12.5 卡时**；
三卡并行墙钟约 **4.5–5.5 小时**（按最长单格补齐）。`dual_mixer` 与 V0 一样全量训练，
不存在「只训头所以更快」的折扣。

---

## 2. 注入形式

对每个方法，找到**产出标量的那一层**（回归头），在它的输入上加一项：

```
h  ←  h + pe_proj(pe(cycle))
```

用 `register_forward_pre_hook` 挂在该层上。**不插入任何新层**，不替换任何模块，不改该方法的
`loss`，不改它的前向顺序；方法自己的头仍然产出预测。代价是 `pe_proj: 8 → d_head`，即
`9 · d_head` 个参数。

`pe(cycle)` 与第一半完全一致：`cycle = min(c/250, 1)`，`c_abs = cycle × 250`，周期
`(25, 50, 125, 250)`，输出 8 维 `sin`/`cos`。实现见 `paper_b/baselines/open_source/cycle.py`。

### 2.1 逐方法注入点与语义

头输入**已经过归一化**的，加法落在归一化之后，语义上是 `post_norm` 而不是 `pre_norm`；
一律标 `pre_norm` 是错的。

| 方法 | 回归头 | 头输入宽度 | 头之前是否有归一化 | 本次注入的语义 |
|---|---|---:|---|---|
| `aidgn` | `net.channel_decoder.2` | 56 | 无（`Linear(14,56)→ReLU→Linear(56,1)`） | `pre_norm` |
| `two_phase` | `net.out.mlp.3` | 32 | **有**（`Linear(16,32)→ReLU→LayerNorm(32)→Linear(32,1)`） | **`post_norm`** |
| `dual_mixer` | `net.output.0` | 960 | 无（`Linear(960,1)`） | `pre_norm` |
| `ffve` | `decoder.regressor.2` | 200 | 无（`Linear(3,200)→Tanh→Linear(200,1)`） | `pre_norm` |
| `fcstgnn` | `net.fc.fc4` | 8 | 无（`fc1→ReLU→fc2→ReLU→fc3→ReLU→Linear(8,1)`） | `pre_norm` |

`aidgn` 的头是 `channel_decoder` 的最后一层，不是 `decoder.2`（256 维那一层）：
源码路径是 `fc_output → decoder → channel_decoder → torch.abs(cd_output.squeeze(-1))`
（`AIDGN.py:119-126`），`abs()` 在最后。

### 2.2 参数量与容量门

| 方法 | V0 参数 | `d_head` | `pe_proj` | V1 参数 | 增量 |
|---|---:|---:|---:|---:|---:|
| `aidgn` | 30,070 | 56 | 504 | 30,574 | **1.68%** |
| `two_phase` | 12,969（FD001/3） | 32 | 288 | 13,257 | **2.22%** |
| `dual_mixer` | 64,123 | 960 | 8,640 | 72,763 | **13.47%** |
| `ffve` | 207,480 | 200 | 1,800 | 209,280 | 0.87% |
| `fcstgnn` | 19,921（FD001） | 8 | 72 | 19,993 | 0.36% |

过 1% 门槛的三个方法（`aidgn` / `two_phase` / `dual_mixer`）加 sham 臂；
`ffve`（0.87%）与 `fcstgnn`（0.36%）不过线，V1 对 V0 即等容量比较，不加臂。

`two_phase` 与 `fcstgnn` 的参数量随子集变化（`two_phase` 12,969/24,289，`fcstgnn`
19,921–592,989）；上表取最小子集，增量按该子集计，报告时逐子集列出。

---

## 3. 协议：与 V0 逐项相同

V1 与 V0 **同预算、同调度、同早停、同种子、同 FP32**。以下四条按 V0 代码（`d7bd9c17`）写死：

1. **训练窗 = all-cycle**。`data.py::prepare` 对 `train` 调 `windows(tr, tr_values, cfg["window"], average_life)`，
   **不传 cuts**，即 `points = range(length, n+1)`；只有 `val` 用 5 个 `round(f·n)` 截点
   （f = 0.40/0.55/0.70/0.85/0.95）。README 里「complete-only training windows」那句**已过时**，
   以 `data.py` 为准。
2. **选轮 = pooled engine-cut validation RMSE**。V0 的 `FREEZE.json` 明写
   `selection = "minimum pooled engine-cut validation RMSE"`；V1 沿用，否则 V0 得重跑。
3. **FC-STGNN 41 轮固定、不早停**。
4. **FFVE 的训练轮数上限 100000 不得改小**。

为了「统一」而给 FC-STGNN 加早停、或把 FFVE 上限砍到几百，恰恰是破坏同预算。

每方法其余预算（`configuration()`）：`aidgn` 300 轮 / pat12 / batch128 / lr1e-3 /
ReduceLROnPlateau；`two_phase` 200 或 600 轮 / pat12 / batch256 / lr1e-3 或 1e-4 +
配对预训练 20 或 100 轮；`dual_mixer` 100 轮 / pat12 / batch204 / lr1e-3；
`ffve` 100000 上限 / pat10 / batch128 / lr5e-3 或 1e-3 / StepLR(10)；`fcstgnn` 41 轮 / batch100。

**`dual_mixer` 不冻结基座**：V1 与 V0 一样全量训练。冻结会把协议从「全量」改成「只训头」，
使 V1−V0 变成协议变更而非插件效应；插件不改它的 `loss`，梯度自然流进 `pe_proj`。
（旧方案的 5,057 参数残差头 + 冻结基座**已撤回**，原因是它改了架构与训练机制。）

---

## 4. sham 臂（容量对照）

**定位**：等容量下的编码特异性对照。参数量、计算量、训练预算与 V1 逐项相同，唯一被破坏的是
「这段编码对应的是这台发动机的真实年龄」。

**构造**：在 subset × split 内，对 cycle 向量做一次**均匀随机置换**，每个 (subset, seed) 固定一个，
不每 epoch 重抽。多重集精确保留（边际分布不变）；置换必须非恒等。
记录：置换索引数组、其 SHA-256、以及置换后 cycle 与真实 RUL 的相关系数（应接近 0）。

**信息边界**：sham 只置换插件的编码，**不置换 `age`**。`aidgn`/`two_phase`/`ffve`/`fcstgnn`
的 V0 已经把归一化 `age` 当输入，所以 sham 是「编码特异性」控制，**不是** cycle 信息消融。
要做信息消融得同时置换 `age`，那是另一个实验，本轮不做。

**`two_phase` 的配对预训练必须用同一置换**，否则真实 cycle 会从预训练泄漏进 sham 臂。

**硬条件**：sham 与 V1 的 `params` 与 `trainable_params` 逐方法逐子集相等；两者的
`config.json` diff 只允许 `sham` 标志与 `permutation_seed` 不同；`split.json`/`scaler.json`
哈希相同。

---

## 5. 指标与判据

**指标**：engine-level RMSE，从每格 `validation_predictions.csv` 重算——先对每台发动机的 5 个
截点算 RMSE，再对发动机等权平均。`metrics.json::val_engine_rmse` 是 **pooled**，禁用
（第一半实测最大差 6.1382）。

**配对**：同方法、同子集、同 seed、同发动机的逐机差；四子集**等权**合并；95% CI 用
分层 bootstrap（子集内对发动机有放回，10,000 次，RNG `20261114`），点估计用观测值。
另报 Bonferroni 98.33% CI、方向一致计数、按机数加权敏感性。

### 5.1 主张 R（重编码增量）

范围：`aidgn` / `two_phase` / `ffve` / `fcstgnn` 四个方法。它们已吃归一化 `age`，测的是
**Fourier 重编码在已有 cycle 输入之上的增量**，不是「cycle 信息有没有用」。

主比较逐方法不同：
- `aidgn`、`two_phase`（有 sham 臂）：**V1 − sham**；
- `ffve`、`fcstgnn`（无 sham 臂）：**V1 − V0**。

**强结论**：≥3/4 方法各自通过门槛 **且** 各自主比较的合并 95% CI 不含 0。
**弱结论**：只有方向一致（≥3/4 方法在 ≥3/4 子集上点估计为负），合并 CI 跨 0。

「≥3/4 方法」是**逐方法过门槛的计数**，不是把四个 Δ 跨方法平均。

### 5.2 主张 I（信息效应）

只在 `dual_mixer` 上单方法报告 **V1 − sham**（n = 1），不做跨方法合并。
`dual_mixer` 是唯一没有 `age` 输入的方法，V1 与 sham 同容量（72,763）、同全量训练、
只差编码置换，所以这是一个干净的单方法信息对照。

**多方法合并的信息效应不在本套件内**，归第一半 DSPT（`none` 臂没有任何 cycle 通道）。

### 5.3 标签

- 有 sham 臂的三方法：**V1 − sham = 等容量信息对照（主）**；
  **V1 − V0 = 容量 + 信息（次要，不得当信息效应读）**。
- 无 sham 臂的两方法：V1 − V0 即主比较（容量差 <1%，可忽略）。

**五个方法不得平权平均成一个「Cycle 效应」**：`dual_mixer` 单独一列，另外四个一列，标签不同。

---

## 6. 边界与披露

- 官方 TEST 不参与任何选择；本阶段只报验证集。
- 第二半的绝对 RMSE 与第一半**不可比**（V0 用 pooled 选轮、第一半用 engine-mean 选轮），
  只能比 within-method Δ。
- 失败、缺失、未完成如实记录，不重试到成功；不因结果不理想重跑已落地的格。
- `pre_norm` 是**默认位置**，不是胜出位置。正文不得写「最优位置」。
- `two_phase` 的注入语义是 `post_norm`（头输入前有 LayerNorm），报告里逐方法标。

---

## 7. 验收

```bash
python scripts/probe_cycle_injection.py     # 五个方法的头、d_head、梯度自检
python scripts/probe_dual_mixer_step.py     # dual_mixer 真实训练步
```

冻结时的自检输出：五个方法 `pe_proj.weight.grad is not None`、头权重有梯度；
`dual_mixer` 真实训练步 `loss=0.8855`、`pe_proj` 梯度范数 0.2713、128/128 参数张量有梯度、
优化器步进成功；sham 置换 `identity=False`、`same_multiset=True`。
