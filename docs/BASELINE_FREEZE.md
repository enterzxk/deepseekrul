# 基线冻结清单（BASELINE FREEZE）

> **依据**：D016 下一步第 **2** 项——「冻结代码/配置/划分/seed/预测文件/指标实现的对应关系，
> 形成可追溯基线清单；无法复算的覆盖对齐材料继续排除」。
>
> **状态**：已完成（2026-11-09）。执行化验收 `scripts/freeze_audit.py` **50/50 PASS**。
>
> **冻结对象**：当前 CQR 方法（D016 判定的「冻结为基线」的那个方法）。本文件不含新方法、
> 不含方法选择、不重跑任何训练。所有数字均从本地产物重算，未引用记忆或转述。

---

## 0. 这份文件解决什么问题

D009/D016 之前，论文数字分散在 `results_summary.md`、`decision_log.md`、`research_state.md`、
`docs/tables/*.csv` 与 80 个 JSON 里，且存在**同一实验多个数字**（D017 就是例子）。
本文件建立**唯一映射**：

```
主张中的数字  ←  指标实现  ←  预测文件  ←  训练配置 + 划分 + seed  ←  代码（提交号 + 哈希）
```

并给每一项标注**可复算性等级**。等级决定它能出现在哪里：

| 等级 | 含义 | 允许用途 |
|---|---|---|
| **A 本地可复算** | 本机有预测/JSON 产物 + 冻结脚本，重算与记录一致（已实测） | 可进论文主表、可被审稿人复核 |
| **B 仅远端可复算** | 需要 a40 的 checkpoint 或重训才能重建 | 可作内部证据；论文中须注明复现条件 |
| **C 不可复算** | 源产物已不存在，或无法确认产生它的实现版本 | **排除出论文证据体系**，仅作内部记录 |

---

## 1. 冻结对象：方法的一句话定义

**IDA-PE 主干 + 端到端双分位头 + split-conformal CQR。**

- 主干：`scripts/run_ida.py::CandidateModel`（`d_model=64, num_layers=1, dropout=0.15,
  window=50, use_pe=True, use_ema_features=True, use_graph=False`），M1 先验经零初始化投影注入。
- 区间头：`scripts/run_e2e_cqr.py::E2ECQR` —— `centre = W·ctx`，`half = softplus(W·ctx)`，
  输出 `[centre − half, centre + half]`（`ctx = mean_pool(body) + proj(prior)`）。
- 训练：与主干**联合**训练（不是后处理头），损失
  `pinball(lo, y, 0.05) + pinball(hi, y, 0.95) + 0.1 · MSE(centre, y)`；
  AdamW(lr=1e-3, wd=1e-4)，CosineAnnealing(T_max=80)，grad_clip=1.0，epochs=80，patience=20，
  以验证集 centre-RMSE 选 checkpoint。
- 校准：`src/punet/cqr.py::CQRCalibrator`，CQR 分数
  `E_i = max(q_lo(x_i) − y_i, y_i − q_hi(x_i))`，
  `q̂ = Quantile_{(1−δ)(1+1/n)}({E_i})`（`method="higher"`，`δ=0.1`），
  区间 `[q_lo − q̂, q_hi + q̂]`。
- 点基线臂：`--heads point`，损失退化为 `MSE(centre, y)`，预测文件里 `y_lo = y_hi = centre`，
  于是 CQR 校准退化为**绝对残差 split conformal**（常数宽度）。这就是论文里的「常数宽度基线」。

> **冻结含义**（D016）：不再对该方法做任何改动；它只作为新问题的基线与参照。
> 任何后续改动必须作为**新立项**处理，不得回写本文件。

---

## 2. 代码指纹

| 角色 | 文件 | 最后提交 | SHA256[:16] | 备注 |
|---|---|---|---|---|
| 训练/预测驱动（80 格 + e2e 共用） | `scripts/run_e2e_cqr.py` | `4b896fd` (11-08) | `EE3300D3E147E7F9` | 单一入口，`--heads cqr/point/qr_raw`、`--ablate none/A4` |
| 主干与数据管线 | `scripts/run_ida.py` | `a920d4d` (11-03) | `D897C28D529D03B0` | `OUTER_SEED=20261104`、`CYCLE_SCALE=250`、clip125 |
| CQR 校准器 | `src/punet/cqr.py` | `dd9467a` (11-07) | `3F747BFDF6B842EB` | 分数与分位数定义 |
| 指标实现（权威） | `src/punet/metrics.py` | `762e418` (11-07) | `6F4CE1052400051F` | `picp / pinaw / winkler(α) / engine_coverage` |
| 评估包装 | `src/punet/evaluate.py` | `d0ad488` (11-08) | `55C754ECCAE3111F` | 写 `engine_min/p05/median`、`rmse`、bootstrap CI |
| 划分函数 | `src/punet/conformal.py` | `d0ad488` (11-08) | `F35B809F42E7FBCA` | `engine_calib_split(frac, seed)` |
| 80 格覆盖对齐表 | `scripts/table_all_methods.py` | `6ec2a21` (11-08) | `EDB1086A850873EF` | **自带一份 winkler 副本**，见 §7.3 |
| 论文数字断言 | `scripts/verify_paper_numbers.py` | `975cea4` (11-08) | `B171D4D3E6B02978` | 17 项，本地实测 17/17 PASS |
| 冻结验收（本文件） | `scripts/freeze_audit.py` | 新增（未提交） | `270A72B31FDBB91C` | 50 项断言 |
| FEMTO 评测 | `scripts/evaluate_femto.py` / `run_femto_cqr.py` | `4b896fd` (11-08) | `710F20C6BF1C840A` / `3481BD11206B16C5` | 6 轴承 |
| MC-Dropout 预测 | `scripts/run_mcdropout.py` | `e3c3e79` (11-08) | `44BF3E087A8A5CD8` | **工作区有未提交改动**（80 格回填，见 §7.5） |
| 稿件（数字来源） | `docs/manuscripts/v1.0-draft.md` | `18c6710` (11-08) | `F30B57D7C141B4F4` | `verify_paper_numbers.py` 读它 |
| 80 格证据表 | `docs/tables/cells_all.csv` | `c86c90b` (11-08) | `4CBE621E9C7EF34B` | 与 a40 同名文件数值完全一致 |

仓库 HEAD = `ce7965c`（分支 `punet-interval`）。工作区有未提交改动（`AGENTS.md` 的 §0.0、
`manuscript.tex` 一个空行、`scripts/run_mcdropout.py`、以及并行线的 `research/`）——
本文件**不依赖**这些未提交改动即可复算（`freeze_audit.py` 只读已提交产物 + 本地 preds）。

---

## 3. 协议链（逐项参数，全部可在代码中定位）

### 3.1 数据

| 项 | 值 | 位置 |
|---|---|---|
| 数据集 | C-MAPSS FD001–FD004（`train_FD00x.txt` / `RUL_FD00x.txt`） | `data/CMAPSS/`（本地 13 个 txt，gitignore） |
| 训练目标 | `min(125, last_cycle − cycle)`，piecewise-linear | `run_ida.py` `build_piecewise_linear_rul` |
| 窗口 | `window=50`，每台发动机从 cycle 50 起逐周期滑窗 | `run_ida.py::prep_fold` |
| 工况处理 | FD002/FD004：6 簇 KMeans 条件归一化 + `pop_ref`；FD001/FD003：普通 | `run_ida.py::prep_fold` |
| 先验 | M1 staircase prior，`--prior-dir experiments/ida/prior_boot` | `run_ida.py::prior_vectors` |

> **注意**：这里的「test」是**训练数据内部的留出折**（官方 test 集不参与），
> 与官方 `test_FD00x.txt` 无关。官方 test 只用于最终点预测评估，不用于区间主结果。

### 3.2 折划分

| 项 | 值 |
|---|---|
| 划分 | `KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)`，按**发动机**切 |
| `OUTER_SEED` | `20261104`（`run_ida.py:45`，须与 `src/lifetime_prior/nested_prior.py` 一致） |
| 每折验证发动机数 | FD001 20 / FD002 52 / FD003 20 / FD004 50 |
| 折内再分 | 验证折内再按发动机 30% 校准 / 70% 测试（见 3.4） |

**5 折对所有 seed 相同**：seed 只改初始化与训练轨迹，**不改引擎划分**。
这是「80 格 ≠ 80 个独立证据」的结构性原因（D014/D015）。

### 3.3 训练 seed

| 阶段 | seed | checkpoint |
|---|---|---|
| Stage A（40 格） | `42`, `7` | ✅ `--save-ckpt`（a40 上 40 个 `model.pt`） |
| Stage B（40 格） | `2024`, `13` | ❌ 未保存（`run_more_seeds.sh` 无 `--save-ckpt`） |

目录名即记录：`experiments/punet_matrix/{FD00x}/f{fold}_s{seed}/`。

### 3.4 校准 / 测试划分（区间评估的实际单位）

```
engine_calib_split(units, calib_frac=0.3, seed=42)
  rng = np.random.default_rng(seed); uniq = np.unique(units); rng.shuffle(uniq)
  n_cal = max(1, round(uniq.size * 0.3));  calib = uniq[:n_cal];  test = uniq[n_cal:]
```

| 子集 | 验证折发动机 | 校准发动机 | 测试发动机 |
|---|---|---|---|
| FD001 | 20 | **6** | 14 |
| FD002 | 52 | **16** | 36 |
| FD003 | 20 | **6** | 14 |
| FD004 | 50 | **15** | 35（f4 实际 34） |

**两种用法必须分清**：

1. **e2e 20 格**（`experiments/punet_e2e/*/f*_s42/*.json`）：运行期用 `seed=args.seed`（=42）校准，
   `q̂` 写进 JSON。
2. **80 格表**（`docs/tables/cells_all.csv`）：分析期由 `table_all_methods.py` 用**固定 seed 42**
   重新划分并**重新校准**所有 80 格。因此 80 格是**同一校准/测试引擎划分上的配对比较**
   （const vs CQR 用同一批引擎），训练 seed 只改权重。

> **已验证**（`freeze_audit.py` H1）：对 seed-42 格子，分析期重算的 `q̂` 与运行期 JSON 中的
> `q̂` 逐位一致（FD001 f0：7.303215 vs 7.303215；FD002 f0：−0.933046 vs −0.933046）。
> 即两处划分实现等价，不存在「分析期偷偷换划分」。

### 3.5 指标实现（唯一权威：`src/punet/metrics.py`）

| 指标 | 定义 | 备注 |
|---|---|---|
| PICP | `mean(y ≥ lo ∧ y ≤ hi)` | 边际覆盖 |
| PINAW | `mean(hi − lo) / (max(y) − min(y))` | 归一化宽度 |
| Winkler | `mean((hi−lo) + (2/α)((lo−y)₊ + (y−hi)₊))`，**α 固定 0.1** | D005 修正后的口径 |
| engine_coverage | 逐发动机覆盖率 → `min / p05 / median / mean` | 引擎级失效的证据来源 |

**α 纪律**：所有跨方法比较固定 α=0.1；覆盖对齐（若使用）只用于在**各自前沿上找达到 0.90 的 δ**，
不改变评分规则（D005）。

### 3.6 聚合单位（三层，不得混用）

| 层 | n | 用法 |
|---|---|---|
| 格（cell） | 80 = 4 子集 × 5 折 × 4 seed | 明细/附录；**非独立** |
| 折（fold） | 20 = 4 子集 × 5 折（seed 先平均） | **主表单位**（D015） |
| 发动机（engine） | 14 / 36 / 14 / 35 每折 | 引擎级覆盖的唯一合法单位 |

---

## 4. 产物清单与本地/远端对应

| 产物 | 本地 | a40 | 一致性 |
|---|---|---|---|
| 80 格 CQR 预测 | `experiments/punet_matrix/*/f*_s*/preds_cqr.csv`（80） | 80 | md5 抽样一致 |
| 80 格点基线预测 | 同目录 `preds_point.csv`（80） | 80 | 同上 |
| MC-Dropout 预测 | `preds_mcd.csv`（**10**） | 10 | 同上 |
| 训练 checkpoint | 0（matrix） | **40**（仅 seeds 42/7） | 本地无 |
| e2e JSON | `experiments/punet_e2e/*/f*_s42/*.json`（80） | 80 | md5 抽样一致 |
| e2e checkpoint | 4（各子集 f0_s42） | 4 | 一致 |
| e2e preds 明细 | 4（各子集 f0_s42 `preds.csv`） | 4 | 一致 |
| FEMTO | `experiments/femto/Bearing*/preds_{q,p}.csv`（6） | 6 | 一致 |
| 汇总表 | `docs/tables/{cells_all,failure_inventory,femto_cells}.csv` | 17 个 csv（多出已关闭路线的表） | `cells_all.csv` 数值 maxdiff = 0 |

**md5 抽样**（本地 vs a40，逐字节相同）：
- `preds_cqr.csv` FD001/f0_s42 → `afcc06416cdafbf05e0fe21c6494fe00`
- `e2e_cqr_none.json` FD001/f0_s42 → `1874cfe2904d84762e80cfe3d2a61d92`

> 本地 `docs/tables/cells_all.csv` 与 a40 同名文件**字节不同（5711 vs 5630 B）但数值完全相同**
> （const/cqr/loc 三列 maxdiff = 0.0，`mcd` 均 10 个非空、`qr` 均全空）。
> 差异来自浮点打印格式/行尾，不是数据差异。

---

## 5. 可复算性分级（本文件的核心结论）

### 5.1 A 类：本地可复算（已实测，`freeze_audit.py` 覆盖）

| # | 主张 | n | 复算方式 | 状态 |
|---|---|---|---|---|
| A-1 | CQR vs 常数宽度（覆盖对齐，固定 α=0.1） | 80 格 | `preds_{cqr,point}.csv` + `table_all_methods.py` 逻辑 | ✅ 50/50 中 B1–B6 |
| A-2 | 折级主表 | 20 折 | seed 先平均 | ✅ C1–C3 |
| A-3 | seed 方差披露 | 80 格 | 单因素方差分解 | ✅ C4–C7 |
| A-4 | 名义水平配对（CQR 区间 vs 点基线区间） | 20 格 | `e2e_*.json` 的 `winkler` | ✅ D1–D4 |
| A-5 | A4 置换 σ 消融 | 20 折 | `e2e_cqr_A4.json` vs `e2e_cqr_none.json` | ✅ E1–E3 |
| A-6 | 引擎级覆盖失效 | 20 折 | JSON 的 `engine_min/p05/median` | ✅ F1–F3 |
| A-7 | 覆盖–误差相关 | 20 折 | JSON 的 `picp / rmse / engine_min` | ✅ F4–F5（**循环性见 §7.2**） |
| A-8 | 跨域 FEMTO | 6 轴承 | `experiments/femto/*/preds_*.csv` | ✅ G1–G2 |
| A-9 | 论文数字断言 | 17 项 | `verify_paper_numbers.py` | ✅ 本地 17/17 PASS |

### 5.2 B 类：仅远端可复算

| # | 项目 | 为什么不是 A | 复现条件 |
|---|---|---|---|
| B-1 | 80 格**训练**本身 | 本地无 torch、无 matrix checkpoint；seeds 2024/13 连 a40 也没存 checkpoint | a40：`scripts/launch_matrix_80cells.sh`，≈80 格 × 20 min |
| B-2 | e2e 20 格训练 | 仅 4 个 f0 checkpoint | a40 重训（每格 ≈20 min） |
| B-3 | MC-Dropout 全 80 格 | 只有 10/80；回填正在跑（`research_task mcd80-backfill`） | 见 §7.5 |
| B-4 | 已关闭路线的明细表（width_scaling / d002_centre / asymmetric / mondrian / engine_vs_inflation / ensemble_scaling / grid_sensitivity / protocol_audit / h004/） | 本地只有 3 个 csv；a40 有 17 个 | a40 `docs/tables/`；**只作内部记录，不进论文** |

### 5.3 C 类：不可复算（排除出论文证据体系）

| # | 项目 | 为什么不可复算 | 处置 |
|---|---|---|---|
| C-1 | **29/40 覆盖对齐**（D004 记载：FD003 −1.6%、FD004 −0.8%） | 原始表不在本地也不在 a40 的 `docs/tables/`（17 个 csv 中无对齐明细）；且**无法确认它是否用 D005 修正后的评分实现**（D005 记录该协议曾有 α 漂移 bug） | **D015 已删除出证据体系**。内部记录保留，论文不得引用，不得写「覆盖对齐后接近打平」 |
| C-2 | H004 引擎-阶段覆盖正则 | 未立项即关闭（D013），无产物 | 不得写成机制贡献 |
| C-3 | `wshift` / `instnorm` 两个方向 | 早期在**官方 test** 上判定成败（违反口径纪律） | 永久停止，不进任何表 |
| C-4 | γ 自适应（`results_summary.md` §9 #8） | 校准集 ρ̂=+0.064、测试集 ρ̂=−0.084，符号翻转 → 含泄漏 | 作废（原文已标） |
| C-5 | 「σ_T 是有效机制」 | D017 重算：A4 置换 **+0.246%**、11/20 折变差、FD004 反而 −3.02% | 结论改为 **σ_T 冗余**；`results_summary.md` §1 已划线撤回 |
| C-6 | 「MC-Dropout 尺度无效」的普遍陈述 | 只有 10/80 格 | 只能限定为「这 10 格上」；CRULP 的 UACQR ≠ MC-Dropout（`difftable_crulp_ijphm.md` §4） |

---

## 6. 已验证数字（`freeze_audit.py` 实测输出，2026-11-09）

> 下表每个数字都有断言覆盖；`scripts/freeze_audit.py` 退出码 0 即全部成立。

### 6.1 80 格（覆盖对齐，固定 α=0.1，CQR vs 常数宽度）

| 统计量 | 值 | 断言 |
|---|---|---|
| 胜场 | **75/80** | B1 |
| 均值 | **−9.087%** | B2 |
| 中位 | **−10.754%** | B3 |
| FD001 / FD002 / FD003 / FD004 | **−7.568% / −14.067% / −6.492% / −8.220%** | B4 |

### 6.2 折级（seed 先平均，n=20，**主表单位**）

| 统计量 | 值 | 断言 |
|---|---|---|
| 胜场 | **19/20** | C1 |
| 均值 | **−9.087%** | C2 |
| 中位 | **−9.687%** | C3 |
| 折内 seed 方差占比 | **46.82%** | C4 |
| 折内 seed 极差 中位 / 最大 | **6.13 pp / 36.63 pp** | C5 / C6 |
| 四 seed 同向的折 | **16/20** | C7 |

### 6.3 名义水平（e2e，单 seed 42，n=20）

| 统计量 | 值 | 断言 |
|---|---|---|
| 胜场 | **19/20** | D1 |
| 中位 | **−8.675%** | D2 |
| 逐子集（**ratio-of-means**） | −5.254% / −16.206% / −8.832% / −6.990% | D3 |
| 逐格比值的均值（另一口径） | −9.310% | D4 |

> **口径警告**：`audit_closeout.md` §4 的逐子集 e2e 数字用的是 **ratio-of-means**
> （先按子集汇总 Winkler 再求相对变化），而 19/20 与中位数用的是**逐格比值**。
> 两者差最多 0.5 pp，引用时必须写明是哪一种（见 §7.2）。

### 6.4 A4 置换 σ（n=20）

| 统计量 | 值 | 断言 |
|---|---|---|
| 均值 | **+0.246%** | E1 |
| 变差的折 | **11/20** | E2 |
| FD001 / FD002 / FD003 / FD004 | +2.539% / +0.238% / +1.225% / **−3.020%** | E3 |

### 6.5 引擎级失效（n=20 折）

| 统计量 | 值 | 断言 |
|---|---|---|
| `engine_min < 0.80` 的折 | **18/20** | F1 |
| **真实最小单引擎覆盖**（逐子集取最小） | FD001 **0.4214** / FD002 **0.4500** / FD003 **0.5521** / FD004 **0.2814** | F2 |
| 折均值（= `failure_inventory.csv` 的 `engine_min` 列） | 0.690 / 0.529 / 0.714 / 0.501 | F2b |
| 最坏单元 | **FD004 f2：0.2814** | F3 |
| ρ(picp, engine_min) | **+0.725** | F4 |
| ρ(rmse, engine_min) | **−0.783**（**循环，不可部署**） | F5 |

### 6.6 跨域 FEMTO

| 统计量 | 值 | 断言 |
|---|---|---|
| 胜场 | **5/6** | G1 |
| 均值 | **−12.21%** | G2 |

---

## 7. 冻结过程中发现并纠正的口径问题

> 这一节是本文件的**增量价值**：四条都是本轮实算发现的，不是转述。

### 7.1 `failure_inventory.csv` 的 `engine_min` 列是「折均值」，不是「最小值」

- 表头写 `engine_min`，D018 正文称「引擎级最小」，但数值 = **各折 per-fold 最小覆盖的均值**
  （F2b：0.690/0.529/0.714/0.501 逐位吻合）。
- **真实最小单引擎覆盖更低**：FD001 **0.4214**、FD002 **0.4500**、FD003 **0.5521**、
  FD004 **0.2814**（F2）。
- 影响：**不改变**「18/20 折 engine_min<0.80」这一计数（该计数按折取，正确），
  也不改变「最坏 FD004 f2 = 0.281」；但**逐子集那一列的语义必须改写**为
  「各折最小引擎覆盖的均值」，否则读者会把 0.690 误当 FD001 的最坏发动机（实际 0.4214）。
- 处置：本文件为准；`failure_inventory.csv` 保留原列（不改历史），新立项引用时用 F2 的数值。

### 7.2 两个相关都是测试集导出量 → 只能用于「发现问题」

`rmse` 与 `engine_min` 都来自测试折，ρ=−0.783 是**循环**的，不能当作「可用 rmse 预测失效」。
可部署版本必须换成**校准集可观测**的量（校准集残差、校准集覆盖分布、轨迹长度、
协变量漂移统计）。这是 D018 已写明的限制，本轮用断言 F5 把它钉住。

### 7.3 指标实现有两份拷贝

- 权威：`src/punet/metrics.py::winkler`（α 作参数）。
- 副本：`scripts/table_all_methods.py::winkler`（α 硬编码 0.1）——**产出 `cells_all.csv` 的那份**。
- 两者公式一致；本轮以副本重算并与 `cells_all.csv` 对齐（B5/B6 max|diff| = 7e-15），
  证明副本没有引入偏差。**新立项必须调用 `src/punet/metrics.py`，不得再复制第三份。**

### 7.4 80 格共用同一校准/测试划分

80 格的校准/测试引擎划分固定为 `seed=42`（分析期），训练 seed 只改权重。
因此：
- 优点：const vs CQR 是**同一批引擎上的配对比较**，配对有效；
- 限制：**4 个 seed 不提供新的引擎划分**，只提供训练随机性；这再次确认 80 格 ≠ 80 个独立证据。

### 7.5 启动脚本此前未被版本控制

80 格的启动命令原本只存在于 `tmp/matrix2seeds.sh`、`tmp/run_more_seeds.sh`，
而 `/tmp/` 在 `.gitignore` 里（a40 上也没有同名脚本）。已逐字恢复为
`scripts/launch_matrix_80cells.sh`（带来源注释），否则「B-1 远端可复算」无法执行。

### 7.6 MC-Dropout 的工作区改动

`scripts/run_mcdropout.py` 有未提交改动（并行执行线为 80 格回填加了 `--save-ckpt` 等）。
本文件冻结的是**已提交版本** `e3c3e79` 产出的 10 格结果；80 格回填结果另立条目，
不得与本文件的 A-1 主表混算。

---

## 8. 明确排除在论文证据体系之外的项（汇总）

1. **29/40 覆盖对齐**与由此得出的「FD003/FD004 接近打平」（D015，C-1）。
2. H004 引擎-阶段覆盖正则（D013，C-2）。
3. `wshift` / `instnorm`（用 test 做方法选择，C-3）。
4. γ 自适应（泄漏，C-4）。
5. 「σ_T 是有效机制」（D017，C-5）。
6. 「MC-Dropout 普遍无效」（仅 10/80，C-6）。
7. 任何把 80 格当独立样本的显著性检验（D015）。

> 排除 ≠ 删除。以上均在 `results_summary.md` / `decision_log.md` 保留原文与撤回理由。

---

## 9. 冻结的可执行验收

```bash
# 本地（Windows，用绝对路径 python）
C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe scripts\freeze_audit.py

# 期望：50/50 checks passed，退出码 0
```

- 断言覆盖：产物清点（A）、80 格主表（B）、折级（C）、名义水平（D）、A4（E）、
  引擎级（F）、FEMTO（G）、划分同一性（H）。
- **任何后续改动使该脚本失败 = 破坏了基线**，必须先解释再继续。
- 与 `verify_paper_numbers.py`（17 项，读稿件文本）互补：后者管「稿件说了什么」，
  本脚本管「证据能否重算」。

---

## 10. 这个冻结**不**保证什么

1. **不保证引擎级覆盖**：真实最小单引擎覆盖 0.28–0.55，远低于名义 0.90。
2. **不保证 20 折独立**：5 折交叉验证训练集重叠（D015 已限定为描述性结论）。
3. **不保证跨 seed 稳定**：折内 seed 方差占 46.82%，仅 16/20 折四 seed 同向。
4. **不保证方法新颖性**：CRULP（TIM 2025）与 Robinson（IJPHM 2026）已占住「CQR + RUL」位置。
5. **不保证可投 TIM**：D016 已放弃当前版本的 TIM 目标；本文件只提供基线。
6. **不保证官方 test 集结果**：本文件全部数字来自训练数据内部的留出折。

---

## 11. 复现命令（逐条可执行）

```bash
# ---- a40（重训，B 类）----
ssh a40
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
bash scripts/launch_matrix_80cells.sh          # 80 格 = 4 子集 × 5 折 × 4 seed × 2 头

# ---- 本地（重算，A 类）----
python scripts/table_all_methods.py            # → docs/tables/cells_all.csv
python scripts/aggregate_diagnostics.py        # 覆盖诊断
python scripts/make_tim_tables.py              # 论文表
python scripts/verify_paper_numbers.py         # 17/17
python scripts/freeze_audit.py                 # 50/50
```

---

## 附录 A：恢复的 80 格启动命令（逐字，来源 `tmp/matrix2seeds.sh` + `tmp/run_more_seeds.sh`）

```bash
# Stage A —— seeds 42, 7，保存 checkpoint
for SEED in 42 7; do for F in 0 1 2 3 4; do
  $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads cqr \
      --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot \
      --out experiments/punet_matrix
  $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads point \
      --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot \
      --out experiments/punet_matrix
done; done

# Stage B —— seeds 2024, 13，不保存 checkpoint
for SEED in 2024 13; do for F in 0 1 2 3 4; do
  $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads cqr \
      --dump-preds --prior-dir experiments/ida/prior_boot --out experiments/punet_matrix
  $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads point \
      --dump-preds --prior-dir experiments/ida/prior_boot --out experiments/punet_matrix
done; done
```

e2e 20 格（`experiments/punet_e2e`）用同一驱动，`--out` 取默认值，并分别跑
`--heads cqr/point/qr_raw` 与 `--ablate A4`；仅 `f0_s42` 加了 `--save-ckpt --dump-preds`。

## 附录 B：每格产物结构

```
experiments/punet_matrix/{FD00x}/f{fold}_s{seed}/
  preds_cqr.csv     unit,cutoff,y_true,y_lo,y_hi,sigma_T,mu_T     # CQR 臂（未校准的 q_lo/q_hi）
  preds_point.csv   同上，y_lo=y_hi=centre                        # 点基线臂
  preds_mcd.csv     unit,y_true,mu,sd                             # 仅 10 格有

experiments/punet_e2e/{FD00x}/f{fold}_s42/
  e2e_cqr_none.json            picp/pinaw/winkler/engine_min/p05/median/rmse/q_hat/n_calib/n_test
  e2e_point_none.json          点基线臂
  e2e_cqr_A4.json              σ 置换臂
  e2e_qr_raw_none.json         未校准分位臂（仅 f0）
  e2e_point_none.EMA_stale.json 早期 EMA 变体（**作废**，勿引用）
  model.pt / preds.csv / preds_mcd.csv（仅 f0_s42）
```

> `e2e_point_none.EMA_stale.json` 是历史遗留的 EMA 评估，与 `e2e_point_none.json`
> 不是同一权重，**不得混用**。
