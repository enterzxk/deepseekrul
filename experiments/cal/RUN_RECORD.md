# 校准层：实验运行记录与数据来源

本文件是 `rul-calibration` 分支的全部实验资产的来源凭据（provenance record）。
任何人应当能仅凭本文件 + 仓库内容，复现 `experiments/cal/eval_{A,BE,PE}/eval_report.json` 里的每一个数字。

## 0. 证据等级声明（必须最先读）

**当前全部结果标记为探索性（exploratory），不是确认性验证（confirmatory）。**

理由是可核查的事实，不是谦辞：

1. 本研究在形成最终结论之前，已多次读取 FD001–FD004 的官方测试集指标。至少包括：CET 七个缺陷的排查、CIDO 的两次可辨识性实验、R1/R2/R3 滤波器的五轮对照、以及校准层的十余次迭代（尺度选择 bug、均值保持消融、PIT 原子问题、三骨干对照）。
2. 校准层的若干设计决定是在看到测试结果之后做出的——`choose_scale` 从"全局最小偏差"改为"达到名义水平的最小尺度"、放弃均值保持约束、PIT 改为随机化处理。这些修改都有独立的正确性理由（见 `paper/CALIBRATION_LAYER_RESULTS.md` §5），但它们的**触发**来自测试集表现。
3. 因此，任何后续针对这些数据所做的改动，都**不能**声称为全新确认性验证。真正独立的确认证据只能来自第四阶段的 N-CMAPSS / XJTU-SY 实验，或一次事前注册、只读一次的新划分。

在此声明之前的所有数字，定位是"探索性证据，支持方法可行性的初判"，不是"论文主表"。
论文主表必须由第四阶段的冻结协议 + 独立数据产生。

## 1. 数据流

```
C-MAPSS 原始 train 文件 (data/CMAPSS/train_FD00x.txt)
  └─ split_frames(subset, fold)          引擎级 5 折，SPLIT_SEED=7，val=20%/折
       ├─ tr  ──► 骨干训练（A / BE / PE，各 3 seeds × 5 folds）
       │            └─ 检查点 e21.ckpt_path(subset, arm, seed, fold)
       └─ va  ──► build_windows_matched(cutoff_seed=2026)   部署式匹配窗口
                    └─ 15 个模型逐个预测 → scripts/cal/dev_preds/{subset}/{arm}_s{seed}_f{fold}.csv
                         └─ 按发动机对 3 个 seed 取均值 → 709 条折外 dev 预测
                              ├─ 拟合 CalibNet（右删失 NLL）
                              ├─ 5 折交叉拟合选尺度（名义 90%）
                              └─ 应用到官方测试预测
C-MAPSS 原始 test 文件 + RUL_FD00x.txt
  └─ 15 模型集成 → results/dsh/e21/preds/{subset}_{arm}.csv（12 个文件，已入库）
```

### 1.1 骨干

| 项 | 值 |
|---|---|
| 臂 | A（主模型，41,858 参数）、BE、PE（对照架构） |
| 训练 | 5 折 × 3 seeds（2027/2028/2029），共 45 个检查点/子集/臂 |
| 窗口 | `WINDOW=50`，`MAX_RUL=125`，分段线性标签，`LABEL_MODE="cap"` |
| 归一化 | FD001/FD003：`SensorMinMaxScaler`；FD002/FD004：`OperatingConditionScaler(n_clusters=6)` |
| 检查点位置 | lab-a40 `/home/yndx/zxk/bearing-rul-pi/experiments/{e3_fd001,e18,e10,e20}/...`，共 1066 个 `.pt` |
| 检查点入库 | 否。按 `.gitignore` 惯例排除；权重可由同代码同种子复现（见 `paper/E4_RESULT.md`） |

**注意（dev/test 集成规模不一致）**：dev 预测是每发动机 3 个 seed 的均值（3 模型集成），测试预测是 5 折 × 3 seed 的 15 模型集成。集成规模差异会系统性降低测试残差方差，是 dev→test 覆盖漂移的一个已识别来源，尚未校正。第三阶段须量化。

### 1.2 折外 dev 预测

- 每台发动机恰好在 1 个折中作为 `va`，由 5 个从未见过它的模型（该折对应检查点）中的 3 个 seed 预测。
- 因此**不存在发动机级泄漏**：拟合校准层的每一行，其 y_pred 都来自没见过该发动机的模型。
- 行数：FD001 100、FD002 260、FD003 100、FD004 249，合计 **709**。
- 文件：`experiments/cal/dev_preds/{subset}/{arm}_s{seed}_f{fold}.csv`，列 `unit_id,y_true,y_pred`。
- `y_true` 已按 `min(125, T−t)` 封顶。

### 1.3 校准/尺度选择/评价数据的分工

| 用途 | 数据 | 是否接触测试标签 |
|---|---|---|
| 拟合 `CalibNet` 权重 | pooled dev（709 条），pooled 协议 | 否 |
| 选择尺度 `scale` | 同上，5 折**交叉拟合**（每折的层不曾在被评折上训练） | 否 |
| 评价 | `results/dsh/e21/preds/{subset}_{arm}.csv` | 是（仅评价） |

`choose_scale` 内部再对 dev 做 5 折交叉拟合，所以名义覆盖是构造性质：它只依赖 dev 的折外预测。测试标签从不参与拟合或尺度选择，只用于最后的评分。

### 1.4 测试集实际样本量（不是官方全集）

| 子集 | 官方 test 发动机 | 本研究评价 n | 封顶样本占比 |
|---|---:|---:|---:|
| FD001 | 100 | 93 | 0.10 |
| FD002 | 259 | 235 | 0.17 |
| FD003 | 100 | 97 | 0.13 |
| FD004 | 248 | 228 | 0.21 |
| 合计 | 707 | **653** | 0.16 |

差额来自匹配截止协议 `build_windows_matched`：只保留最后观测周期恰好落在匹配截止点的发动机，被提前截断的发动机不进入评价。**论文中必须报告 653，不能报 707。**

## 2. 三种协议的精确定义（不得混用措辞）

| 协议 | 拟合数据 | 目标舰队 dev 是否参与 | 正确表述 | 错误表述 |
|---|---|---|---|---|
| `per_subset` | 目标舰队自己的 dev | 是 | 舰队内校准，需要该舰队的标注 dev 集 | — |
| `pooled` | 四个舰队 dev 的并集（709） | 是 | 多源开发数据联合校准 | **"零样本迁移到未知舰队"**（错：目标舰队的 dev 已在训练集里） |
| `loso` | 除目标舰队外的三个舰队 dev | 否 | 不使用目标舰队标注数据的迁移 | "零样本"（也不准确：仍用了同族数据） |

`pooled` 是最强结果，但它的语义是"目标舰队有标注 dev 可用，且与其它舰队联合拟合"。任何把它写成零样本迁移的表述都是错的，审稿人会抓。

## 3. 结果文件清单

| 文件 | 内容 |
|---|---|
| `experiments/cal/eval_A/eval_report.json` | A 臂 × 四舰队，pooled 协议，含 bootstrap CI、分箱覆盖、PIT |
| `experiments/cal/eval_BE/eval_report.json` | 同上，BE 臂 |
| `experiments/cal/eval_PE/eval_report.json` | 同上，PE 臂 |
| `experiments/cal/eval_pooled/eval_report.json` | 早期 pooled 运行（单臂 A，2000 次 bootstrap），保留作时间戳证据 |
| `experiments/cal/eval/eval_report.json` | 更早的 per-subset 协议运行，**已被 eval_A 取代**，仅存档 |
| `experiments/cal/pooled/calibration_report.json` | 三种协议的 PICP90 消融（pooled） |
| `experiments/cal/loso/calibration_report.json` | 同上（loso） |
| `experiments/cal/per_subset/calibration_report.json` | 同上（per_subset） |
| `experiments/cal/calibration_report.json` | 最早的单臂 per-subset 运行，仅存档 |
| `experiments/cal/shift_report.json` | dev→test 残差分布漂移诊断 |
| `experiments/cal/calibrators/calibrator_{A,BE,PE}.json` | **冻结的 66 参数层**：权重、scale、拟合目标、dev 行数、dev 数据摘要 |
| `experiments/cal/MANIFEST.sha256` | 上述全部文件 + 180 个 dev 预测 CSV 的 SHA256 |

**权威结果 = `eval_{A,BE,PE}/eval_report.json`（20:53–20:54 生成）。** 其余目录是迭代痕迹，不得引用其数字。

### 3.1 冻结校准器与端到端校验

`calibrator_<arm>.json` 存的是层的**全部 66 个参数**加上 `scale` 与名义水平。这是刻意的：
一个 66 参数的层由权重完全决定，把它存下来，报告里的每个数字就成为"已提交文件的函数"，
不再依赖拟合过程的随机种子或某台机器上的检查点。

`scripts/cal/verify_frozen.py` 做这件事的校验：只用 JSON + `results/dsh/e21/preds/*.csv`，
手工重建 pmf，再和 `eval_report.json` 逐项比。实测：

| 臂 | max\|冻结复算 − 报告\|（CRPS / PICP90 / MPIW90） |
|---|---:|
| A | 1.05e-06 |
| BE | 1.45e-06 |
| PE | 2.45e-06 |

误差在浮点精度内。**这意味着校准层的结论不依赖检查点是否还在 A40 上。**
`dev_sha256` 是 dev 预测行的摘要：若预处理变化导致 dev 预测改变，摘要会变，记录会显式失效而不是悄悄出错。

## 4. 环境与版本

| 项 | 值 |
|---|---|
| 机器 | lab-a40（A40），共享机，运行时 load 曾达 49 |
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu121 |
| NumPy / pandas | 1.26.4 / 2.0.3 |
| 线程 | `OMP_NUM_THREADS=4`，脚本内 `torch.set_num_threads(4)`（不设会因线程争用慢 20 倍） |
| 代码版本 | 本分支首次提交（见 `git log rul-calibration`） |

A40 上**不是** git 仓库：代码以 scp 同步，因此代码版本以本仓库提交为准。脚本的 SHA256 见 MANIFEST。

## 5. 复现命令

```bash
export RUL_ROOT=$PWD PYTHONPATH=$PWD/src:$PWD/scripts OMP_NUM_THREADS=4

# 1) 从检查点重建折外 dev 预测（需 1066 个 .pt，仅在 lab-a40 上存在）
python scripts/cal/make_dev_preds.py --subsets FD001,FD002,FD003,FD004 --arms A,BE,PE

# 2) 三骨干 pooled 评估（含 bootstrap CI、分箱、PIT）
for arm in A BE PE; do
  python scripts/cal/evaluate.py --stage $PWD --mode pooled --arm $arm \
         --n-boot 2000 --out experiments/cal/eval_$arm
done

# 3) 协议消融
for m in per_subset loso pooled; do
  python scripts/cal/calibrate.py --stage $PWD --subsets FD001,FD002,FD003,FD004 \
         --mode $m --out experiments/cal/$m
done

# 4) 打印表
python scripts/cal/print_table.py experiments/cal/eval_A/eval_report.json
```

单臂单协议在 A40 上约 21 秒（限线程后）。第 1 步需要检查点，耗时约 10 分钟。

## 6. 已知未修正的问题（第三阶段处理）

1. **dev 是 3 模型集成、test 是 15 模型集成**——覆盖漂移的一个来源，未校正。
2. **`y_true_raw` 未用于删失建模**：现在把所有 `y >= 125` 当作右删失，但其中一部分的真实 RUL 已知（`y_true_raw` 列）。严格说应使用真实失效时间做删失，而不是"封顶即删失"。
3. **FD003 `[75,100)` 段覆盖 0.67**（n=18，A 臂）：整体覆盖率掩盖的局部失效。BE 0.78、PE 0.72 在同段同样失效，且三者偏差同向（+6.4/+7.0/+7.9），说明这是**骨干的条件偏差**被校准层忠实传递，不是校准层引入的。第三阶段须按此定位并量化 CI（n=18 时 95% CI 极宽，须报样本量）。
4. **种子只有 3 个**：骨干的不确定性未被传播到校准层，bootstrap 只反映发动机抽样变异。
5. **bootstrap 以发动机为单位，但没有按折/种子聚类**——同一发动机在 dev 中只出现一次（seed 已平均），故无重复，但测试侧每发动机 1 行，聚类结构简单。第三阶段需明确写出聚类单位。
