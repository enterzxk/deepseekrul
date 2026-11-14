# PrefixCal 实现面勘察（裁定前，未写代码）

日期 2026-11-14 · tmh · **只读勘察**，等军师裁定后再动手

## 1. 硬约束推出的第一个结论

任务写「不碰 `train_d.py` 九文件指纹链」，而 `forward_engine` 对**未注册的 arm 现在直接 raise**（`train_d.py:403`，M4 那次修复的产物）：

```python
raise ValueError('forward_engine has no branch for arm %r: it would silently fall ...')
```

`evaluate`（`:656-682`）又完全经由 `forward_engine`，`train_arm` 经由 `build_d` + `forward_engine`。所以：

> **不碰 `train_d.py` ⇒ 不能复用 `forward_engine` / `evaluate` / `train_arm`，候选必须自带这三件。**

这不是选择，是那条 raise 的直接后果。**它同时也意味着：任何「偷偷把候选塞进现有 arm 名」的做法（比如借用 `B0`）都会被这条 raise 或分支逻辑吃掉**——借名等于让候选的前向不被执行。

## 2. 自建部分与复用部分

**必须自建（约 120 行）**

| 件 | 行数 | 说明 |
|---|---:|---|
| `forward_engine` 等价 | ~15 | `engine_tensors` → `encode` → 候选头 → `pred` |
| `evaluate` 等价 | ~25 | 逐引擎取 5 个截断点，最终**仍调 `metrics_from_arrays`** |
| 训练循环 | ~80 | 按 batch 累加、余弦调度、早停、选轮 |

**可以复用（都不在九文件链里，或只读导入）**

- `paper_b.hc.data_d.engine_batch`（数据接口，**在**九文件链里，但只 import，不改）
- `paper_b.training.metrics.metrics_from_arrays`（**指标口径的唯一实现**，只 import）
- `paper_b.hc.model_hc.DSPT_HC` / `DSPT_HC_Frozen` / `CycleOnly`（`model_hc.py` **不在**九文件链里）
- `paper_b.hc.config_hc`（协议常数，只读）

## 3. 口径漂移是这条路最大的风险

自建训练循环最容易出的事，是**协议参数被抄成字面量**——`epochs=40`、`patience=12`、`lr`、`wd`、`batch`、`eta_min=1e-5`、`clip=1.0`。抄错一个，结果就与母体不同协议，而**报错不会发生**。

两条对策：

1. **全部从 `config_hc` 读**，不在新 runner 里出现数值字面量；
2. **step-0 恒等断言**：候选若在 step 0 是恒等（PrefixCal 的前缀校准在无观测时应退化为恒等），则新 runner 的第一轮 `val_engine_mean_rmse` 必须与**同 seed 母体格逐位相同**；不等就拒绝开跑。

第 2 条是本轮「闸门必须走真入口」的延续：**它比较的是新 runner 与冻结母体的实际输出，不是比较两段代码。**

## 4. 指纹策略

九文件链**一个字节不动**。新 runner 在写 `metrics.json` 时额外记录**自身文件的 sha256**（`prefix_cal.py` + runner），与九文件链并存：

```
code_fingerprint.files_sha256        九文件链（母体，不动）
code_fingerprint.extension_sha256    候选自己的文件（新增）
```

这样母体的可比性不变，候选的可归因性也有——**而不需要把候选文件塞进九文件链**（塞进去会让所有历史格与未来格的指纹不可比）。

## 5. 尚未决定、等裁定的两件事

1. **候选是否存在**：军师若判「无候选」，本文档只作为一次勘察记录，不产生代码。
2. **若存在**：数学差量写在哪个位置（`encode` 之后加一项？还是对 `reg_head` 的输入做校准？）——**这决定 §2 那 15 行前向怎么写**，也决定 step-0 是否真的恒等。**这一步不预先实现。**

## 6. 复现

```
Select-String paper_b/hc/train_d.py -Pattern "no branch for arm"   # :403
Select-String paper_b/hc/train_d.py -Pattern "^def evaluate"       # :656
Select-String paper_b/hc/train_d.py -Pattern "names = \("          # 九文件链
```
