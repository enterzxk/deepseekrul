# Gate A 判定快照 · 2026-11-11T19:40Z（a40 时钟）

> 判定器：`tools/gate_a_judge.py`　**sha256** `58c074a847f06f4af9b6876f8e8f246d258091746c2550afae7c8ecf6b6975d0`（未修改）
> 数据来源：只读每格 `close.json` / `metrics.json` / `preds.csv`，**不读 `registry/runs.csv`**。
> 取格规则：`HC-C-{v}-{s}-s101-W1` + `['', r2..r6]` 中第一个 `status=ok`。
> 上一份快照：`GATE_A_VERDICT_20261111T1927Z.md`

## 本快照的新增内容

**`A3a-FD001-W1` 已落盘**，这是 G2 的第一个数据点，且**符号反向**。

## 输出原文

```
cell         FD001       FD002       FD003       FD004
A0         30.0986     26.2874     39.9834     37.1218
A1         11.2806     11.8031     12.0306     13.2087
A2         10.5925     10.2074     10.6517     11.7896
A3a        11.3167          --          --          --
A4              --          --          --          --
A5              --          --          --          --
A3p        10.8951          --          --          --

G1    mean=+10.46%    [PASS]  +6.10% +13.52% +11.46% +10.74%
      rule: mean d(A1->A2) >= +1.0% and no subset < -2.0%
G2    mean=-6.84%     [PENDING]  -6.84% n/a n/a n/a
      rule: mean d(A2->A3a) >= +1.0% and >=3/4 subsets same sign
G2p   mean=+3.73%     [PENDING]  +3.73% n/a n/a n/a
      rule: report only; no threshold; not in the main table
G3    mean=-175.73%   [PASS]  -166.82% -122.72% -232.35% -181.04%
G5   {'G1': {'positive': 4, 'consistent': True}}

INTEGRITY: all collected cells pass baseline/written_by/preds/Jensen
```

## FD001 上的三臂排序

| 臂 | 编码 | 注入点 | `val_engine_mean_rmse` | 相对 A2 |
|---|---|---|---|---|
| **A2** | 线性年龄 `Linear(1,64)` | post-norm | **10.5925** | — |
| A3p | Fourier `Linear(8,64)` | pre-norm（冻结） | 10.8951 | +2.78% |
| A3a | Fourier `Linear(8,64)` | post-norm | 11.3167 | **+6.84%** |

**简单线性编码在两个注入点上都优于 Fourier 编码。**

## 三条必须一起记的事实

1. **不是噪声**：`|A3a − A2| = 0.724 RMSE`，而投影抽样噪声实测 `0.030`（`HC-C-A2-FD001-s101-NOISE1`）——效应是噪声的 **24 倍**。
2. **跨批次可复现**：`A3a-FD001-W1 = 11.3167056611`，与 2×2 的 `0030Z = 11.316705661052403` **逐位相同**。W1 批次完整复现了 2×2，σ=0 跨批次成立。
3. **G2′（A3a→A3p）= +3.73%，但这是两变量混杂，不得称「注入点效应」**：A3p 是 `DSPT_HC_Frozen`（`frozen_injection=True`，pre-norm、循环模块**冻结**），A3a 是 `DSPT_HC`（`frozen_injection=False`，post-norm、循环模块**可训练**）。两者同时差「冻结 vs 可训练」与「pre vs post-norm」两个变量，所以 +3.73% **只能报数，不能归因**。若要单独测注入点，需要第四个臂：可训练 + pre-norm。（本条为按主控指正修正，原先写作「注入点效应」。）

## 判定与限定

- **G2 目前 −6.84%，符号反向**。G2 是四子集判据（均值 ≥1.0% 且 ≥3/4 同号），**一格不作结论**；`A3a-FD002` 已在跑。
- 若该方向在四子集复现，**论文主张须退到「cycle 先验有用」**（G1 已 PASS），「Fourier 编码是创新点」须放弃——即任务书 §9 预设的分支。
- 判定器未改逻辑，本快照与上一份的唯一差异是 `A3a-FD001` 的加入。
