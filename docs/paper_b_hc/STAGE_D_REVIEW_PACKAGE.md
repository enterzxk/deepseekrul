# Stage D 审查包：接口 + 数学

状态：**只出包，不放行训练**。截至本文件，Stage D 没有任何训练 run；下面所有"已证"都指
代码级断言或前向推理，不是实验结果。
规格正文：`METHOD_SPECIFICATION.md`（D1–D8）。代码：`paper_b/hc/data_d.py`、
`paper_b/hc/test_d_engine.py`、`docs/paper_b_hc/probes/*`。

---

## 1. 三条验收口径（军师）

| 口径 | 含义 | 现状 | 怎么验 |
|---|---|---|---|
| **正确性** | 损伤递推、窗口一致性、阈值到达、损失量纲按规格成立 | **代码级已证**（§4） | 断言 + 契约检查，已全绿 |
| **非坍缩** | $\delta_t$ 不塌缩到 0、$D$ 不用一个常数糊过去 | **未测**（要训练） | D-G5：训练期 $\overline{\delta_t}>0$ 且不趋 0；$D_{n_e}$ 分布跨发动机有差异 |
| **公平比较** | B0–B3 同骨干、同划分、同预算；B2/B3 无旁路 RUL 头 | **未跑** | 字段级 config diff + 参数表断言 |

**必须先说清**：本包能证明的是"照这个规格写出来的代码，其数学性质成立"；
它**不能**证明"这个目标函数能训出有用的模型"。后者只有训练能回答，不在本包范围。

---

## 2. 接口（D7.1）

### 2.1 为什么是新模块而不是改 `data_hc`

`paper_b/hc/data_hc.py` 是 v9/v11 code lock 的 16 个 blob 之一（md5 `3ad7c111…`）。
改它会让所有记录 `code_md5` 的 Stage C run 失效。所以接口放在新模块
`paper_b/hc/data_d.py`，只 import `data_hc`、不改它。

### 2.2 字段与不变量

```
engine_batch(subset, data_dir, span, cuts, split='train'|'val') -> {
  x (E,Lmax,span,C)  mask (E,Lmax,span)  baseline (E,Lmax,C)  cond (E,Lmax)
  valid (E,Lmax)     cycle_raw (E,Lmax)  r (E,Lmax)           y (E,Lmax)
  engine_id (E,)     lengths (E,)        Lmax  split
  cut_t (E,K)  cut_valid (E,K)           # 仅 'val'
}
```

**`split='val'` 返回整条轨迹，不是那 5 个 cut 窗口。** 原因：一个 cut 窗口在 cycle $c$
处无法提供 $D_c$——递推需要 1..c 的每一个周期，而 cut 窗口只有 5 个采样点。所以 val 侧
用 `make_train_windows` 重建全轨迹，再用 `cut_t` 定位五个协议 cut。

**`r = n_e − t` 在 val 上也是精确已知的**：val 发动机就是训练发动机按比例截断出来的，
失效时刻已知。这是 D4.6 的 `rmse_uncapped` 能存在的前提。

### 2.3 契约检查（实跑）

`probes/d_interface_contract.py`，每个断言都抄自 D7.1：

```
FD001  train engines= 80 Lmax=362 windows= 16762
FD001  val   engines= 20 Lmax=283 windows=  3869 val_windows=100
FD004  train engines=199 Lmax=543 windows= 48691
FD004  val   engines= 50 Lmax=435 windows= 12558 val_windows=250
CONTRACT OK
```

检查项：字段与形状、`valid` 真实长度内为 1/补齐区为 0、**补齐区 `x`/`mask`/`baseline`
逐元素为 0**（非零补齐会混进前缀和）、`cycle_raw` 在位置 $t$ 等于 $t+1$、
`r = n−t`、`y = min(r, 125)`、`cut_t` 与冻结 val 窗口的 cut 一一对上。

---

## 3. 数学（D1–D4）

### 3.1 D1：损伤递推，与"窗口开头不许重置"如何被保证

$$\delta_t=\mathrm{softplus}\big(f_\theta(z_t)\big)\ge0,\qquad D_0=0,\quad D_t=\sum_{u=1}^{t}\delta_u$$

**不是靠约定，是靠定义域**：递推下标是**发动机周期** $u$，求和从 1 起；窗口只是 $z_u$ 的载体。
定义里**不存在窗口局部状态变量**，所以"重置为 0"不是一个可以写出来的操作。
实现上只暴露 `engine_forward(engine)`，不暴露 `window_forward(window)`。

**引理 1（一致性）**：固定 $s$，$W_u$ 唯一 ⇒ 同一发动机同一 $t$ 的 $D_t$ 与 batch 组成无关。
**失效条件**（都写成了断言）：$s$ 非常量、窗口非右对齐、跨发动机补齐未 mask。

### 3.2 D2：未来速率

$$\nu_t=(1-\alpha)\nu_{t-1}+\alpha\delta_t,\qquad
\hat D_{t+k\mid t}=D_t+\sum_{j=1}^{k}\mathrm{softplus}\big(g_\phi(D_t,\nu_t,z_t,k)\big)$$

$\hat\nu\ge0$ ⇒ $\hat D$ 对 $k$ 单调 ⇒ 到达时间良定义。B2 取 $\hat\nu\equiv\nu_t$（可解析），
B3 显式含 $k$（用固定基 $(k,\log(1+k))$，不学 embedding 表，避免外推无定义）。

### 3.3 D3：可微阈值到达

$$h_j=\sigma\big(\beta(\hat D_{t+j\mid t}-1)\big),\quad S_k=\prod_{j\le k}(1-h_j),\quad
\boxed{\hat T^\star=\sum_{k=0}^{K-1}S_k}$$

依据离散非负随机变量恒等式 $\mathbb E[T]=\sum_{k\ge0}\Pr(T>k)$。$\beta\to\infty$ 时
$\hat T^\star\to$ 硬到达步数，**无系统偏移**（这是 D-G4 要检验的）。

**$K$ 只需要大于 `rul_cap`**：每个 $r_t\ge125$ 的样本都按 D4.1 删失在 125，任何
$\hat T^\star\ge125$ 都给出零损失；把 $K$ 提到最长寿命 543 对**任何样本的损失都没有影响**，
只白付 2.7 倍算力。取 $K=200$。代价已量化：真 RUL>200 的窗口占训练 10.3–24.6%、val 0–3.2%，
这些窗口上的 $\hat T^\star$ 只是**下界**，不得当校准过的 RUL 报出去。

**复杂度**：$k$ 外层、窗口走批维 → $\mathcal O(KNd h)$ FLOPs、$\mathcal O(KN)$ 内存。
FD004 最大：$K=200$、$N=48691$ → 前向约 $2.0\times10^{10}$ FLOPs、状态张量 39 MB。
**若按窗口逐个滚**，一轮就是 $9.7\times10^6$ 次小网络调用，Python 开销直接爆——这条是硬约束。

### 3.4 D4：损失、量纲、删失

$$L=L_\text{RUL}+\lambda_T L_\text{terminal}+\lambda_S\tilde L_\text{smooth}$$

- $L_\text{RUL}$：**删失 hinge**，不是 Tobit 似然。$r<125$ 用平方误差，$r\ge125$ 用
  $(\max(0,125-\hat T^\star))^2$。代价：对 $\hat T^\star>125$ 零梯度（D4.6 量化）。
- $L_\text{terminal}$：$\overline{(D_{n_e}-1)^2}$，只用训练发动机的最后一个周期。
- $\tilde L_\text{smooth}=L_\text{smooth}/S_0$，$S_0=\max(10^{-12},\mathrm{stopgrad}\,\widehat{\mathbb E}[(\Delta\delta)^2])$。
  **不归一化这一项等于关掉**：实测 $(\Delta\delta)^2\sim10^{-8}$，乘 $\lambda_S=125^2$ 后仍只有 $10^{-4}$。
  $S_0$ 只测一次、B0–B3 共用、detach、带下限。
- **量纲**：$D_t=\sum\delta_u$ 是求和不是积分 ⇒ $\dim\delta=\dim D=1$；$L_\text{RUL}$ 是周期²，
  另两项无量纲 ⇒ $\lambda_T=\lambda_S=c\,L_\text{scale}^2$，**没有四次**。
  若改写成连续导数 $\dot D=\delta/\Delta t$ 才会出现周期⁴，两种写法差常数 $125^2$——
  所以配置里必须写 $\lambda$ 的**数值**，不能只写 $c_S=1$。

**封顶偏差已量化**：训练窗封顶 40.3/39.9/51.2/48.9%，val 封顶 7.0/8.5/11.0/20.4%。
删失样本只要 $\hat T^\star\ge125$ 就对 RMSE 零贡献，所以 `val_engine_mean_rmse`
实际是在未封顶子集上算的。因此必须并报 `frac_capped`、`boundary_hit`、`rmse_uncensored`、
`rmse_uncapped`（限 $r\le K$）、`rmse_by_censoring`、`exact_boundary` 六项。

---

## 4. 正确性证据（全部实跑）

`paper_b/hc/test_d_engine.py`，a40 上 6/6 通过：

| 断言 | 内容 | 结果 |
|---|---|---|
| `test_overlap_consistency` | 同发动机两窗重叠周期 $D_t$ 相同 | PASS |
| `test_increment_closure` | $D_s(w_2)-D_{s-1}(w_1)=\delta_s$ | PASS |
| `test_monotonicity` | $\Delta D\ge0$、$\delta\ge0$、$D_0=0$ | PASS |
| `test_reset_is_detected_by_overlap` | **负控**：被禁的按窗重置必须失败 | PASS（误差 >1.0） |
| `test_reset_drifts_with_window_start` | **负控**：漂移量须等于被丢弃的前缀 | PASS（1e-9） |
| `test_two_interface_parity` | 冻结 cut 窗口 vs 轨迹切片的 $h$ 逐位一致 | PASS（FD001 全部 100 窗） |

最后一条是关键：它证明 `make_val_windows` 与 `make_train_windows` 在同一 (engine, cycle)
上产出的窗口**逐位一致**。不过这条的话，D 的每个 val 数字都是拿错输入算的。

---

## 5. 非坍缩：判据已定，**尚未测量**

D-G5：$D$ 逐周期不减（构造保证，已断言）+ 训练期 $\overline{\delta_t}>0$ 且不趋 0。
失败即"增量网络没被用"，B2/B3 结论作废。

**现在不能声称它成立**：没有训练，$\overline{\delta_t}$ 没测过。
软肋在 $L_\text{terminal}$：它只锚最后一个周期，若 $\delta$ 整体压到极小，$D_{n_e}\to1$ 仍可
通过把 $K$ 拉长满足，而 $D_t$ 在中期几乎为 0——这会同时满足锚定和单调，却是坍缩解。
所以 D-G5 必须报 $\delta$ 的**分布**（中位数、p05、跨发动机离散度），不能只报均值。

---

## 6. 公平比较：设计已定，**尚未运行**

| 编号 | RUL 来源 | HI | 未来速率 |
|---|---|---|---|
| B0 | 直接回归头 | 无 | — |
| B1 | 直接回归头 | 有，仅辅助 | — |
| B2 | **仅** $\hat T^\star$ | 有 | 固定 $\nu_t$ |
| B3 | **仅** $\hat T^\star$ | 有 | 可变 $g_\phi(\cdot,k)$ |

硬约束：B2/B3 不得有绕过 HI 的直接 RUL 头（断言参数表里没有 `rul_head`）。
B0–B3 同编码器、同划分、同 `epochs/patience/batch/lr/wd`；报参数量与单 epoch 墙钟。

**一个已知的公平性隐患**：B0 用直接回归头、B2/B3 用 $\hat T^\star$，两者的**输出参数化不同**，
所以"B2 不如 B0"既可能是结构化 HI 的瓶颈，也可能是阈值到达这个参数化本身更难优化。
要区分，需要 B1（有 HI 但仍走直接头）作桥。B1 的结论不能省。

---

## 7. 已知缺口与风险

1. **全部未训练**：D-G1..D-G6 无一测量。
2. **坍缩解**：见 §5，$L_\text{terminal}$ 对中期 $D_t$ 无约束。
3. **输出参数化混杂**：见 §6，B1 是必需的桥。
4. **$K=200$ 截断**：真 RUL>200 的窗口 $\hat T^\star$ 是下界（训练 10.3–24.6%，val 0–3.2%）。
5. **bf16 测量精度**：`predict_hc` 在 autocast bf16 下取 `.float()`，预测是 bf16 量化值
   （8 位有效位）。判"代码改动是否改数值"应用 ≥1e-3 阈值，低于 bf16 粒度的 Δ 记 0。
   逐位校验必须在 float32 上做，用更高精度比会系统性产生假阳性。
6. **$\alpha,\beta$ 未定标**：初版固定 $\alpha=0.2,\beta=1$，不搜索；D-G4 不过才按 D3.2 重标定一次。
7. **整条前向的显存**：序列长度可达 543，若不够则退回 D1.4 的窗口状态空间备选，
   但必须同时提交一致性检验。
8. **`f_\theta` 与 `g_\phi` 的容量未定**：规格只给了输入输出与量纲，没有定隐层宽度。

---

## 8. 给军师的审查问题

1. **§3.1 的"不可表示"论证**是否足以保证窗口一致性？我把它落到"只暴露
   `engine_forward(engine)`"，但 API 约束本身不是数学证明——这个论证的强度够不够？
2. **§5 的坍缩解**：除 $L_\text{terminal}$ 外，是否需要一条对中期 $D_t$ 的弱约束
   （例如对 $t/n_e$ 的秩相关）？还是说加了就变成不可证伪的事后约束？
3. **§3.3 的生存和**在 $\beta$ 有限时对 $\hat T^\star$ 的偏差方向能否先验确定？
   如果不能，D-G4 的"≤1 周期"阈值是否应改成"相对偏差 ≤2%"？
4. **§6 的 B1 桥**是否必要？如果 B2 已经不如 B0，B1 的结论会不会反而是噪声。
5. **§3.4 的删失 hinge** 与真删失正态似然：在这个样本量（训练窗 1.7万–4.9万、
   封顶占 40–51%）下，hinge 的零梯度区会不会让 $\hat T^\star$ 在封顶层系统性地漂向边界？
   是否该直接用似然而不是 hinge？
