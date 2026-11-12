# 方法规格（Paper B）

状态：**Stage D 部分草稿，待数学审查**。Stage C 的实测结论见 `CYCLE_DEPENDENCE_STUDY.md`。
本文只写规格，**不含实现**；所有协议常量从 `paper_b/configs/protocol.json` 读取，
本文引用处一律标注键名，不重打字面值。

冻结协议（键名 → 值）：`span`=70、`d_model`=64、`layers`=2、`n_cond`=6、`rul_cap`=125、
`val_cuts`=(0.40, 0.55, 0.70, 0.85, 0.95)、`epochs`=40、`patience`=12、`batch`=512。

---

# Stage D：结构化 HI

## D0. 记号与接口

发动机 $e$ 有 $n_e$ 个周期，周期索引 $t=1..n_e$（$t$ 是**发动机内的绝对年龄**，不是窗口内位置）。

传感器窗口
$$W_t \;=\; X_{\max(1,\,t-s+1)\,:\,t},\qquad s=\texttt{span}=70 ,$$
右对齐、左侧补零并给 mask（与冻结 `make_train_windows` 一致）。

冻结骨干编码器 $E(\cdot)$：输入 $W_t$ 与 mask、baseline、cond，输出池化表示
$$z_t \;=\; E(W_t)\in\mathbb{R}^{d},\quad d=\texttt{d_model}=64 .$$

**$z_t$ 不含绝对 cycle**（任务书 §7 D1 的硬约束）；cycle 只在 Stage E 引入。

真值（训练期可用、推理期不可用）：$r_t=n_e-t$ 为**未封顶** RUL，
$y_t=\min(r_t,\texttt{rul\_cap})$ 为封顶标签。**两者都要落盘**（任务书 §7 D4）。

---

## D1. 结构化累积损伤

### D1.1 定义

增量网络 $f_\theta:\mathbb{R}^{d}\to\mathbb{R}$，非负增量
$$\delta_t \;=\; \mathrm{softplus}\big(f_\theta(z_t)\big)\;\ge\;0 ,$$

发动机级累积
$$\boxed{\;D_0=0,\qquad D_t \;=\; \sum_{u=1}^{t}\delta_u\;}\qquad \mathrm{HI}_t=1-D_t .$$

固定失效阈值 $D^\star=1$，训练发动机满足 $D_{n_e}\to 1$（由 $L_\text{terminal}$ 施加）。

### D1.2 「窗口开头不许重置 D」如何在数学上保证

**不是靠约定，是靠定义域。** 上式里递推下标是**发动机周期** $u$，求和从 $u=1$ 起；
$W_u$ 只是 $z_u$ 的载体。定义中**根本不存在窗口局部状态变量**，因此"重置为 0"
不是一个可以写出来的操作，而是**不可表示**。

实现上的对应要求（写进接口，不写进注释）：

> 模块只暴露 `engine_forward(engine)`，返回该发动机长度上的 $D_{1:n_e}$；
> **不暴露** `window_forward(window)`。任何以窗口为单位返回 $D$ 的 API 都违反本节。

三种失败模式与本节的关系：

| 失败模式 | 后果 | 本节如何排除 |
|---|---|---|
| 每窗重置 $D\leftarrow0$ | 同一 $t$ 不同窗长给出不同损伤尺度 | 递推下标是 $u$，窗长不进入定义 |
| 窗内累积当全寿命损伤 | $D$ 的尺度随窗长漂移，阈值 1 失去意义 | $D_t$ 是**前缀和**，与 $s$ 无关 |
| 训练按窗打乱 | 前缀和断裂，$D$ 与 $t$ 失去对应 | 见 D7：batch 以发动机为单位 |

### D1.3 一致性引理（可检查条件）

**引理 1.** 固定 $s$，令 $W_u$ 为上式定义的**唯一**规范窗口。则对同一发动机的同一时刻 $t$，
无论它出现在哪个 batch、哪次前向里，$D_t$ 相同。

*证明.* $D_t=\sum_{u\le t}\delta_u$ 且 $\delta_u$ 只依赖 $z_u=E(W_u)$；$W_u$ 由 $(u,s)$ 唯一确定，
与 batch 组成无关。∎

**引理 1 的失效条件（必须写成断言）**：

1. **$s$ 不是常量**：若某些路径用变长窗口，$W_u$ 不再唯一，引理失效。
   → 断言 `span` 全局唯一，等于 `protocol['span']`。
2. **窗口不是右对齐**：若窗口取 $X_{t:t+s-1}$ 或居中，$z_u$ 会看到 $u$ 之后的测量，因果性破坏。
   → 断言 mask 只在右侧连续、且 $W_u$ 的最后一个位置是 $u$。
3. **同一次前向里混入不同发动机的补齐**：跨发动机 padding 若未用 per-engine 长度 mask 屏蔽，
   短发动机尾部会读到别的发动机的传感器。
   → 断言 $D_t$ 对 $t>n_e$ 不参与任何损失（mask 掉，不是置零参与）。

### D1.4 与「窗口状态空间模型」备选的关系

任务书允许另一条路：显式初始状态估计 $\hat D_{t-s+1}$ + 一致性惩罚。本文**不采用**，
理由：它把"不重置"变成软约束（惩罚项），而 D1.1 是硬约束（定义使然）。
只有在 $n_e$ 过大、无法整条前向时才退回备选，且必须同时满足
$\big|\hat D^{(s)}_t-\hat D^{(s')}_t\big|\le\varepsilon$ 对 $s\ne s'$ 的显式一致性检验。

---

## D2. 未来退化速度

### D2.1 当前速率

$$\nu_0=\delta_1,\qquad \nu_t=(1-\alpha)\,\nu_{t-1}+\alpha\,\delta_t ,$$
$\alpha\in(0,1]$ 为 EMA 系数（初始版取 $\alpha=0.2$，随 `span` 缩放，**不做网格搜索**）。
$\nu_t$ 的量纲：损伤/周期。

### D2.2 未来速率与预测损伤

$$\hat\nu_{t+k\mid t}=\mathrm{softplus}\big(g_\phi(D_t,\ \nu_t,\ z_t,\ k)\big)\;\ge 0 ,$$
$$\hat D_{t+k\mid t}=D_t+\sum_{j=1}^{k}\hat\nu_{t+j\mid t} .$$

- **B2（固定速率）**：$\hat\nu_{t+k\mid t}\equiv\nu_t$，即 $g_\phi$ 不含 $k$ 且输出 $\nu_t$。
  退化为 $\hat D_{t+k\mid t}=D_t+k\,\nu_t$，可解析求到达时间。
- **B3（可变速率）**：$g_\phi$ 显式含 $k$。$k$ 的嵌入用固定基 $\psi(k)=\big(k,\ \log(1+k)\big)$，
  **不学 $k$ 的 embedding 表**（避免把 $k$ 当类别，外推时无定义）。

$\hat D$ 对 $k$ 单调不减（$\hat\nu\ge0$ 保证），因此到达时间良定义。

---

## D3. 可微失效阈值到达

### D3.1 生存和形式

设观测窗口的预测时域 $K$（见 D3.4）。定义危险率与生存函数
$$h_j=\sigma\!\big(\beta(\hat D_{t+j\mid t}-1)\big),\qquad
S_0=1,\quad S_k=\prod_{j=1}^{k}(1-h_j),$$
$$\boxed{\;\widehat{\mathrm{RUL}}_t=\hat T^\star_t=\sum_{k=0}^{K-1}S_k\;}$$

依据离散非负随机变量的恒等式 $\mathbb{E}[T]=\sum_{k\ge0}\Pr(T>k)$，而 $\Pr(T>k)=S_k$。
$\beta>0$ 是锐度：$\beta\to\infty$ 时 $S_k=\mathbb{1}[k<k^\star]$，
$\hat T^\star\to k^\star$ 恰好等于硬到达步数，**无系统性偏移**（这是 D3.2 检验的对象）。

### D3.2 硬到达与软硬一致性

推理期硬规则：$\;T^\text{hard}_t=\min\{k\ge0:\hat D_{t+k\mid t}\ge1\}$，截断到 $K$。

**预注册检验（D-G4）**：在 val 上要求
$\big|\overline{\hat T^\star}-\overline{T^\text{hard}}\big|\le 1$ 周期。
超出则说明 $\beta$ 过小（软到达被过度平滑），按 $\beta\mapsto2\beta$ 重标定一次，**只一次**。

### D3.3 为什么不用硬 argmin 训练

硬 argmin 的梯度几乎处处为 0，RUL 误差无法反传到 $\delta$ 与 $g_\phi$——这正是任务书
§7 D3 禁止"只用不可导硬 argmin 作唯一训练算子"的原因。D3.1 的 $S_k$ 对 $D$ 处处可导。

### D3.4 预测时域 $K$ 与上限处理

$K$ 取**该子集训练发动机最长寿命减一**（从数据读，不写字面值）。
理由：封顶标签 $y_t\le125$ 只说明"至少 125"，真实 RUL 可达数百；
若取 $K=125$，$\hat T^\star$ 被硬性钉在 125，等于把封顶当真实值——任务书 §7 D4 明确反对。

**上限内未达阈值**：$S_K>0$ 时 $\hat T^\star<K$ 仍良定义（和为有限项），
不引入"未命中"特判；样本的封顶状态只通过 $L_\text{RUL}$ 的删失项处理（D4.1）。

---

## D4. 训练损失

$$L=L_\text{RUL}+\lambda_T L_\text{terminal}+\lambda_S L_\text{smooth}$$

### D4.1 $L_\text{RUL}$：删失（Tobit）平方误差

$$\ell_t=\begin{cases}
\big(\hat T^\star_t-r_t\big)^2, & r_t<\texttt{rul\_cap}\quad(\text{未删失})\\[2pt]
\big(\max(0,\ \texttt{rul\_cap}-\hat T^\star_t)\big)^2, & r_t\ge\texttt{rul\_cap}\quad(\text{右删失})
\end{cases}$$

**为什么不能直接用 $(\hat T^\star_t-y_t)^2$**：$y_t=\min(r_t,125)$ 把"至少 125"写成"恰好 125"，
会惩罚 $\hat T^\star>125$ 的预测，等于让模型去拟合截断本身。任务书 §7 D4 要求明确
"封顶是截断监督目标，不等于真实退化状态"，本节即是它的数学形式。

**指标口径不变**：论文报的 `val_engine_mean_rmse` 仍在封顶标签上算，
计算前把 $\hat T^\star$ **截断到 `rul_cap`**。这一步必须写进评估代码并在结果文档注明。

### D4.2 $L_\text{terminal}$：失效终点锚定

$$L_\text{terminal}=\frac{1}{|\mathcal{E}_\text{tr}|}\sum_{e\in\mathcal{E}_\text{tr}}\big(D^{(e)}_{n_e}-1\big)^2$$

只取每台训练发动机的**最后一个周期**；用训练集，绝不用 val/test。

### D4.3 $L_\text{smooth}$：速率平滑

$$L_\text{smooth}=\frac{1}{|\mathcal{T}|}\sum_{t}\big(\delta_{t+1}-\delta_t\big)^2
=\frac{1}{|\mathcal{T}|}\sum_t\big(D_{t+1}-2D_t+D_{t-1}\big)^2$$

即 $D$ 的二阶差分能量，抑制速率的逐周期抖动。

### D4.4 量纲检查（$\lambda$ 不是自由旋钮）

| 量 | 量纲 |
|---|---|
| $\delta_t,\ D_t$ | 无量纲 |
| $\nu_t,\ \hat\nu$ | 无量纲 / 周期 |
| $\hat T^\star,\ r_t$ | 周期 |
| $L_\text{RUL}$ | 周期² |
| $L_\text{terminal}$ | 无量纲 |
| $L_\text{smooth}$ | 无量纲 / 周期² |

要使三项可加，$\lambda_T$ 必须带 **周期²** 的量纲，$\lambda_S$ 必须带 **周期⁴**。取

$$\lambda_T=c_T\,L_{\text{scale}}^2,\qquad \lambda_S=c_S\,L_{\text{scale}}^4,\qquad
L_\text{scale}=\texttt{rul\_cap}=125 ,$$

$c_T,c_S$ 为无量纲 $O(1)$ 常数（初始版 $c_T=c_S=1$）。
**禁止把 $\lambda$ 当超参扫描**：它们已被量纲锁死，扫 $\lambda$ 等于扫单位，没有物理意义。

### D4.5 梯度贡献检查（必须实测并落盘）

每个 epoch 记录
$$g_i=\Big\|\nabla_\theta L_i\Big\|_2,\qquad i\in\{\text{RUL},\text{terminal},\text{smooth}\},$$
并要求 $g_i/\max_j g_j\in[0.1,\,1]$ 持续成立。任一辅助项低于 0.1 即视为"辅助任务失效"
（任务书 §7 D4 的原话），按 $c_T$ 或 $c_S$ 上调一次后重测，**不靠猜**。
$g_i$ 写入 run 的 `metrics.json` 与训练日志。

---

## D5. 四组核心对照（B0–B3）

| 编号 | RUL 来源 | HI | 未来速率 | 说明 |
|---|---|---|---|---|
| **B0** | 直接回归头 | 无 | — | 冻结 DSPT 的直接 RUL 回归，Stage D 的基线 |
| **B1** | 直接回归头 | 有，**仅辅助** | — | 加 $L_\text{terminal}$，RUL 仍走回归头 |
| **B2** | **仅** $\hat T^\star$（D3） | 有 | 固定 $\nu_t$ | 无旁路 RUL 头 |
| **B3** | **仅** $\hat T^\star$（D3） | 有 | 可变 $g_\phi(\cdot,k)$ | 无旁路 RUL 头 |

**硬约束（任务书 §7 D5）**：B2/B3 **不得**存在绕过 HI 的直接 RUL 预测头。
实现上断言：B2/B3 的参数表里不存在以 `rul_head` 命名且被调用的模块。

**可比性**：B0–B3 同编码器、同划分（`split_seed`/`fold`/`val_frac`/`val_cuts` 全冻结）、
同 `epochs`/`patience`/`batch`/`lr`/`weight_decay`；报告参数量与单 epoch 墙钟。

---

## D6. 预注册判据（写死，事后不改）

| 编号 | 判据 | 阈值 | 不过的后果 |
|---|---|---|---|
| **D-G1** | 结构化 HI 不劣于直接回归：B2 或 B3 的 `val_engine_mean_rmse` ≤ B0 | 逐子集 ≤ B0 | 结构化 HI 在此 benchmark 上无增益，按负结果报告，不得调参救 |
| **D-G2** | 可变速率有用：B3 ≤ B2 | 均值，且 ≥3/4 子集同号 | 退化为 B2 的固定速率叙事 |
| **D-G3** | 终点锚定：训练发动机 $\overline{\lvert D_{n_e}-1\rvert}$ | ≤ 0.10 | 阈值 1 不可解释，D2/D3 全部失效 |
| **D-G4** | 软硬到达一致：$\lvert\overline{\hat T^\star}-\overline{T^\text{hard}}\rvert$ | ≤ 1 周期 | $\beta$ 重标定一次，仍不过则软到达不可用 |
| **D-G5** | 单调与不塌缩：$D$ 逐周期不减（构造保证，需断言）+ 训练期 $\overline{\delta_t}$ | $>0$ 且不趋 0 | $\delta$ 塌缩说明增量网络没被用，B2/B3 结论作废 |
| **D-G6** | 梯度平衡：$g_i/\max_j g_j$ | ∈ [0.1, 1] | 辅助项失效，按 D4.5 重配权重一次 |

**证伪条件（提前写明，避免事后找理由）**：
若 B2/B3 在四个子集上都不优于 B0，则"用可微阈值到达替代直接回归"这一主张在本 benchmark 上
被证伪，论文只能把它写成负结果或机制分析，**不得**靠加辅助损失、换骨干、调 $\lambda$ 翻盘。

---

## D7. 实现约束（写进代码断言，不写进注释）

1. **禁止窗口级打乱**：batch 以发动机为单位；一次前向内对整条 $1..n_e$ 递推。
   窗口只是 $z_u$ 的载体，不是样本。
2. **$s$ 全局唯一**且等于 `protocol['span']`；断言 `span` 不被任何路径覆盖。
3. **mask 右对齐**：$W_u$ 的最后一位是 $u$；断言不存在左侧 mask 或居中窗口。
4. **$D$ 单调断言**：`assert (diff(D) >= -1e-6).all()`；失败即视为实现 bug，不是数值噪声。
5. **补齐不参与损失**：$t>n_e$ 的位置 mask 掉，不得置零后参与前缀和或损失。
6. **两套标签都存**：`r_t`（未封顶）与 `y_t`（封顶）同时落盘，评估时显式声明用的是哪一套。
7. **评估截断**：$\hat T^\star$ 在算 `val_engine_mean_rmse` 前截断到 `rul_cap`，并在结果文档注明。

---

## D8. 未决项

- **cycle 在 D 中的形态**：Stage C 的结论是 raw 标量优于 Fourier（−5.03%，四子集全负），
  但 G4 判据替换仍 **PENDING-L2**。因此 D 的 B0–B3 **先不含 cycle**（符合任务书 C→D 顺序），
  cycle 融合留到 Stage E 的 C0–C3。
- **$\alpha$ 与 $\beta$ 的取值**：初始版固定（$\alpha=0.2$、$\beta=1$），不做搜索；
  若 D-G4 不过再按 D3.2 重标定一次。
- **$n_e$ 较大时的显存**：整条前向的序列长度可达数百，若显存不够，退回 D1.4 的备选，
  但必须同时提交一致性检验结果。
