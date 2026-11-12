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

**引理 1 的适用边界（军师 Q1，必须一起写）**：该引理只在下列三条**同时**成立时有效——

1. **固定参数**：$\theta$ 在比较的两次前向之间不变（权重更新后状态当然会变，这不是矛盾）；
2. **规范因果窗口**：$W_u$ 只含 $u$ 及之前的测量，且 $s$ 全局唯一（D7 的断言 2、3）；
3. **确定性推理**：模型处于 `eval()`、无 dropout、无随机增强。

**接口命名不构成证明。** 把 `window_forward` 从 API 里删掉只是让错误难以发生，
不是让引理成立；成立靠的是上面三条。因此必须有实际网络上的检查，而不只是参考递推：
前缀扰动（改 $X_{1:t}$ 不应改变 $D_{t+1}$）、未来扰动（改 $X_{t+1:}$ 不应改变 $D_t$）、
padding 扰动、批内串扰（同批里放别的发动机不应改变本发动机的 $D$）。

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

### D3.1 直接软到达求和（v2；危险率连乘已证伪）

**v1 算子已证伪，仅作记录。** v1 用 $h_j=\sigma(\beta(\hat D_{t+j}-1))$、
$S_0=1$、$S_k=\prod_{j=1}^{k}(1-h_j)$、$\hat T^\star=\sum_k S_k$。当 $\beta=1$ 且 $\hat D\ge0$ 时
$h_j\ge\sigma(-1)$，于是

$$\hat T^\star\le\frac{1-q^K}{h}<\frac{1}{h}=1+e=3.718,\qquad h=\sigma(-\beta),\ q=1-h,$$

**与轨迹无关**，够不到 `rul_cap=125`。这是解析上界，不是调参能修的：$\beta$ 在本规格里
**写死、不搜索**，所以上界贯穿整个训练。该上界在 $D\equiv0$ 处**取到**，
即算子最大值落在坍缩解上。`test_arrival_old_operator_is_capped` 把它钉成回归见证。

**v2 也已证伪（军师反例）。** v2 把离阈值的距离除以**局部**速率。减速反例：
$D_0=0.90$，当前速率 $0.5$，但下一步增量 $0.01$（$D_1=0.91$）——此时
$S_0=\sigma(4\times0.2)=0.690$、$S_1=\sigma(4\times9)\approx1.000$，**生存函数回升**。
逐步除以局部速率保证不了单调性；统一减 0.5 也补不掉"已失效但未来增量极小"时的累积。
算子形式不坏在"软"，坏在**用局部速率做温度**。

**v3：分段线性首达（精确解，不是代理）。** 统一为 **$K$ 个区间、$K+1$ 个节点**：

$$D[0]=\hat D_t,\qquad D[k+1]=D[k]+\hat\nu_{t+k+1\mid t},\qquad k=0..K-1 .$$

每段贡献"阈值以下的时间比例"：

$$\text{frac}_k=\begin{cases}
\mathrm{clip}\Big(\dfrac{1-D[k]}{D[k+1]-D[k]},\,0,\,1\Big), & D[k+1]>D[k]\\[6pt]
\mathbb{1}[D[k]<1], & D[k+1]=D[k]\ \text{(零增量是上式的极限，不是额外约定)}
\end{cases}$$

$$\boxed{\;\hat T^\star_t=\sum_{k=0}^{K-1}\text{frac}_k\;}$$

单调轨迹上**只有一个区间横穿阈值**，故 $\hat T^\star=j+(1-D[j])/\hat\nu_{j+1}$，
**正好是首达点**——这是恒等式，不是近似（牢莫在 12493 条轨迹上验到误差 0）。

**语义逐条钉死**：

| 情形 | 各段 $\text{frac}$ | $\hat T^\star$ |
|---|---|---|
| 已失效 $D[0]\ge1$ | 全部 0 | **0**（RUL=0）|
| 恰好命中 $D[j]=1$ | 前 $j$ 段各 1，第 $j$ 段 0 | $j$（整数）|
| 非整数命中 | 横穿段 $\in(0,1)$ | 插值，不跳格 |
| 未命中 | 全部 1 | $K$，按 $K$ **右删失** |
| 平台（零增量且 $D<1$） | 1 | 全额计入 |

**梯度结构（军师更正过的一条）**：设横穿区间为 $j$，则

$$\frac{\partial\hat T^\star}{\partial\nu_i}=-\frac{1}{\hat\nu_{j+1}}\ (i\le j),
\qquad \frac{\partial\hat T^\star}{\partial\nu_i}=0\ (i>j+1).$$

**穿越之前的每个增量都有梯度**——它们通过累计损伤移动穿越位置；只有穿越之后的没有。
早先"扰动早期速率 T 不变"的写法是错的。

**精确节点处的自动微分约定单列**：$\mathrm{clip}$ 在边界点上导数取 1，
所以恰好落在节点上的命中**仍有**梯度（实测 $\partial T/\partial D_0=-60$，$K=120,n^\star=60$）。
这条不能写成"边界处零梯度"。

**零梯度陷阱**：若 $\hat\nu\cdot K\le1$，轨迹在 $K$ 步内到不了阈值，
$\hat T^\star\equiv K$ 且**梯度恒为 0**，模型卡在删失地平线上回不来。
$\mathrm{softplus}(0)=0.693$ 起步时 $\hat\nu K\approx138\gg1$，安全；
但**$\nu$ 初始化必须保证 $\hat\nu K>1$**，`test_arrival_zero_gradient_trap` 守这条。

**单调性是前提**：clip 求和等于首达只在 $D$ 单调时成立；$D$ 由 $\delta=\mathrm{softplus}\ge0$
累加保证，测试断言而非默认。

**可表示范围用构造性证据**，不靠随机初始化恰好输出大值：预设轨迹在
$n^\star=125/150/300$ 处穿越时，$\hat T^\star$ 必须分别给出对应值
（`test_arrival_representable_range_exceeds_rul_cap`）。这是最便宜的门：3.718 那种上界
应当在一秒内暴露，比任何接口检查都先。

### D3.2 硬到达与软硬一致性

推理期硬规则：$\;T^\text{hard}_t=\min\{k\ge0:\hat D_{t+k\mid t}\ge1\}$，截断到 $K$。

**预注册检验（D-G4）**：在 val 上要求
$\big|\overline{\hat T^\star}-\overline{T^\text{hard}}\big|\le 1$ 周期，**按速率变化幅度分层**。
超出则说明 $\beta_s$ 过小（软到达被过度平滑），按 $\beta_s\mapsto2\beta_s$ 重标定一次，**只一次**。
注意分层不是可选项：$S_k$ 用局部速率，速率曲率大的样本天然偏差更大（D3.1 已量化）。

### D3.3 为什么不用硬 argmin 训练

硬 argmin 的梯度几乎处处为 0，RUL 误差无法反传到 $\delta$ 与 $g_\phi$——这正是任务书
§7 D3 禁止"只用不可导硬 argmin 作唯一训练算子"的原因。D3.1 的 $S_k$ 对 $D$ 处处可导。

### D3.4 预测时域 $K$、复杂度与截断

**实测（`probes/d_label_stats.py`，a40 落盘 `experiments/hc/tmp/d_label_stats.json`）**：

| 子集 | 训练发动机最长 | 长度 p95 | 训练窗数 | val 窗数 |
|---|---|---|---|---|
| FD001 | 362 | 294 | 16762 | 100 |
| FD002 | 378 | 302 | 43251 | 260 |
| FD003 | 525 | 460 | 20486 | 100 |
| FD004 | **543** | 378 | 48691 | 250 |

（543 才是最长寿命；此前口头说的 435 是 val 里的最大 cycle，不是发动机长度。）

**$K$ 的取值**：$K$ 决定**可表示的地平线**与删失点，它**不是**一个可以随便取的值。
取 $K=200$：大于 `rul_cap=125`，使封顶区内的 $\hat T^\star$ 可自由大于 125
（这正是不能取 $K=125$ 的原因）；同时远小于最长寿命 543，省下 2.7 倍算力。
（**撤回**：早先"$K$ 对任何样本的损失都没有影响"是错的——$K$ 改变未命中样本的
$\hat T^\star=K$，而它同时进入 `rmse_uncapped`、`boundary_hit` 与删失阈值，
所以 $K$ 影响指标与损失，只是影响方式已由 D4.6 的六项指标显式量化。）

**复杂度**：把 $k$ 放外层、窗口放批维，每 epoch
$$\mathcal{O}\big(K\cdot N\cdot(d\,h+h)\big)\ \text{FLOPs},\qquad \mathcal{O}(K\cdot N)\ \text{内存}.$$
FD004 最大：$K=200$、$N=48691$、$d=64$、$h=32$ → 前向约 $2.0\times10^{10}$ FLOPs、
状态张量 $200\times48691\times4\text{B}\approx39$ MB。40 epoch 在单卡上是秒级到分钟级。

**硬性实现要求（否则不可行）**：$k$ **必须是外层循环**，窗口是批量维。
若按窗口逐个滚，FD004 一轮就是 $48691\times200\approx9.7\times10^{6}$ 次小网络调用
（$K=543$ 时 $2.6\times10^{7}$），Python 层开销直接爆炸。这条写进 D7。

**尾部截断的代价（必须报，不许隐去）**：实测真 RUL $>200$ 的窗口占比——

| 子集 | 训练窗 | val 窗 |
|---|---|---|
| FD001 | 10.3% | 0.0% |
| FD002 | 10.6% | 0.0% |
| FD003 | 24.6% | 0.0% |
| FD004 | 21.6% | 3.2% |

这些窗口全部 $r_t>200>125$，已按 D4.1 删失，只要 $\hat T^\star\ge125$ 损失就是 0，
所以它们主要影响指标而非梯度。但 $\hat T^\star$ 在 $K$ 处右截断，这些窗口上的
$\hat T^\star$ **是右删失值，不是真实 RUL 的下界，也不是它的上界**——预测值本身
没有单调保证，只能说"未在 $K$ 内观察到穿越"。因此它们不得当成校准过的 RUL 报出去；
`rmse_uncapped` 只在 $r_t\le K$ 上算（D4.6），被截断的尾部单列 `frac_r_gt_K`。
（**撤回**：早先"$\hat T^\star$ 只是真实 RUL 的下界"是错的。）

**上限内未达阈值**：$S_K>0$ 时 $\hat T^\star<K$ 仍良定义（有限项和），
不引入"未命中"特判；封顶状态只通过 $L_\text{RUL}$ 的删失项处理（D4.1）。

---

## D4. 训练损失

$$L=L_\text{RUL}+\lambda_T L_\text{terminal}+\lambda_S L_\text{smooth}$$

### D4.1 $L_\text{RUL}$：删失 hinge（censored hinge）

$$\ell_t=\begin{cases}
\big(\hat T^\star_t-r_t\big)^2, & r_t<\texttt{rul\_cap}\quad(\text{未删失})\\[2pt]
\big(\max(0,\ \texttt{rul\_cap}-\hat T^\star_t)\big)^2, & r_t\ge\texttt{rul\_cap}\quad(\text{右删失})
\end{cases}$$

**为什么不能直接用 $(\hat T^\star_t-y_t)^2$**：$y_t=\min(r_t,125)$ 把"至少 125"写成"恰好 125"，
会惩罚 $\hat T^\star>125$ 的预测，等于让模型去拟合截断本身。任务书 §7 D4 要求明确
"封顶是截断监督目标，不等于真实退化状态"，本节即是它的数学形式。

> **名字要说准：这是 hinge，不是 Tobit 似然。** 单边 hinge 只惩罚"低于 125"，
> 对 $\hat T^\star>125$ **零惩罚**；真删失正态似然会给 $r\ge125$ 的样本
> $-\log\int_{125}^{\infty}\mathcal N(u;\hat T^\star,\sigma^2)\,du$，它对
> $\hat T^\star\gg125$ 仍有（很弱的）梯度，且顺带估出 $\sigma$。
> 本规格**不采用**似然形式，理由有二：① 冻结协议的指标是封顶 RMSE（平方误差），
> hinge 与它的口径一致、可比；② 似然要引入 $\sigma$ 与截断正态 CDF 的数值稳定性问题，
> 初始版不增加自由度。
> **代价必须写进结果**：hinge 对 $\hat T^\star>125$ 无梯度，因此封顶样本的预测值
> 不受约束（D4.6 量化其影响）。若将来要报校准过的 RUL 而非排序，应换回似然形式。

**指标口径不变**：论文报的 `val_engine_mean_rmse` 仍在封顶标签上算，
计算前把 $\hat T^\star$ **截断到 `rul_cap`**。这一步必须写进评估代码并在结果文档注明。

### D4.2 $L_\text{terminal}$：失效终点锚定

$$L_\text{terminal}=\frac{1}{|\mathcal{E}_\text{tr}|}\sum_{e\in\mathcal{E}_\text{tr}}\big(D^{(e)}_{n_e}-1\big)^2$$

只取每台训练发动机的**最后一个周期**；用训练集，绝不用 val/test。

### D4.3 $L_\text{smooth}$：速率平滑

$$L_\text{smooth}=\frac{1}{|\mathcal{T}|}\sum_{t}\big(\delta_{t+1}-\delta_t\big)^2
=\frac{1}{|\mathcal{T}|}\sum_t\big(D_{t+1}-2D_t+D_{t-1}\big)^2$$

即 $D$ 的二阶差分能量（逐周期增量的变化率），抑制速率抖动。

**必须按数据尺度归一化，否则这一项等于关掉。** 实测 $\delta$ 的逐周期变化极小，
$(\Delta\delta)^2$ 在 $10^{-8}$ 量级；乘 $\lambda_S=125^2$ 后仍只有 $10^{-4}$，
相对 $L_\text{RUL}\sim10^{2}$ 完全不产生梯度——D4.5 的梯度比会直接掉到 0，被误判成"辅助任务失效"。

因此定义**归一化平滑项**：

$$\tilde L_\text{smooth}=\frac{L_\text{smooth}}{S_0},\qquad
S_0=\frac{1}{|B_0|}\sum_{b\in B_0}\frac{1}{|\mathcal T_b|}\sum_t\big(\delta_{t+1}-\delta_t\big)^2 ,$$

$B_0$ 是**初始化后、训练开始前**固定抽取的若干 batch（初版 8 个、固定种子）。
$S_0$ **只测一次、写进 `config.json`、训练全程复用**：
逐 epoch 重测等于让目标函数随训练漂移，禁止。

**$S_0$ 的三条硬约束**：

1. **detach**：在初始化权重上算，`no_grad`，不带梯度、不参与反传。
2. **B0–B3 共用同一个 $S_0$**：从**同一份参考 init**（四组共用的那份骨干初始化）算一次，
   写进共享 config，四个 run 读同一个数值。
   否则每个 run 的 $S_0$ 不同，$\lambda_S/S_0$ 的**实际权重**就不同，四组对照的公平性直接没了。
3. **下限** $S_0\leftarrow\max(S_0,\,10^{-12})$：初始化时 $\Delta\delta$ 可能极小，
   不加下限会让 $\lambda_S/S_0$ 把平滑项梯度放大到失控。

写成一行：

$$S_0=\max\!\Big(10^{-12},\ \ \mathrm{stopgrad}\ \widehat{\mathbb E}_{B_0}\big[(\Delta\delta)^2\big]\Big),\qquad
\tilde L_\text{smooth}=L_\text{smooth}/S_0 .$$

### D4.4 量纲检查：从表达式推，不先写结论

时间在本规格里是**离散的、以周期为单位**，因此递推是求和而非积分：

$$D_t=\sum_{u\le t}\delta_u\;\Longrightarrow\;\dim\delta=\dim D=1\ (\text{无量纲}).$$

$\nu_t$ 是 $\delta$ 的 EMA，仍是**逐步增量**，故同样无量纲。于是逐项推：

| 量 | 量纲 | 依据 |
|---|---|---|
| $\delta_t,\ \nu_t,\ D_t$ | 无量纲 | 求和递推，$D_t=\sum\delta_u$ |
| $\hat T^\star,\ r_t$ | 周期 | 到达步数 = 周期数 |
| $L_\text{RUL}$ | 周期² | $(\hat T^\star-r_t)^2$ |
| $L_\text{terminal}$ | 无量纲 | $(D_{n_e}-1)^2$ |
| $L_\text{smooth}$ | **无量纲** | $(\delta_{t+1}-\delta_t)^2$，两个无量纲量之差 |

三项可加要求 $\lambda_i L_i$ 同为周期²。注意 $\tilde L_\text{smooth}$ 除以 $S_0$（无量纲）后
量纲不变，故 $\lambda_T$ 与 $\lambda_S$ **同为 $L_\text{scale}^2$**：

$$\lambda_T=c_T\,L_\text{scale}^{2},\qquad \lambda_S=c_S\,L_\text{scale}^{2},
\qquad L=L_\text{RUL}+\lambda_T L_\text{terminal}+\lambda_S\tilde L_\text{smooth},
\qquad L_\text{scale}=\texttt{rul\_cap}=125 .$$

归一化只改数值尺度、不改量纲：$L_\text{terminal}$ 与 $\tilde L_\text{smooth}$ 现在都是 $O(1)$，
$\lambda$ 取 $125^2$ 后三项量级可比，D4.5 的梯度比才有意义。

**两次都取自表达式**：$L_\text{terminal}$ 与 $L_\text{smooth}$ 都是无量纲，
所以两者都乘 $L_\text{scale}^{2}$，没有 $L_\text{scale}^{4}$。

> **为什么容易出现"四次"的错觉**：若把 $L_\text{smooth}$ 改写成**连续时间导数**的形式
> $\dot D_t=\delta_t/\Delta t$（$\Delta t$ 以周期计），则 $\dim\dot D=1/\text{周期}$，
> $(\Delta\dot D)^2$ 的量纲变成 $1/\text{周期}^2$，$\lambda_S$ 就需要周期⁴，
> 数值上等于上式的 $L_\text{scale}^{2}=125^{2}$ 倍。两种写法只差一个**常数因子 15625**。
> 因此：**配置里必须写 $\lambda$ 的数值，不能只写 $c_S=1$**——"$c_S=1$"在两种读法下相差 15625 倍，
> 而损失曲面对这个倍率是敏感的。初始版取 $c_T=c_S=1$，即 $\lambda_T=\lambda_S=125^2=15625$，
> 并把该数值写进 `config`，训练日志打印三项的加权值与 $g_i$（D4.5）。

**禁止把 $\lambda$ 当超参扫描**：它们已被量纲锁死，扫 $\lambda$ 等于扫单位，没有物理意义。

### D4.5 梯度贡献检查（必须实测并落盘）

每个 epoch 记录
$$g_i=\Big\|\nabla_\theta L_i\Big\|_2,\qquad i\in\{\text{RUL},\text{terminal},\text{smooth}\},$$
并要求 $g_i/\max_j g_j\in[0.1,\,1]$ 持续成立。任一辅助项低于 0.1 即视为"辅助任务失效"
（任务书 §7 D4 的原话），按 $c_T$ 或 $c_S$ 上调一次后重测，**不靠猜**。
$g_i$ 写入 run 的 `metrics.json` 与训练日志。

### D4.6 封顶偏差：必须量化，不能只看总体 RMSE

**实测封顶占比**（`probes/d_label_stats.py`）：

| 子集 | 训练窗 $r_t\ge125$ | val 窗 $r_t\ge125$ |
|---|---|---|
| FD001 | 40.3% | 7.0% |
| FD002 | 39.9% | 8.5% |
| FD003 | 51.2% | 11.0% |
| FD004 | **48.9%** | **20.4%** |

**偏差机制（方向是乐观的）**：对删失样本 $\ell_t=(\max(0,125-\hat T^\star))^2$，
任何 $\hat T^\star\ge125$ 都给出**零梯度**；而评估时 $\hat T^\star$ 被截断到 125，
误差同样是 0。两处叠起来，**封顶窗口对报出的 RMSE 贡献恒为 0**，
无论模型预测 126 还是 543。于是 `val_engine_mean_rmse` 实际上是**在未封顶子集上算的**，
有效样本量按子集缩水 7%–20%；FD004 有五分之一的 val 窗口不参与区分。

训练侧同样偏：FD001–FD004 有 **40%–51%** 的训练窗口是删失的，
$L_\text{RUL}$ 的梯度有一半来自"把 $\hat T^\star$ 推过 125"这一个方向，
这正是 D4.5 梯度检查必须逐 epoch 记录的原因。

**量化方案（B2/B3 落盘时必须报，缺一不可）**：

1. `frac_capped`：各子集 val 封顶占比（上表，从数据读）。
2. `boundary_hit`：$P(\hat T^\star\ge125 \mid r_t\ge125)$。若 < 1，说明有封顶窗口被压到 125 以下、
   对 RMSE 有真实贡献；若 = 1，封顶窗口全部"隐身"。
3. `rmse_uncensored`：只在 $r_t<125$ 的窗口上算 RMSE。这是不受边界影响的诚实读数。
4. `rmse_uncapped`：**对未封顶真值 $r_t$ 算 RMSE，但只在 $r_t\le K_\text{model}$ 的窗口上算**，
   同时报 `frac_r_gt_K`。
   原因：真值 $r_t$ 最大可达 543，而 $\hat T^\star\le K_\text{model}=200$。FD004 的长引擎在
   $0.40/0.55$ 两个 cut 上 $r_t=326/244$，都超过 200——若把这些样本算进去，
   `rmse_uncapped` 会被**系统性压低**（不是抬高），看起来"变好"，实际是截断造成的。
   被截断的尾部必须单列，不能混进比较。
5. `rmse_by_censoring`：把 val 窗口按 $r_t<125$ / $r_t\ge125$ 分层，**两层分别报 RMSE**；
   并报封顶层里 $\hat T^\star$ 恰好落在 125 的比例（`exact_boundary`）。

第 4 条是关键：**val 窗口是训练发动机的截断切分，发动机跑到失效，所以 $r_t=n_e-t$ 是已知的**，
不是未知量。因此可以完全绕开封顶，直接对真实 RUL 评估。这条必须和封顶指标并报。

**D-G4（软硬到达一致）也要分层**：$\overline{\hat T^\star}$ 与 $\overline{T^\text{hard}}$
的差在**未封顶层**和**封顶层**分别计算。封顶层本来就该落在边界上，混在一起会把
软到达的系统偏差掩盖掉。

**判据归属**：D-G1 用封顶指标判（与 Gate A 口径可比），但 `rmse_uncapped` 同时报出。
若封顶指标改善而 `rmse_uncapped` 恶化，判为**拟合截断**而非真实进步，D-G1 记不过。
该比较**只在 $r_t\le K_\text{model}$ 的子集上成立**，且四个 run 用同一 val 集，
`frac_r_gt_K` 应当逐位相同——若不同，说明数据接口有漂移，先查接口再判。

**hinge 的分层检查（军师 Q5）**：hinge 在封顶区不识别超过 125 之后的真实大小，
但**不能先验断言**它一定把预测推向边界——那取决于完整损失与反传，不是常值轨迹能推出来的。
所以处置是：**暂保留 hinge**，在 D3 算子修完、训练可跑之后，按封顶层/未封顶层**分层**检查
$\hat T^\star$ 的分布与 `boundary_hit`；只有当分层结果显示封顶层系统性贴边、且
`rmse_uncapped` 同步恶化时，才升级为删失正态似然（那是一次预注册变更，不是调参）。
顺序不能倒：先修算子，再分层查，最后才谈换损失。

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

**四臂必须统一的四件事（军师 Q4）**：B1 桥只能说明"有 HI 但走直接头"这一层，它**不足以**
单独分离"输出参数化"与"优化难度"。要让四臂可比，除骨干与预算外还必须统一：

1. **损失**：$L_\text{RUL}$ 的形式与删失处理在四臂间一致（B0/B1 也按 D4.1 的删失口径，
   不得 B0 用封顶 MSE 而 B2/B3 用删失 hinge）；
2. **裁剪**：$\hat T^\star$ 或直接头输出的截断区间四臂一致（都用 $[0,\texttt{rul\_cap}]$）；
3. **样本权重**：封顶样本是否加权、权重多少，四臂同一套；
4. **更新预算**：同一 `epochs`/`patience`，且**报实际更新步数**——早停触发点不同会让
   有效预算不同，这一项必须显式报告而不是默认相等。

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
8. **$k$ 是外层循环**：未来速率滚动对 $k$ 循环、窗口走批量维。禁止按窗口逐个滚（D3.4）。

### D7.1 数据接口（`data_hc` 现在逐窗口采样，D 需要发动机级）

现有 `data_hc.build` 返回的是**窗口数组**（`x/y/mask/baseline/cond/cycle_hc/cycle_raw`），
窗口之间没有"属于哪台发动机、从第几个周期开始"的字段，D 分支无法据此做前缀递推。
必须新增一个并列接口，**不改动现有接口**（C 的 28 格证据依赖它）。

**而且不能加在 `data_hc.py` 里**：该文件是 v9 code lock 的 15 个 blob 之一
（md5 `3ad7c111cf0268409bb40d79f7271cee`），改它会让所有记录 `code_md5` 的 Stage C run 失效。
因此接口放在**新模块 `paper_b/hc/data_d.py`**，只 import `data_hc`、不修改它。

```
data_d.engine_batch(subset, data_dir, span, cuts, split='train'|'val') -> {
  x:          (E, Lmax, span, C)   # 按发动机补齐，右侧有效
  mask:       (E, Lmax, span)      # 窗口内 mask（右对齐，与冻结一致）
  baseline:   (E, Lmax, C)
  cond:       (E, Lmax)
  valid:      (E, Lmax)            # 该发动机真实长度内为 1，补齐位为 0
  cycle_raw:  (E, Lmax)            # 真实年龄（推理期只用 <= c 的部分）
  r:          (E, Lmax)            # 未封顶 RUL = n_e - t
  y:          (E, Lmax)            # 封顶标签 = min(r, rul_cap)
  engine_id:  (E,)                 # 发动机标识，用于 cluster/分组统计
  lengths:    (E,)
  cut_t:      (E, K)               # 仅 'val'：五个协议 cut 的周期位置
  cut_valid:  (E, K)
}
```

约束：

- `Lmax` 是**该 batch 内**的最大长度，不是全局最大；`valid` 负责屏蔽补齐位。
- val 侧的 5 个 cut 必须**共用同一条发动机级 $D$**：先对 $1..n_e$ 递推，
  再在各 cut 位置取值，**不得每个 cut 重新起 0**（牢莫第 4 条）。
- 训练/验证的前缀只能来自 $X_{1:s-1}$，与推理一致。
- 该接口与窗口接口必须对同一 (engine, t) 给出同一个 $z_t$：用
  `max|z_window(t) - z_engine(e,t)| <= 1e-6` 断言两者一致，否则说明窗口构造漂移。

### D7.2 可跑断言（写代码前先写这个测试）

1. **重叠一致性**：同一发动机取窗 $w_1$ 与 $w_2$，对重叠周期 $t$ 断言
   `max|D_t(w1) - D_t(w2)| <= 1e-6`。
2. **增量闭合**：`D_s(w2) - D_{s-1}(w1) == δ_s`（容差 1e-6）。
3. **单调**：`diff(D) >= -1e-6`。
4. **两接口一致**：D7.1 最后一条的 $z_t$ 断言。

这四条不过，不写训练代码。

---

## D8. 未决项

- **cycle 在 D 中的形态**：Stage C 的结论是 raw 标量优于 Fourier（−5.03%，四子集全负），
  但 G4 判据替换仍 **PENDING-L2**。因此 D 的 B0–B3 **先不含 cycle**（符合任务书 C→D 顺序），
  cycle 融合留到 Stage E 的 C0–C3。
- **$\alpha$ 与 $\beta$ 的取值**：初始版固定（$\alpha=0.2$、$\beta_s=4$/周期），不做搜索；
  若 D-G4 不过再按 D3.2 重标定一次。
- **$n_e$ 较大时的显存**：整条前向的序列长度可达数百，若显存不够，退回 D1.4 的备选，
  但必须同时提交一致性检验结果。
