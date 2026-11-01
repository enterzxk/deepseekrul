# 校准层：数学规格（CALIBRATION_MATH_SPEC）

本文件给出校准层的完整数学定义，使任何读者可仅凭本文件与冻结权重复算全部数字。
凡与代码不一致处，以代码为准并须在本文件修正——这是它存在的唯一理由。

- 代码：`scripts/cal/layer.py`（分布与拟合）、`scripts/cal/v2_compare.py`（对照协议）、`scripts/cal/calibrate.py`（数据装配）
- 冻结权重：`experiments/cal/calibrators/frozen_A.json` 等三个文件，含 66 个数与 `dev_sha256`
- 端到端复算：`scripts/cal/verify_frozen.py`，误差 1.05e-06 / 1.45e-06 / 2.45e-06
- 数据来源与证据等级：`experiments/cal/RUN_RECORD.md`

---

## 1. 对象与记号

RUL 标签定义在整数格 $\mathcal{Y}=\{0,1,\dots,125\}$ 上。上限 125 由**标签截断约定**产生：
所有真实剩余寿命超过 125 的样本被写作 125。记

$$
Y_i=\min\{R_i,\,125\},\qquad R_i \text{ 为真实剩余寿命},\qquad \hat y_i=g_\theta(x_i)\in\mathbb{R}
$$

其中 $g_\theta$ 是**骨干**（冻结，本层不触碰），$x_i$ 是长度为 50 的传感器窗口，$\hat y_i$ 是骨干的点预测
（多检查点集成后的均值）。校准层只使用 $\hat y_i$，不使用 $x_i$。

关键在于 $Y_i$ 不是 $R_i$ 的观测，而是 $R_i$ 的**区间删失观测**：

$$
Y_i=125 \iff R_i\ge 125
$$

因此 $y_i=125$ 承载的信息是"存活到 125 之后"，而非"恰好等于 125"。这条读法决定后面全部数学。

---

## 2. 校准层的定义

### 2.1 位置-尺度参数化

设 $u=\hat y/125$。两层网络 $f:\mathbb{R}\to\mathbb{R}^{2}$，

$$
(\mu,\sigma)=\bigl(\hat y+20\,f_1(u),\ \ 20\,\operatorname{softplus}\bigl(\mathrm{clamp}(f_2(u),-4,3)\bigr)+1\bigr)
$$

- 位置偏移 $20 f_1$ 的量纲是 RUL 周期，$20$ 使该层在 $\pm20$ 周期内可自由移动；
- 尺度下界 $+1$ 防止 $\sigma\to0$ 造成似然发散，上界经 softplus 的 clamp 限制在 $\approx 204$；
- $f$ 的形式：`1→16→tanh→2`，参数 $1\cdot16+16+16\cdot2+2=66$。

对照形式（同一接口，仅 $f$ 换掉）：

| 记号 | $f$ 结构 | 参数 |
|---|---|---:|
| `cens` | `1→16→tanh→2` | 66 |
| `trunc` | 同 `cens`，仅似然换为密度网格 | 66 |
| `emos_c` | 单层线性 `1→2` | 4 |
| `emos_u` | 单层线性，密度网格似然 | 4 |
| `homosc` | $\mu=\hat y$，$\sigma=s_0=\mathrm{sd}(y-\hat y)$（dev 上），不训练 | 1 |

### 2.2 概率质量函数

**正确形式（`mode="censored"`）**：以正态 CDF $\Phi$ 差分给格点质量，最后一格给存活函数

$$
p_0=\Phi\!\Bigl(\tfrac{1-\mu}{\sigma}\Bigr),\quad
p_b=\Phi\!\Bigl(\tfrac{b+1-\mu}{\sigma}\Bigr)-\Phi\!\Bigl(\tfrac{b-\mu}{\sigma}\Bigr)\ (1\le b\le124),\quad
p_{125}=1-\Phi\!\Bigl(\tfrac{125-\mu}{\sigma}\Bigr)
$$

即 $p_b=P(Y=b)$ 对 $b<125$，$p_{125}=P(Y\ge125)=P(R\ge125)$。最后一格是**原子**，
其质量等于尾部概率，而不是密度在 125 处的取值。

**错误形式（`mode="density"`，v1 的做法）**：

$$
p \propto \operatorname{softmax}\bigl(-\tfrac12 z^2\bigr),\qquad z_b=\frac{b-\mu}{\sigma}
$$

这是在格点上取密度再归一化，数学上等于把正态分布**截断**到 $[0,125]$。
两者的顶格质量相差约 8 倍（实测见 §3）。这就是 v1 报告里最大的技术错误。

### 2.3 似然

整数格上的区间删失对数似然：

$$
\ell_i=\begin{cases}
\log\Bigl[\Phi\bigl(\tfrac{y_i+1-\mu_i}{\sigma_i}\bigr)-\Phi\bigl(\tfrac{y_i-\mu_i}{\sigma_i}\bigr)\Bigr], & y_i<125\\[4pt]
\log\Bigl[1-\Phi\bigl(\tfrac{125-\mu_i}{\sigma_i}\bigr)\Bigr], & y_i=125
\end{cases}
$$

第二行就是 Tobit 型右删失项：封顶样本只要求 $R\ge125$，不惩罚超出部分。

**这是本层唯一有技术内容的部分，而它是标准 Tobit（Tobin 1958）在离散格上的重写。**
审计结论见 `paper/CALIBRATION_NOVELTY_AUDIT.md`。

### 2.4 估计

$$
\hat\Theta=\arg\min_{\Theta}\ -\frac1n\sum_i \ell_i+\lambda\|\Theta\|_2^2,\qquad \lambda=10^{-3}
$$

Adam，$\mathrm{lr}=0.05$，400 步全批量，`float64`，`seed=0`。样本量 $n=709$（四个舰队 dev 合并）。

### 2.5 尺度校准

$\sigma$ 乘一个标量 $s$，$s$ 在**交叉拟合的 dev** 上按"达到名义 90% 覆盖的最小 $s$"选取：

$$
s^\star=\min\Bigl\{s\in\{0.70,0.75,\dots,2.50\}:\ \widehat{\mathrm{PICP}}_{90}^{\text{cv}}(s)\ge0.90\Bigr\}
$$

交叉拟合：按 `seed=0` 置换后均分 5 折，第 $k$ 折的预测由其余 4 折拟合的模型给出。
取"最小可达"而非"gap 最小"，因为覆盖率对 $s$ **非单调**：$s$ 过大时 pmf 趋于格上均匀分布，
其中央 90% 区间不再包含顶端原子，覆盖率反而回落。若不取最小可达，argmin 会落到这条病态分支上。

**注意 $s$ 的三重身份**：它既是尺度校准，也是本层唯一的"覆盖率旋钮"。因此用它来把不同方法的
实测覆盖率对齐到 0.90 是循环论证——那等于用测试标签做了一次本层未包含的校准（见 §6.3）。

---

## 3. 顶格质量：错与对的量化

`scripts/cal/diag_cap_mass.py` 在冻结的 v1 权重上直接测量：

| 子集 | 实测平均 $p_{125}$（density） | 应为 $1-\Phi\bigl(\tfrac{125-\mu}{\sigma}\bigr)$ | 比值 |
|---|---:|---:|---:|
| FD001 | 0.0228 | 0.1892 | 8.3× |
| FD002 | 0.0265 | 0.2125 | 8.0× |
| FD003 | 0.0350 | 0.2721 | 7.8× |
| FD004 | 0.0360 | 0.2813 | 7.8× |

后果：dev 中 27.9% 的样本标签为 125（FD003 达 36%），每一个都被系统性低估概率约 8 倍，
似然把它们当作"几乎不可能发生"，于是把高 RUL 段的预测往下拉。
这正是之前被归因于"标签约定本身"的 $[100,125]$ 段 −10 周期偏差的真实来源。

修正后，同一段的偏差从 $[-7.37,-8.42,-4.59,-6.39]$ 变为 $[-2.54,-3.25,+1.35,-0.66]$。

---

## 4. 该族的表示能力边界（本文件最重要的一节）

校准层是 $\hat y$ 的可测函数。设 $\mathcal G=\{\text{所有 }\hat y\text{ 的可测函数}\}$，
则本层、EMOS、CRUDE、等渗 PIT 全部是 $\mathcal G$ 的元素。于是有一个与实现无关的上界：

**命题 1（MSE 最优性）.** $\min_{g\in\mathcal G}\mathbb E[(g(\hat y)-Y)^2]$ 的解是 $g^\star(\hat y)=\mathbb E[Y\mid\hat y]$。
在单调约束下其估计量是等渗回归，即 `isot_exact` 与 `v2_compare.make_isotonic` 所做的事。

**推论.** 任何"只吃 $\hat y$"的校准器，其平方误差不可能优于 $g^\star$。若 $g^\star$ 仍未消除
按真实 RUL 分箱的条件偏差，则该偏差属于 $\hat y$ 的信息含量，而非校准器的容量或目标函数。

实测（`scripts/cal/diag_bias_tradeoff.py`，三骨干）。**关键是要用"在 test 本身上拟合"的等渗回归**，
否则会把 dev→test 的漂移误当成"映射不对"：

| 映射 | RMSE | 最差分箱偏差 | 五个分箱偏差 |
|---|---:|---:|---|
| 骨干原样（A） | 11.61 | **5.80** | — |
| 等渗，dev 上拟合（A） | 11.76 | 6.84 | 2.7 / 1.7 / 5.0 / 4.6 / −6.8 |
| **等渗，test 上拟合（A）** | 10.96 | **6.53** | 1.1 / 1.1 / 5.6 / 5.0 / −6.5 |
| 逐分箱 oracle（不可实现） | 10.60 | 0.00 | 0 / 0 / 0 / 0 / 0 |

三骨干的 test 内拟合等渗，最差分箱偏差为 **6.53 / 7.18 / 6.39**。
即便给映射器直接看测试集、并取该函数类的 MSE 最优解，$\pm6.5$ 周期的分箱偏差依然存在。
**MSE 最优与消除分箱偏差是两个不同的泛函，前者不蕴含后者。**

**命题 2（收缩单调恶化）.** 对一族收缩映射 $g_t(\hat y)=\hat y+t(\bar y-\hat y)$，
最差分箱偏差随 $t$ 从 0 到 1 **单调上升**（5.80→70.94）。故偏差不是"向均值回归"造成的，
把它拉回均值只会放大偏差。

**命题 3（只有看到标签才能修）.** 允许对每个分箱做最优平移（即允许读取真实 $y$）时，
最差分箱偏差降到 0，RMSE 由 11.61 降到 10.60。这说明偏差是**真实存在**的（修掉它连 RMSE 都变好），
但该平移依赖真实分箱归属，而分箱归属在预测时不可观测。

**机制.** `diag_identifiability.py` 给出直接证据：真实 RUL 落在 $[75,100)$ 的发动机，
其 $\hat y$ 横跨 32 到 124 周期。一个 $\hat y$ 的函数无法只作用于该分箱。
给定 $\hat y\in[80,90)$ 时 $Y$ 的标准差是总体标准差的 0.36 倍——预测值并不能把分箱切开。

**结论.** 条件覆盖的缺陷不能靠事后重校准消除。这不是本层实现得不好，而是
$\hat y\mapsto Y$ 的映射本身丢失了分箱信息。任何声称"加一个校准层改善条件覆盖"的
C-MAPSS 论文都应当先出示这条证据，否则其改进只可能来自测试标签的泄露。

**一个必须区分的边界：$[75,100)$ 的偏差与封顶无关.** 该段在 12 个实例上的封顶样本比例全部为 0.00
（`identifiability.json` 的 `p_capped`），故其 +2.96…+13.82 的高估不能用标签截断解释，是纯粹的条件偏差。
而 $[100,126)$ 的 −3.39…−7.88（该段封顶比例 0.32–0.60）是封顶约定的直接后果：
模型的支撑止于 125，均值必然低于 125。把后者当主缺陷是选错了靶子。

---

## 5. 评价泛函（逐个写清，避免名实不符）

设 pmf $p\in\Delta^{125}$，$y$ 为观测标签。

**5.1 覆盖与区间.** 令 $\alpha=0.10$，$F_k=\sum_{j\le k}p_j$，

$$
\mathrm{lo}=\min\{k:F_k\ge\alpha/2\},\qquad \mathrm{hi}=\min\{k:F_k\ge1-\alpha/2\}
$$

$\mathrm{PICP}=\frac1n\sum_i\mathbb 1[\mathrm{lo}_i\le y_i\le\mathrm{hi}_i]$，
$\mathrm{MPIW}=\frac1n\sum_i(\mathrm{hi}_i-\mathrm{lo}_i)$。

**5.2 Winkler（区间得分）.** $\ W=(\mathrm{hi}-\mathrm{lo})+\frac{2}{\alpha}(\mathrm{lo}-y)_++\frac{2}{\alpha}(y-\mathrm{hi})_+$。

**5.3 CRPS.** 对整数支撑有恒等式 $\mathbb E|X-y|=\sum_k|F_k-\mathbb 1[y\le k]|$ 与
$\mathbb E|X-X'|=2\sum_k F_k(1-F_k)$，故

$$
\mathrm{CRPS}(p,y)=\sum_{k=0}^{125}\bigl|F_k-\mathbb 1[y\le k]\bigr|-\sum_{k=0}^{125}F_k(1-F_k)
$$

**这一条曾经写错。** 早期表格里的"CRPS"只算了第一项，即 $\mathbb E|X-y|$，是
（合法的）绝对误差得分，不是 CRPS。第二项与 $y$ 无关但随方法变化，因此它不会改变
同一方法内部的一致性，却会改变**方法之间的排序**，而且把数字抬到与已发表 C-MAPSS CRPS 不可比。
现行 `v2_compare.py` 同时输出 `crps`（真 CRPS）与 `crps_l1`（旧的 $\mathbb E|X-y|$），旧表可用后者核对。

**关于离散支撑上的第三种口径.** 文献里还有一个常用量，Ranked Probability Score
$\mathrm{RPS}=\sum_k(F_k-\mathbb 1[y\le k])^2$。它与上式**不同**，两者都是合法严格得分，
但数值不可互换。注意经典恒等式 $\mathrm{CRPS}=\int(F-\mathbb 1[y\le x])^2dx$ 对纯离散分布会退化
（阶梯函数的平方在跳跃点外处处为 0），所以不能由它推出 RPS；本文一律用 Gini 形式
$\mathbb E|X-y|-\tfrac12\mathbb E|X-X'|$，它与按样本计算 CRPS 的常用实现一致。
论文中须写明用哪一个，否则复现者算出的数会与表格不符。

**5.4 NLL.** $-\log p_{y}$，$p$ 截断于 $10^{-12}$。对经验型分布（CRUDE）该指标会因 pmf 有零格而爆炸，
这是它们的真实性质，不作平滑处理。

**5.5 条件版本.** 上述全部按真实 RUL 分箱重算，分箱
$\{[0,25),[25,50),[50,75),[75,100),[100,126)\}$，并给出 Wilson 95% 区间与分箱内的
平均偏差 $\overline{\hat y-y}$。**边际 PICP 与分箱 PICP 必须同时报告**，理由见 §4。

---

## 6. 协议（三个名字不能混用）

**6.1 per-subset.** 在舰队 $k$ 的 dev 上拟合，在舰队 $k$ 的 test 上评价。4 个独立结论。
**6.2 pooled.** 四舰队 dev 合并（709 台）拟合，逐舰队评价。**这不是零样本迁移**，
因为拟合时已见过该舰队的 dev 样本；它检验的是"一套参数跨舰队共用"。
**6.3 LOSO.** 留一舰队：在 3 个舰队 dev 上拟合，在第 4 个舰队 test 上评价。
**6.4 independent transfer.** 在 A 数据上拟合、在从未参与的 B 数据上评价（N-CMAPSS 属于此类）。

**6.3 的附加警告（本次新增）.** 用测试集把宽度加宽到实测 0.90 再比较 CRPS，是**自证**：
它等价于给每个方法单独做了一次以测试标签为监督的校准。该方法在比较中赢，
反映的是"它离 0.90 更远、因此加宽后获益更多"，不是形状更好。
正确的做法是 §2.5：宽度只在交叉拟合 dev 上选，测试集只看一次。
本次重跑后 CRUDE 的排名由第 1 掉到第 4，就是这个自证被移除的直接后果。

**6.5 pooled 与 LOSO 的实测差.** 因为本层只消费 $\hat y$，它天然跨舰队通用：
LOSO（在三个舰队 dev 上拟合、第四个舰队 test 上评价）与 pooled 的 CRPS 差在
−0.07 到 +0.14 之间（12 个实例），平均 +0.03，见 `experiments/cal/v2_tables.md` §T6。
这有两个后果：
(a) **不能**再把 pooled 结果说成"乐观"，它确实没乐观；
(b) 同时也**不能**把"跨舰队迁移"当作贡献来写——没有需要迁移的东西。

---

## 7. 已知未修正的问题

按严重程度排列，均须在论文中如实披露：

1. **dev 与 test 的封顶比例不一致**：dev 27.9%，test 16.0%（FD003 为 36.0% vs 13.4%）。
   校准层在 dev 上见过 1.7 倍的封顶样本，测试集没有那么多。
2. **测试集不是官方全集**：输入窗口需 50 周期，迫使丢弃 54 台（707→653），
   被丢弃者的平均 RUL 为 119–157，显著高于保留者（72–80）。即**最健康的一批发动机被系统剔除**。
3. **dev 是 3 模型集成、test 是 15 模型集成**，两者的预测噪声不同，会人为压低 dev 残差。
4. **骨干仅 3 个，且同属一个架构族**，因此"跨骨干一致"只证明不依赖单一训练种子，
   不证明不依赖架构族。
5. **正态形状固定**。Student-t 因 PyTorch 无可用可微 CDF 而放弃；
   训练与评分必须同族，否则重演 §3 那类名实不符。

---

## 8. 复算路径

```bash
# 1. 用冻结权重重建 pmf，与既有报告比对（应 < 1e-5）
python scripts/cal/verify_frozen.py

# 2. 13 方法同协议对照（每臂约 2 分钟，需 torch）
python scripts/cal/v2_compare.py --arm A

# 3. 生成论文表格
python scripts/cal/v2_tables.py

# 4. 条件覆盖与可辨识性证据
python scripts/cal/diag_band_shift.py
python scripts/cal/diag_identifiability.py
python scripts/cal/diag_bias_tradeoff.py
```
