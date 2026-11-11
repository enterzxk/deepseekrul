# CYCLE_DEPENDENCE_STUDY — 注入点对照（A3a vs A3p，第一步）

状态：进行中（仅完成 Gate A 前置的注入点对照；A0–A5 全表未跑）
最后更新：2026-11-12 00:25 (+08)
证据 run_id：`HC-C-A3p-FD001-s101-20261112T0030Z`、`HC-C-A3a-FD001-s101-20261112T0030Z`
基线：v9（代码 `d1e0527`，锁 `6c405ad`，`baseline_version=v9`）

> **符号约定（全文统一，与 `GATE_A_WAVE1_PLAN.md` §9 一致）**
>
> `Δ(A→B) = (RMSE_A − RMSE_B) / RMSE_A`，**正号 = B 优于 A**（B 的误差更低）。
>
> 因此 G1 的「均值 ≥ 1.0%」读作「B 平均改善 ≥1.0%」，G3 的「≤ −10.0%」读作「A0 劣化 ≥10%」。
> 本文早期版本曾在 §3 用「A3a − A3p」的相反方向记录过一处，已按本约定改写并保留原始 RMSE 差值。

## 1. 为什么先做这一步

Gate A 第一条判据是「Fourier 是否优于简单年龄编码」。但 A3a 把 Fourier 项加在
`encode()` 返回的 **post-norm** 表示上，而冻结 DSPT（`dspt.py:239-244`）把它加在
**pre-norm**。形式同、位置不同，若不做对照，「Fourier 有用」与「注入点有用」分不开。

## 2. 对照设计（v9）

- 共享初始化：`--dump-init` 在 v9 上是**导出后退出**（不训练、不写 launch/close）；
  产物 `experiments/hc/tmp/init_v9_A3p_s101.pt`，sha256 `336074a6e173d8df70bba5f5053508990318260c793a4c4515ad88033456c3c8`。
- 两个训练 run 都 `--init-from` 同一份 `.pt`，manifest 的 `init_sha256` 相同（已核）。
- **同一组初始权重：双重证据，缺一不可。** 两份 dump 的 payload sha256 不同
  （`init_A3p_FD001_s101.pt` = `2138ac51…`，`baseline_version=v8`；
  `init_v9_A3p_s101.pt` = `336074a6…`，`baseline_version=v9`），但 60 个 `state_dict`
  张量**逐位相等**（`torch.equal` 逐张量比对，60 比对 / 0 不匹配），差异仅在 payload
  元数据（`baseline_version` v8/v9、`created_utc` 16:01:36Z vs 16:06:30Z）。
  因此重复 run 的逐位一致来自**同一组初始权重（起点相同）＋ 训练确定性（同 seed 同输入
  同输出）**，两个独立事实合起来才使 σ=0 成立；只有其中一个都不够。
  **这也说明一致「而非产物复制」**——两个 run 的数字一模一样时，审稿人的第一反应是
  「是不是复制了结果」，这句直接堵住那个怀疑。
- 附带推论（**推断，非已核事实**）：v8→v9 改了 init remap（`copy pe_periods, do not pop`），
  但 A3p 的 dump 权重一位没变，说明该修复只影响**载入路径**、不影响**导出权重**。
  这与「A3p 在 v6/v8/v9 三版上数值完全一致」互相印证。
- 键映射：`pe_proj ↔ cycle_enc.proj`、`pe_periods → cycle_enc.periods`（copy，不 pop）；
  载入后 `missing=0 unexpected=0`，否则 raise。
- `verify_loop` 三步断言（v9）：共享骨干逐张量 0 差；`pe_proj(fourier(c)) == cycle_enc(c)` 0 差；
  两侧循环投影置零后 `A3a(x) == A3p(x)` ≤ 1e-6。三项在 a40 上均为 `0.000e+00`，`VERIFY_LOOP PASS`。
  真实初始化下两者**必须不相等**（pre-norm vs post-norm），那个差就是 treatment。

## 3. 结果（FD001，seed 101，select-on-val）

| 变体 | 注入点 | `val_engine_mean_rmse` | `val_pooled_rmse` | best epoch | params |
|---|---|---|---|---|---|
| A3p | frozen pre-norm | **10.8951** | 11.8242 | 24 | 115586 |
| A3a | post-norm | 11.3167 | 11.9339 | 25 | 115586 |

**Δ(A3p→A3a) = −3.87%**（RMSE 差 `A3a − A3p = +0.4216`，按统一约定记作负值）：
同一份初始化、同一 seed、同一骨干，仅把 Fourier 项从 pre-norm 移到 post-norm，
验证集发动机级 RMSE 变差 0.42，即 **A3a 比 A3p 差 3.87%**。

## 4. 解释与对 Gate A 的影响

- 注入点**不是中性的**：A3a 相对 A3p 有一个量级与「Fourier vs 简单年龄」预期增益相当的惩罚。
- 因此 Gate A 的 Fourier 臂应以 **A3p（冻结注入点）** 为主，A3a 作为敏感性臂单列；
  若只跑 A3a，任何「Fourier 有用」的结论都会被注入点惩罚稀释甚至反号。
- 该 Δ 只用于判断注入点是否可忽略，**不构成 Fourier 有效性的证据**。

## 5. 限制

- 单 seed、单子集（FD001）、单次 select-on-val。同 seed 重复的噪声底线**已测**（见 §7），
  但四子集与多 seed 仍未出。
- 两个 run 都只跑了 FD001；四子集结论未出。
- 该对照是 Gate A 的前置，不进 Gate A 主表。

## 6. 失败的尝试（全部 superseded，不作证据）

v5/v6/v8 上的共享初始化尝试均因 `init remap` 失败而终止（`missing=['pe_periods']`、
`unexpected=['cycle_enc.proj.*']` 等），对应 run 标 `superseded`；v9 修复后才得到上表。

## 7. 确定性证据（按证据强度排序）

**① 跨版本可复现（最强：同时证明改动未触碰该路径 + 跨版本可复现）**

A3p 在 v6 / v8 / v9 三个代码版本上四次运行**逐位相同**：
`1508Z`(v6) = `0005Z`(v8) = `0030Z`(v9) = `R1-0035Z`(v9) = **`10.895096839289387`**。
A3a 在 v9 上三次运行逐位相同：`1640Z` = `0030Z` = `R1-0035Z` = **`11.316705661052403`**。

**② 同 seed 重复 σ = 0**

A3a 3 次、A3p 5 次，同一配置重复运行得到**逐位相同**的值（不是「差在 1e-6 以内」，是浮点逐位相等）。
两份 init dump 的 payload sha256 不同（`2138ac51…` v8 / `336074a6…` v9），但 60 个
`state_dict` 张量逐位相等，差异仅在 `baseline_version` 与 `created_utc` 两个溯源字段——
所以重复一致来自**同一组初始权重 + 训练确定性**，**而非产物复制**。

**③ 初始化能造成 0.77 的假效应（反衬 Δ = 0.42 的量级）**

v6 的 `A3a-1505Z`（**无** `--init-from`，即 A3a 与 A3p 各抽各的初始化）得 `10.54547792713183`，
而 v9 的 A3a（**共享**初始化）得 `11.316705661052403`，**差 0.77**，且方向相反
（v6 下 A3a 优于 A3p，v9 下 A3a 劣于 A3p）。这说明：不控制初始化，注入点效应会被
初始化噪声淹没甚至反号。**Δ = 0.42 只有在共享初始化下才是可解释的量。**

**④ 范围限定（必须与结论一起出现）**

单 seed、单子集（FD001）、单次 select-on-val。结论是
「**注入点有可测、可复现的效应**」，**不是**「pre-norm 普遍更好」。
推广到其他子集/其他骨干需要另外的实验，本文不主张。

## 8. 不作为证据的运行

`A3p-0105Z`：在 **ep001 被主动终止，无指标**。台账 note 记
「ep001 被主动终止，无指标，不构成任何证据」——**它不能被读成「第四条 A3p 跑出了不同值」**，
否则会变成非确定性的假证据。它已被另外 5 个逐位相同的点完全覆盖。
