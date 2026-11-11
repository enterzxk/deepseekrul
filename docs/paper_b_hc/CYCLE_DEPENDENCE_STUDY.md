# CYCLE_DEPENDENCE_STUDY — 注入点对照（A3a vs A3p，第一步）

状态：进行中（仅完成 Gate A 前置的注入点对照；A0–A5 全表未跑）
最后更新：2026-11-12 00:25 (+08)
证据 run_id：`HC-C-A3p-FD001-s101-20261112T0030Z`、`HC-C-A3a-FD001-s101-20261112T0030Z`
基线：v9（代码 `d1e0527`，锁 `6c405ad`，`baseline_version=v9`）

## 1. 为什么先做这一步

Gate A 第一条判据是「Fourier 是否优于简单年龄编码」。但 A3a 把 Fourier 项加在
`encode()` 返回的 **post-norm** 表示上，而冻结 DSPT（`dspt.py:239-244`）把它加在
**pre-norm**。形式同、位置不同，若不做对照，「Fourier 有用」与「注入点有用」分不开。

## 2. 对照设计（v9）

- 共享初始化：`--dump-init` 在 v9 上是**导出后退出**（不训练、不写 launch/close）；
  产物 `experiments/hc/tmp/init_v9_A3p_s101.pt`，sha256 `336074a6e173d8df70bba5f5053508990318260c793a4c4515ad88033456c3c8`。
- 两个训练 run 都 `--init-from` 同一份 `.pt`，manifest 的 `init_sha256` 相同（已核）。
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

**Δ = A3a − A3p = +0.4216（相对 A3p +3.87%）**：同一份初始化、同一 seed、同一骨干，
仅把 Fourier 项从 pre-norm 移到 post-norm，验证集发动机级 RMSE 变差 0.42。

## 4. 解释与对 Gate A 的影响

- 注入点**不是中性的**：A3a 相对 A3p 有一个量级与「Fourier vs 简单年龄」预期增益相当的惩罚。
- 因此 Gate A 的 Fourier 臂应以 **A3p（冻结注入点）** 为主，A3a 作为敏感性臂单列；
  若只跑 A3a，任何「Fourier 有用」的结论都会被注入点惩罚稀释甚至反号。
- 该 Δ 只用于判断注入点是否可忽略，**不构成 Fourier 有效性的证据**。

## 5. 限制

- 单 seed、单子集（FD001）、单次 select-on-val，尚未测同 seed 重复的噪声底线；
  +0.42 是否显著要在噪声底线出来后才能判。
- 两个 run 都只跑了 FD001；四子集结论未出。
- 该对照是 Gate A 的前置，不进 Gate A 主表。

## 6. 失败的尝试（全部 superseded，不作证据）

v5/v6/v8 上的共享初始化尝试均因 `init remap` 失败而终止（`missing=['pe_periods']`、
`unexpected=['cycle_enc.proj.*']` 等），对应 run 标 `superseded`；v9 修复后才得到上表。
