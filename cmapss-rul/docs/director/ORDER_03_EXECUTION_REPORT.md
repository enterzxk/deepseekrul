# ORDER_03 执行报告（截至提交时刻）

- 基线提交：`5134c05`
- 本轮提交：见文末
- 服务器：`/home/yndx/zxk/bearing-rul-pi`（A40）
- 总控：`~/p8/o3.sh`（setsid + nohup，按 cell 断点续跑）
- 输出：`experiments/order03/<M_tag>/<subset>/<variant>_s<seed>/`
- 日志：`logs/order03/`

> **诚实声明**：报告写于网格启动后不久。**所有 M0–M4 单元格仍在运行**，
> 本文件不含任何性能结论；已实现的机制有代码与冒烟证据，性能待跑完后补。

---

## 1. Step 1：Sensor-ID 证据链与零初始化门控

**已实现**（`src/candidates/model.py`）：

```
e_id = alpha_id * Embed(sensor_id),   alpha_id = 0.0  (可学习标量, 零初始化)
h = h + e_id
```

- 单个可学习标量控制整张身份表，初始为 0 → **初始完全无身份偏置**，
  单工况（FD001）上不会先过拟合再纠正；只有多工况数据能把它推开。
- 参数：`Embedding(14,64)=896` + `alpha_id=1` = **897**。
- S2 参数匹配对照：共享低秩 `64→7→64` = **896**（448+448），
  完全保持通道置换对称（`permutation_delta=0`，已由测试验证），与 S1 参数差 **1 个**，
  远优于要求的 ±50。
- S3（14 个独立输入投影）、S4（冻结正弦通道码，+0 可训练参数）已实现。
- **S5（tfA + 身份/通道注意力）未实现**，如实标注。

**状态**：代码完成；Aid/S2/S3/S5 的 3-seed 对照在 ORDER_03 网格中
（S5 缺失，需下一轮补）。

---

## 2. Step 2：真·分工况健康参考（已落地）

`build_windows_condref()` 实现：

1. **工况簇**：训练集 `op_1..op_3` 上 `KMeans(K=6)`（FD001 视为 K=1）。
2. **个体参考**：发动机 i 在簇 m 的早期（cycle ≤ 30）前 5 个样本均值
   `xbar_(i,m)`，样本数 `n_(i,m) ∈ [0,5]`；**只使用截止时刻之前**的行。
3. **群体先验**：训练集全体在各簇前 15 周期的均值 `mu_pop,m`。
4. **贝叶斯收缩**：`x_ref = n/(n+tau)*xbar + tau/(n+tau)*mu_pop`，`tau=2.0`。
   * `n=0` → 100% 回退群体先验（从未进入该工况时不会注入错误个体基线）；
   * `n` 充足 → 平滑过渡到个体出厂基线；
   * FD001（K=1）→ 自然退化为稳定的早期健康基线。
5. **6 维证据**：e1 覆盖度、e2 个体/回退掩码、e3 获取年龄、e4 参考匹配
   （仍为传感器空间代理，**未按规范用工况中心距离**，已标注）、e5 参考稳定性、e6 EMA 跳变。
6. **RACDR 门控**：`g = σ(MLP(e) + b_sensor) ∈ (0,1)^14`；
   `z = Proj_raw(x, ema) + g ⊙ Proj_ref(Δx)`。

**三种参考模式**：`--ref-mode cond | pop | global`（`pop` 令 tau=1e6 → 纯群体先验；
`global` 走旧的前 5 周期全局均值，用于 M1 对照）。

**冒烟证据**：FD002 / A / cond 模式 2 epochs 端到端跑通
（`val=16.42`，仅证明管线可用，不是性能结论）。

---

## 3. Step 3：M0–M4 正交矩阵（已启动，30 epochs，3 seeds）

| 单元 | 含义 | 实现 |
|---|---|---|
| M0 | 原始传感器 Transformer | `tf`（`--ref-mode cond` 但模型不使用 extras） |
| M1 | 全局前 5 周期参考（旧特征） | `A --ref-mode global` |
| M2 | 仅群体分工况参考 | `A --ref-mode pop`（tau=1e6） |
| M3 | 收缩分工况参考（无门控） | `A --ref-mode cond` |
| M4 | RACDR 完整 | `RACDR --ref-mode cond`（+ 零初始化门控 Sensor-ID + 非对称损失 late=1.8） |

- 统一 `split_seed=7`，`model_seed ∈ {2027,2028,2029}`，30 epochs，patience 10。
- 非对称安全损失：`late_weight=1.8`（ORDER_03 要求），并对 `delta>0` 加倍平方。
- 评价：全窗口 val、Protocol B（0.5/0.7/0.9）、Protocol C（60/100/140/180/220 + 存活规则）、
  matched-cutoff（`U[40,0.95T]`，seed 2026）。逐发动机预测 CSV 全部落盘。

**当前状态**：24 个 cell 已启动，**0 个完成**（30 epochs × FD002 约 10-12 分钟/个）。

---

## 4. Step 4：指标口径（已实现，数据待补）

`scripts/error_analysis.py` 输出：
- pooled RMSE（全部 cutoff × 种子合并）
- macro RMSE（各 cutoff 先算 RMSE 再平均）
- NASA、late fraction、severe-late(>10)
- 分寿命段误差

`scripts/audit_stats.py`：发动机级聚类 Bootstrap（同一发动机的 3 cutoff 一起重采样），
M4 vs M0/M1/M3 的 95% CI 与 `prob_a_better`。

**安全门禁阈值**已按批复写入：`late_fraction ≤ 0.52`。

---

## 5. 未完成 / 风险

| 项 | 状态 |
|---|---|
| M0–M4 全部性能结果 | **运行中**（报告不含结论） |
| S5 | 未实现 |
| e4 工况中心距离 | 未实现（仍为传感器空间代理） |
| 3 seeds 的 S2/S3 | 在队列中（本轮网格不含 S2/S3，需补） |
| Level B | 未启动（预登记已写） |

- **R1**：本轮报告不含性能数据，不能作为阶段结论。
- **R2**：e4 偏离规范，是"可靠性证据"主张的薄弱点。
- **R3**：GPU 上有前一轮遗留进程（19 个 run_candidate），会拖慢本轮；
  建议下一轮先 `pkill -f run_candidate` 再启动。

---

## 6. 续跑

```bash
ssh lab-server
pkill -9 -f run_candidate.py      # 清理遗留进程
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash ~/p8/o3.sh > ~/p8/o3.out 2>&1 < /dev/null &
# 已完成的 cell 自动 skip
python scripts/error_analysis.py --root experiments/order03/M4_racdr --out results/order03/M4
```

---

## REQUEST FOR DIRECTOR DECISION

1. **e4 是否必须改为工况中心距离**？当前是传感器空间代理，若不改，"可靠性证据"的
   创新主张只能靠 e1/e2/e5 支撑。
2. **S5 与 S2/S3 的 3-seed 对照是否必须并入本轮**（会再增加约 40 分钟），还是留到
   M0–M4 结论出来后再补？
3. **late_fraction ≤ 0.52 的硬门禁**是否对所有单元（含 M0 基线）统一适用？
   若基线本身 >0.52，是否改为"相对基线不劣化"？

未解封 FD003/FD004，未宣布最终模型。
