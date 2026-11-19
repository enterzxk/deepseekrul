# Claude 改进方案：首轮执行记录（cap-aware LTAG / RHSE 故障检测）

> 日期：2026-11-19。来源：`C:\Users\20562\Desktop\improvement_plan.md`。
> 结论：**cap-aware 对齐被数据否定；RHSE 对 additive bias 很强，对 drift/stuck 不够，
> 需要 temporal/innovation health monitor，不能直接拿简单 RHSE 过门 1。**

## 1. LTAG v2 修复一：cap 感知年龄对齐（已跑，负结果）

按方案 §3.1 实现 `align_cap_aware(p, delta) = p - delta * sigmoid((125-3-p)/3)`，
在 FD001/FD002/FD004 s101 validation 上对比旧的 `p - delta`：

| 子集 | 版本 | K=1 | 固定 median | 手写 gate | cap 率 | median 均值 |
|---|---|---:|---:|---:|---:|---:|
| FD001 | old | 12.647 | **11.634** | **11.644** | 1.0% | 61.1 |
| FD001 | cap-aware | 12.647 | 13.345 | 12.026 | 1.0% | 65.7 |
| FD002 | old | 12.726 | **13.016** | **12.660** | 1.5% | 62.6 |
| FD002 | cap-aware | 12.726 | 14.444 | 13.699 | 1.5% | 67.8 |
| FD004 | old | 18.407 | **19.335** | **18.448** | 8.4% | 68.2 |
| FD004 | cap-aware | 18.407 | 19.445 | 18.592 | 8.4% | 72.4 |

**cap-aware 在所有三个子集上都更差**，而且把 median 均值推高 4–5 cycles。
cap 率只有 1–8%，不是 FD002/FD004 伤害的主因；旧 median 在这些子集上已经偏高，
不是方案假设的“偏低 5–10 cycles”。

**处理**：不采用 cap-aware 对齐；保留已冻结的 LTAG。方案 §3.1 的预注册检验 1 失败，
检验 2/3（门控增益是否缩小、g–p_k1 曲线）不再有前提。

## 2. RHSE 后验故障检测（不训练，只验证机制）

按方案 §2.2/§2.3 实现 `RHSE`（IRLS + Cauchy 权重）；`H` 用 training engines
最后 30% 的 Δy 做 PCA（m=3），`log_s=0`（Δy 先按训练 std 标准化）；在 validation
最后 cut（0.95）上注入单传感器故障，故障从有效历史的 50% 处开始，计算
`1-w` 的 AUC（故障周期 vs clean/故障前周期）。

| 故障（mag=3） | FD001 mean AUC | FD001 ≥0.85 | FD004 mean AUC | FD004 ≥0.85 |
|---|---:|---:|---:|---:|
| bias | **0.976** | 93.2% | **0.990** | 99.6% |
| noise | 0.849 | 56.4% | **0.866** | 61.4% |
| drift | 0.806 | 36.8% | 0.823 | 42.0% |
| stuck | **0.625** | 16.4% | **0.623** | 16.0% |

**读法**：

- **additive bias 很强**：AUC 0.98–0.99，远超方案的门 0.85。RHSE 的测量残差
  可靠性机制成立。
- **noise 边界**：0.85–0.87，FD004 过门、FD001 差一点；需要更强噪声模型。
- **drift 不够**：0.81–0.82；慢漂移每周残差小，Cauchy 权重不降。
- **stuck 失败**：0.62；卡死传感器残差小且恒定，IRLS 会把它当“低噪声好传感器”。
  这说明 **只看残差幅度不够**，需要 temporal innovation / variance health。

## 3. 结论与下一步

1. **cap-aware LTAG 关闭**；不要改已冻结的 LTAG。
2. **RHSE 作为 bias 检测器成立**；但不能直接写“通用传感器故障检测”。
3. 下一步（按优先级）：
   - **加 temporal health monitor**：对每个传感器计算窗口内的
     `std(dy_j)`、innovation 的方差、残差 CUSUM/slope；与期望噪声尺度比较，
     得到 `w_health`，与 IRLS 的 `w_robust` 相乘。目标：drift/stuck AUC ≥0.85。
   - **重跑 AUC**：bias/noise/drift/stuck × FD001/FD004；只有四类都过门，
     才进入 RHSE 端到端训练（A0/A0+aug/A1/A2）。
   - **stress_suite.py**：把注入协议固定下来，所有臂共用；A0+aug 作为强鲁棒基线。
   - **RHSE → 传感器 attention 偏置 / θ → HI 特征**：等 AUC 过门后再接骨干；
     否则先做后验可靠性诊断，避免又变成“冻结骨干 + 小头”。
4. 评估协议（5 折 CV、配对 seed、EMA、共形区间）暂不启动；先把 RHSE 机制做实。

**一句话**：Claude 方案的 RHSE 方向在 bias 上得到强支持，但简单 IRLS 对 drift/stuck
不够；先加 temporal health monitor 把四类故障的 AUC 拉到 0.85 以上，再谈端到端和论文主张。

---

## 4. Round-2：修正检测口径 + 不训练 RUL 鲁棒性 pilot（2026-11-19）

### 4.1 修正检测口径（`rhse_detect_eval.py`）

- 幅度分层 AUC（drift 只算注入幅度 ≥1σ 的周期）；
- clean 训练集 99% 分位阈值，报告 clean 寿命末 20% 的虚警率（FPR）。

| 子集 | clean EOL FPR | bias mag2 | bias mag3 | noise mag5 | drift mag2 | drift mag3 | stuck |
|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 1.81% | 0.934 | 0.976 | 0.905 | 0.862 | 0.914 | 0.625 |
| FD004 | 1.86% | 0.946 | 0.990 | 0.918 | 0.878 | 0.929 | 0.623 |

- EOL FPR 1.8–1.9%，过“≤3%”硬门。
- bias/drift 在 mag≥2 过 0.90/0.85；noise 只有 mag5 过；**stuck 仍 0.62**。
- 与首轮 raw AUC 相比，drift 的分层口径把 0.81/0.82 提升到 0.86/0.88（mag2）；
  说明首轮的 drift 低 AUC 有一部分是评估口径造成的。

### 4.2 不训练 RUL 鲁棒性 pilot（FD001 s101，bias mag=3，sensors 0/5/10，5 cuts）

`rhse_rul_eval.py`：冻结 B0；故障注入后比较五种输入：

| 臂 | 输入 | RMSE | 恢复率 |
|---|---|---:|---:|
| R0 | clean | **12.647** | — |
| R1 | faulted | 20.482 | — |
| R2 | oracle 重构（已知故障传感器，Hθ 替换） | **12.743** | **0.988** |
| R3 | RHSE 重构（w<0.5 判故障，w 重解 θ） | 18.064 | **0.309** |
| R4 | oracle last-value（已知故障传感器，最后健康值保持） | 13.306 | 0.916 |

**关键读法**：

1. **测量模型本身很好**：R2 用 Hθ 重构，恢复率 0.988，说明“故障传感器可被 Hθ 重构”
   这个前提成立。
2. **部署式 RHSE 很差**：R3 只有 0.309，远低于 R2，也低于 oracle last-value 的 0.916。
   瓶颈不在 Hθ，而在 **故障传感器选择/权重校准**：`w[-1] < 0.5` 这个硬阈值没有稳定
   选中故障传感器；`theta_solve` 里故障传感器仍以低权重参与。
3. **last-value 是强 baseline**：oracle last-value 恢复 0.916；任何 RHSE 重构必须先打败它。
   目前 R3 没有做到。
4. 这正好验证了 round-2 的判断：**检测 AUC 不是判定门，RUL 恢复率才是**；R2 好、R3 差，
   说明问题在检测/选择，不在测量模型。

### 4.3 下一步（按 round-2 §4.3 决策树）

- **先修 R3 的检测/选择**：
  - 在 clean training 上标定 `w` 的 1% 分位阈值（而不是硬 0.5）；
  - 被判故障的传感器权重置 0，再解一次 θ；
  - 用 `w` 的时序锁存（一旦判故障，窗口内保持低权重）。
- **实现 R5（deployable last-value）**：用同一套检测（w 阈值）选传感器，再做 last-value；
  与 R3 对比，回答“测量模型重构是否真的优于朴素填补”。
- 重跑 FD001/FD004 bias/drift/noise/stuck mag 2/3/5；只有 R3 或 R5 的恢复率 ≥50%
  且 clean 代价 ≤0.2，才进入端到端。
- **不要**现在接骨干/训练；R3 还没过 RUL 门。
