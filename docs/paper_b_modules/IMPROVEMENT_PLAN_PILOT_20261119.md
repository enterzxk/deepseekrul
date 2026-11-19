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
