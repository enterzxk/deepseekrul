# LTAG v3 Phase 4：RHSE-TSH 去留决定

> 日期：2026-11-20。依据：`ltag_v3_execution_plan.md` §6；36 runs
> （FD001/FD003 × bias/drift/stuck × onset 0.2/0.5 × seeds 101/202/303，mag=3）。
> **test 封存。**

## 1. 公平朴素基线

- `R5b`（回溯保持）：保持 `t_flag − 15 − 1` 时刻的观测 `x_f`；
- `R5m`（回溯中位数）：取 `[t_flag−25, t_flag−16]` 区间的观测中位数；
- 两者都只用观测 `x_f`，不用 clean 反事实。
- `R3` = TSH + Hθ 重构。

## 2. 结果（3-seed mean recovery）

| subset | fault | onset | R3 | R5 | R5b | R5m | R3−max(R5b,R5m) |
|---|---|---:|---:|---:|---:|---:|---:|
| FD001 | bias | 0.2 | 0.799 | −0.009 | 0.793 | 0.821 | **−0.022** |
| FD001 | bias | 0.5 | 0.826 | 0.078 | 0.865 | 0.906 | **−0.080** |
| FD001 | drift | 0.2 | 0.856 | 0.669 | 0.868 | 0.925 | **−0.069** |
| FD001 | drift | 0.5 | 0.797 | 0.362 | 0.903 | 0.934 | **−0.137** |
| FD001 | stuck | 0.2 | 0.737 | −0.105 | −0.695 | −0.202 | **+0.940** |
| FD001 | stuck | 0.5 | 0.872 | −0.037 | 0.257 | 0.403 | **+0.468** |
| FD003 | bias | 0.2 | 0.908 | 0.190 | 0.927 | 0.939 | **−0.032** |
| FD003 | bias | 0.5 | 0.876 | 0.153 | 0.890 | 0.910 | **−0.034** |
| FD003 | drift | 0.2 | 0.916 | 0.653 | 0.853 | 0.921 | **−0.005** |
| FD003 | drift | 0.5 | 0.772 | 0.494 | 0.760 | 0.797 | **−0.025** |
| FD003 | stuck | 0.2 | 0.605 | −0.280 | −0.179 | −0.288 | **+0.783** |
| FD003 | stuck | 0.5 | 0.570 | −0.244 | −0.124 | −0.077 | **+0.647** |

## 3. 判定门

计划 §6 的门：

> `R3 − max(R5b, R5m) ≥ 0.10`，并且至少在 FD001 和 FD003 的 **2/3 种故障**上成立
> → 保留 RHSE-TSH 为附录；否则删除。

逐故障：

| fault | FD001 通过？ | FD003 通过？ |
|---|---|---|
| bias | 否（−0.022/−0.080） | 否（−0.032/−0.034） |
| drift | 否（−0.069/−0.137） | 否（−0.005/−0.025） |
| **stuck** | **是（+0.940/+0.468）** | **是（+0.783/+0.647）** |

- 只有 **stuck** 通过；bias/drift 全部不通过（R5m 甚至更好）。
- **2/3 故障不成立** → **RHSE-TSH 不保留**。

## 4. 决定

**从论文中删除 RHSE-TSH。** 只在 Limitations 保留一句：

> “On FD001/FD003, a fair backtracked hold/median baseline matches or exceeds the
> measurement-model reconstruction on bias and drift; the reconstruction only helps
> on stuck, so we do not claim it as a contribution.”

- 之前所有 RHSE-TSH 的“条件性前端”主张撤回。
- R5（旧 last-value）是稻草人；R5b/R5m 才是公平基线。
- 这一决定与 Phase 1 的 Gate 1 失败一致：论文只剩 LTAG / 后校准一条线。

## 5. 后果

1. **论文主方法**只能是 LTAG 或 Phase 1 的 LS-ctx / K1-Calib；RHSE 不再出现。
2. Phase 2（跨骨干 RHSE）不启动；Phase 3 的 RHSE 机制分析取消。
3. Phase 5 的主表不再包含 RHSE 附录；Limitations 保留一句。
4. 需要 L2/期刊决策：LTAG 机制已被 Phase 1 否定；LS-ctx 是后校准，创新性弱。
5. test 封存；本决定只基于 validation。
