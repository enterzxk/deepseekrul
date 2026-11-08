from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = "### A.6 判定"
new = """### A.6 逐条对照 PROTOCOL §6 的门槛

| 门槛条款 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 主目标 | (G_FD002+G_FD004)/2 ≥ 3%，且两者 >0 | −11.3% / −11.6%，均值 −11.45% | **FAIL** |
| 种子一致性 | 两主子集各至少 2/3 种子方向一致 | NORM 在 seed 101/202 均更差 | **FAIL** |
| 区间 | 分层配对 bootstrap 95% 下界 >0 | NORM 区间 [+0.595,+1.468] / [+0.989,+2.116]，全为负增益 | **FAIL** |
| 四指标 | MAE 与 NASA 均不劣于参考（0.5% 容差） | NORM 在 FD002/FD004 的 MAE、NASA 均更差 | **FAIL** |
| 单工况护栏 | FD001/FD003 退化 ≤ +1.0%/+1.0%/+2.0% | FD001 −0.38/−0.70/−4.34%；FD003 −6.49/−5.63/−13.58% | **PASS** |
| 机制归因 | NORM 须优于 NORM_SHUF 与 NORM_CONST | NORM 连 M0 都不如（+1.034/+1.535） | **FAIL** |
| 预算一致性 | 若仅轻量有效、完整无效则判 FAIL | 轻量 −4.67%（3 种子），完整 +7.7% | **FAIL（按条款）** |

### A.7 判定"""
assert old in s
s = s.replace(old, new)
p.write_text(s); print('patched')
