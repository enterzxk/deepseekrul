from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = """| 子集 | M0 | FULL | 相对变化 | 判定 |
|---|---|---|---|---|
| FD001 | 见 scope_check.py | 见 scope_check.py | 见 scope_check.py | 待补 |
| FD003 | 见 scope_check.py | 见 scope_check.py | 见 scope_check.py | 待补 |

（运行完成后由 \`scope_check.py\` 填入。）"""
new = """| 子集 | 臂 | RMSE | MAE | NASA | Bias | 相对 M0 |
|---|---|---:|---:|---:|---:|---|
| FD001（100 台） | M0 | 10.274 | 7.240 | 1.86 | +0.80 | — |
| FD001 | FULL | 10.235 | 7.190 | 1.78 | −0.33 | RMSE −0.38% / MAE −0.70% / NASA −4.34% |
| FD003 | 待补 | | | | | |

FD001 判定：**PASS**（三项均在允许退化范围内，实际略优）。
逐折方向不一致：f0 −0.6%、f1 −3.6%、f2 −1.8%、f3 **+3.2%**、f4 **+2.2%**，
合并后为 −0.38%——护栏结论只能作为合并结论。"""
assert old in s
s = s.replace(old, new)
p.write_text(s); print('patched')
