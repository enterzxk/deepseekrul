from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = "### A.5 判定"
new = """### A.5 FD001/FD003 适用范围（协议允许退化 ≤ +1.0% RMSE / +1.0% MAE / +2.0% NASA）

| 子集 | M0 | FULL | 相对变化 | 判定 |
|---|---|---|---|---|
| FD001 | 见 scope_check.py | 见 scope_check.py | 见 scope_check.py | 待补 |
| FD003 | 见 scope_check.py | 见 scope_check.py | 见 scope_check.py | 待补 |

（运行完成后由 \`scope_check.py\` 填入。）

### A.6 判定"""
if old in s and 'A.5 FD001/FD003' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
