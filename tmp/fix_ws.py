from pathlib import Path
p = Path('/home/user/rulb/scripts/run_dspt.py')
s = p.read_text()
old = 'lens = np.array([len(g) for _, g in frame.groupby("unit")], float)'
new = 'lens = np.array([len(g) for _, g in tr.groupby("unit")], float)'
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
import ast; ast.parse(s)
print('fixed')