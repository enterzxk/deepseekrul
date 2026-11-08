from pathlib import Path
p = Path('/home/user/rulb/scripts/run_dspt.py')
s = p.read_text()
old = 'Yr = np.round(Y).astype(int)'
new = 'Yr = np.round(np.asarray(TR[1], np.float64)).astype(int)'
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
import ast; ast.parse(s)
print('fixed Y')