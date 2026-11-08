from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
s = p.read_text()
old = r'encoder to $d=96$ with three layers loses 6.2\% on two FD001 folds.'
new = r'encoder to $d=96$ with three layers loses 4.8\% on every one of the five FD001 folds.'
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))
print('fixed')