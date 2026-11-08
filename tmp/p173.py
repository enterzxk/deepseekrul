from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
L[172] = r'C-MAPSS FD001--FD004 \cite{saxena2008}. Five outer folds $\times$ three seeds'
# general: bare \times / \pm outside math on text lines
import re
n = 0
for i, s in enumerate(L):
    if i == 172: continue
    if '\\times' in s and '$' not in s and not s.strip().startswith('\\'):
        print('bare times at', i+1, repr(s)[:90]); n += 1
p.write_text('\n'.join(L))
print('bare-times lines:', n)