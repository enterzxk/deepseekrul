from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
L[185] = r'over 15 models (5 folds $\times$ 3 seeds); Ensemble averages their predictions. MAE,'
p.write_text('\n'.join(L))
# global sweep: any line with \times not surrounded by $
import re
for i, s in enumerate(L):
    for m in re.finditer(r'\\\\times', s):
        j = m.start()
        if j == 0 or s[j-1] != '$':
            print('check', i+1, repr(s)[:95])