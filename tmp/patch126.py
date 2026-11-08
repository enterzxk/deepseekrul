import re
from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
L[125] = r'absolute position. A learnable table $\mathbf{P}\in\mathbb{R}^{d\times L}$ is'
L[126] = r'added as $\mathbf{u}_t+\mathbf{P}_{:,L-1-t}$. This makes the representation'
p.write_text('\n'.join(L))
# scan for math commands outside $...$ on plain text lines
bad = []
ineq = False
for i, s in enumerate(L):
    t = s.strip()
    if t.startswith('\\begin{equation'): ineq = True; continue
    if t.startswith('\\end{equation'): ineq = False; continue
    if ineq or t.startswith('\\') or not t: continue
    stripped = re.sub(r'\\$[^$]*\\$', '', s)
    if re.search(r'\\(mathbf|mathbb|mathcal|mathrm|frac|sum|hat)', stripped):
        bad.append((i+1, s[:90]))
print('suspicious lines:', len(bad))
for b in bad[:12]: print(b)