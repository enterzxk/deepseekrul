from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
good = 'Let $\\mathbf{x}\\in\\mathbb{R}^{L\\times 14}$ and let $\\mathbf{b}$ be the first-cycle'
n = 0
for i, s in enumerate(L):
    if 'mathbf{x}' in s and 'first-cycle' in s:
        L[i] = good; n += 1
p.write_text('\n'.join(L))
print('replaced', n)
print(L[[i for i,s in enumerate(L) if 'first-cycle' in s][0]])