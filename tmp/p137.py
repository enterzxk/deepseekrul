from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
for i in range(135, 142): print(i+1, repr(L[i])[:100])
new = [
  r'Let $\mathcal{V}=\{i:m_i=1\}$ and let',
  r'$\mathcal{S}=\{i\in\mathcal{V}:\,L-1-i<50\}$ be the recent 50 valid steps. With',
  r'$\bar{\mathbf{h}}(\mathcal{A})=\frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}}\mathbf{u}_i$,',
]
L[136:139] = new
p.write_text('\n'.join(L))
print('--- after ---')
for i in range(135, 142): print(i+1, repr(L[i])[:100])