from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
fixes = 0
for i, s in enumerate(L):
    if s.startswith('Let \\mathbf{x}') and not s.startswith('$'):
        L[i] = '$' + s; fixes += 1
p.write_text('\n'.join(L))
print('fixed', fixes)
print(repr(L[112])[:110])