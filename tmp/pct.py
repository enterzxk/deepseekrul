from pathlib import Path
import re
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
s = p.read_text()
bs = chr(92)
before = s.count('%')
s2 = re.sub(r'(?<!' + re.escape(bs) + r')%', bs + '%', s)
p.write_text(s2)
print('percent signs:', before, '-> escaped:', s2.count(bs + '%'))
for i, l in enumerate(s2.split('\n')):
    if bs + '%' in l and len(l) > 0:
        pass
print('sample:', [l[:60] for l in s2.split('\n') if '21' in l][:2])