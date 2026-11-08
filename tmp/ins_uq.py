from pathlib import Path
base = Path('/home/user/rulb/tim_20261105/full_20261105')
uq = Path('/tmp/uq_section.tex').read_text()
p = base / 'manuscript_v4.tex'
s = p.read_text()
anchor = '\\section{Discussion}'
assert s.count(anchor) == 1
s = s.replace(anchor, uq.strip() + chr(10) + chr(10) + anchor)
p.write_text(s)
print('inserted UQ section')