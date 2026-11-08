from pathlib import Path
base = Path('/home/user/rulb/tim_20261105/full_20261105')
sec = Path('/tmp/span_section.tex').read_text().strip()
p = base / 'manuscript_v4.tex'
s = p.read_text()
if 'span optimum sits' in s:
    print('already'); raise SystemExit
anchor = '\\subsection{Instance normalisation as an ensemble-diversity device}'
assert s.count(anchor) == 1
s = s.replace(anchor, sec + chr(10) + chr(10) + anchor)
p.write_text(s)
print('inserted')