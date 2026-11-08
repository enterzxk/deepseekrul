from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
import re
bad=re.findall(r'\\\\[a-zA-Z]+', s)
print('double-backslash commands:', sorted(set(bad))[:10], len(bad))
s=s.replace('\\\\begin{abstract}','\\begin{abstract}').replace('\\\\end{abstract}','\\end{abstract}')
p.write_text(s)
print('fixed')