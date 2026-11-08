from pathlib import Path
import re
p=Path('manuscript.tex'); s=p.read_text()
s=s.replace('d_model', r'$d_{\mathrm{model}}$')
s=s.replace(r'1.8\,e, & e>0\\ e, & e\le 0,', r'1.8\,e^2, & e>0\\ e^2, & e\le 0,')
p.write_text(s)
print('underscores left:', [l[:60] for l in s.splitlines() if '_' in l and '$' not in l and '\\' not in l][:5])
print('eq:', [l for l in s.splitlines() if 'asym' in l][:2])