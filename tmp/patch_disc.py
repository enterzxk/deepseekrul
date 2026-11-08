from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
anchor='\\section{Conclusion}\\label{sec:concl}'
assert anchor in s
s=s.replace(anchor, open('disc_add.tex').read()+anchor, 1)
p.write_text(s); print('ok')