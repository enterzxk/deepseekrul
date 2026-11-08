from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
anchor='\\subsection{What does not work}'
assert anchor in s
s=s.replace(anchor, open('scope_add.tex').read()+anchor, 1)
p.write_text(s); print('ok')