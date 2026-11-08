from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
anchor='\\subsection{Observation span as a first-order variable}'
assert anchor in s
figs=open('figs_block.tex').read()
s=s.replace(anchor, figs+anchor, 1)
p.write_text(s); print('figures restored:', s.count('includegraphics'))