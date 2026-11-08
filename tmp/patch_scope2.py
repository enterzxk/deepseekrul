from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
a=s.find('\\subsection{Scope}')
b=s.find('\\subsection{What does not work}')
assert a!=-1 and b!=-1 and b>a
s=s[:a]+open('scope2.tex').read()+s[b:]
p.write_text(s); print('ok')