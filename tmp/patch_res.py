from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
start=s.find('\\section{Results}\\label{sec:results}')
end=s.find('\\section{Discussion}\\label{sec:disc}')
assert start!=-1 and end!=-1 and end>start
new=open('results_block.tex').read()
s=s[:start]+new+'\n'+s[end:]
p.write_text(s); print('replaced')