from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
a=s.count('0.79--1.11 cycles'); b=s.count('3.89 percent on FD004')
s=s.replace('0.79--1.11 cycles','1.07--1.52 cycles')
s=s.replace('3.89 percent on FD004','9.44 percent on FD004')
p.write_text(s); print('fixed',a,b)