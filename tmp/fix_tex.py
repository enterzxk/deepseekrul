from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
reps=[('6.64 percent','7.20 percent'),('4.10 percent','3.89 percent'),
      ('5.37 percent','5.53 percent'),('0.53 cycles','0.59 cycles'),
      ('0.65--1.09 cycles','0.79--1.11 cycles')]
for a,b in reps:
    n=s.count(a); s=s.replace(a,b); print(a,'->',b,n)
p.write_text(s)