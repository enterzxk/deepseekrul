from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
old='On its own, mixup brings the plain backbone to parity with the prior arm on FD002 (+0.3 percent) and ahead of it on FD004 (1.7 percent), with the asymmetric score falling from 11.25 to 7.00.'
new=('On its own, mixup brings the plain backbone to parity with the prior arm on FD002 (+0.3 percent) and ahead of it on FD004 (1.7 percent) for the first seed; for the second seed it is worth 3.1 and 6.0 percent, with the asymmetric score on FD004 falling from 8.63 to 6.65. Averaged over the two seeds the gain is 1.4 percent on FD002 and 3.8 percent on FD004. The gain is larger for the seed whose initialisation is worse, which is what one expects from a regulariser that reduces variance.')
assert old in s, 'anchor'
s=s.replace(old,new)
p.write_text(s); print('patched')