from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
old='Combined with the span schedule and the guard, the same model reaches 11.998 on FD002 and 12.170 on FD004, against 13.054 and 13.253 for the prior arm: 8.09 and 8.18 percent, with the stratified interval at 8.13 percent and bounds [1.366, 0.768] cycles. All four metrics improve on both fleets, and the FD004 asymmetric score falls from 11.25 to 7.38.'
new='Combined with the span schedule and the guard, the same model reaches 11.871 on FD002 and 12.173 on FD004 across two seeds, against 13.190 and 13.442 for the prior arm: 10.00 and 9.44 percent, with the stratified interval at 9.72 percent and bounds [1.524, 1.069] cycles. All four metrics improve on both fleets, and the FD004 asymmetric score falls from 9.94 to 7.40.'
assert old in s, 'anchor1'
s=s.replace(old,new)
old2='Relative to the locked reference this reduces the pooled RMSE by 7.20 percent on FD002'
new2='Relative to the locked reference this reduces the pooled RMSE by 10.00 percent on FD002'
assert old2 in s, 'anchor2'
s=s.replace(old2,new2)
s=s.replace('(95% CI 0.79--1.11 cycles) and 3.89 percent on FD004 across two seeds','(95% CI 1.07--1.52 cycles) and 9.44 percent on FD004 across two seeds')
s=s.replace('5.53 percent with a lower bound of 0.59 cycles','9.72 percent with a lower bound of 1.07 cycles')
p.write_text(s); print('patched')