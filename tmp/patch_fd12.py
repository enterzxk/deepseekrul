from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
s=s.replace('| 3 种子 | FD002 seed 101/202 满 5 折（−3.38% / −3.13%），303 为 3 折（−3.73%）；FD001/FD003/FD004 仅 seed 101 | **PARTIAL（接近完成）** |', '| 3 种子 | **FD002 M0/FULL 三种子满 5 折**（−3.38% / −3.13% / −3.59%，均值 −3.37%、sd 0.23%）；NORM 2 种子；FD001/FD003/FD004 仅 seed 101 | **PARTIAL（FD002 已满足）** |')
s=s.replace('| FD002 | 4 | 14.523 | 13.054 | 显著更差（+1.034 [+0.595,+1.468] vs M0） |', '| FD002 | 2 种子 | 14.65 | 13.19 | 显著更差（+1.536 vs FULL，gain −11.8%） |')
p.write_text(s); print('patched')