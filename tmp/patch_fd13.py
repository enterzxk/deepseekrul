from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
s=s.replace('NORM 2 种子；FD001/FD003/FD004 仅 seed 101', 'NORM 3 种子（14.59 sd 0.11）；FD001/FD003/FD004 仅 seed 101')
s=s.replace('| FD002 | 2 种子 | 14.65 | 13.19 | 显著更差（+1.536 vs FULL，gain −11.8%） |', '| FD002 | 3 种子 | 14.59 | 13.19 | 显著更差（+1.536 vs FULL，gain −11.8%，sd 0.11） |')
p.write_text(s); print('patched')