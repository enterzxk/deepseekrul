from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
old=('Relative to the locked reference this reduces the pooled RMSE by 6.64% on FD002\n'
     '(95% CI 0.65-1.09 cycles) and 4.10% on FD004, improves all four metrics on both fleets, and improves')
new=('Relative to the locked reference this reduces the pooled RMSE by 7.20% on FD002\n'
     '(95% CI 0.79-1.11 cycles) and 3.89% on FD004 across two seeds, improves all four metrics on both fleets, and improves')
assert old in s, 'anchor'
s=s.replace(old,new)
s=s.replace('5.37% with a lower bound of 0.53 cycles','5.53% with a lower bound of 0.59 cycles')
p.write_text(s); print('ok')