from pathlib import Path
for f in ['PAPER_DRAFT.md','PAPER_RESULTS.md']:
    p=Path(f); s=p.read_text()
    s=s.replace('+1.320 [+0.783, +1.879]','+1.535 [+0.989, +2.116]')
    s=s.replace('| NORM | FD004 | 4 | +1.320','| NORM | FD004 | 5 | +1.535')
    p.write_text(s)
print('patched')
