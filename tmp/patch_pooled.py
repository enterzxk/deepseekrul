from pathlib import Path
for f in ['KEY_FINDINGS.md','TIM_PAPER_PACKAGE.md','PAPER_DRAFT.md','PAPER_RESULTS.md','SESSION_REPORT.md']:
    p=Path(f)
    if not p.is_file(): continue
    s=p.read_text(); n=s.count('14.513')
    if n:
        s=s.replace('14.513','14.523')
        p.write_text(s); print(f,'replaced',n)
    else:
        print(f,'none')
