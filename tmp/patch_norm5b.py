from pathlib import Path
for f in ['RESULTS_NARRATIVE.md','KEY_FINDINGS.md','TIM_PAPER_PACKAGE.md','PAPER_DRAFT.md','PAPER_RESULTS.md']:
    p=Path(f)
    if not p.is_file(): continue
    s=p.read_text(); n=s.count('14.523')
    if n:
        s=s.replace('14.523','14.582'); p.write_text(s); print(f,'replaced',n)
    else:
        print(f,'none')