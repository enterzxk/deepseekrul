from pathlib import Path
for f in ['RESULTS_NARRATIVE.md','TIM_PAPER_PACKAGE.md','KEY_FINDINGS.md','PAPER_DRAFT.md','PAPER_RESULTS.md','FINAL_DECISION.md']:
    p=Path(f)
    if not p.is_file(): continue
    s=p.read_text(); n=s.count('14.582')
    if n:
        s=s.replace('14.582','14.590'); p.write_text(s); print(f,'replaced',n)
    else:
        print(f,'none')