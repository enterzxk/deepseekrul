from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
old = 'An additive lifetime\nprior improves pooled RMSE by 3.3% on FD002 (95% CI 0.07-0.82 cycles) and 6.5% on FD003,'
new = 'An additive lifetime\nprior improves pooled RMSE by 3.3% on FD002 (95% CI 0.07-0.82 cycles; three-seed mean\n3.4%, sd 0.3%) and 6.5% on FD003,'
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor missing')