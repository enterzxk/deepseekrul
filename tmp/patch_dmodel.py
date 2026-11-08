from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
s=s.replace('return ida.PriorWrapper(body, "M1")', 'return ida.PriorWrapper(body, "M1", d_model=d_model)')
s=s.replace('return ida.PriorWrapper(body, "M0", norm=True)', 'return ida.PriorWrapper(body, "M0", d_model=d_model, norm=True)')
p.write_text(s); print('patched')