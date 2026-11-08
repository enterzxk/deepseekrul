from pathlib import Path
p=Path('eval_matrix.py'); s=p.read_text()
s=s.replace("        if len(parts) < 5: continue", "        if len(parts) < 4: continue")
p.write_text(s); print('fixed')