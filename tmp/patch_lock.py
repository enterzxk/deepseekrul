from pathlib import Path
p=Path('lock_reference.py'); s=p.read_text()
s=s.replace('SUBSETS = ["FD002", "FD004"]', 'SUBSETS = ["FD001", "FD002", "FD003", "FD004"]')
p.write_text(s)
q=Path('make_spec.py'); t=q.read_text()
t=t.replace('subsets = list(lock["subsets"])', 'subsets = [s for s in ["FD001", "FD002", "FD003", "FD004"] if s in lock["subsets"]]')
q.write_text(t)
print('patched')
