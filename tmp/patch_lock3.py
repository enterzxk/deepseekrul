from pathlib import Path
p=Path('lock_reference.py'); s=p.read_text()
s=s.replace('if len(vals) < 3:', 'if len(vals) < 2:')
p.write_text(s); print('patched')
