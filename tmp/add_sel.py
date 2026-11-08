from pathlib import Path
p=Path('run_v3b.py'); s=p.read_text()
old="    ap.add_argument('--mixup-mode', default='all')"
new=old+"\n    ap.add_argument('--select', default='final', choices=['final','best'])"
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('added --select')