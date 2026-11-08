from pathlib import Path
p=Path('run_v3b.py'); s=p.read_text()
old='life_tr, life_va, a.target_mode, a.mixup, a.mixup_mode)'
new='life_tr, life_va, a.target_mode, a.mixup, a.mixup_mode, a.select)'
assert old in s, 'call anchor'
s=s.replace(old,new,1)
p.write_text(s); print('wired')