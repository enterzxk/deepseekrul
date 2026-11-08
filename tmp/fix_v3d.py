from pathlib import Path
p=Path('run_v3.py'); s=p.read_text()
old='def train_v3(arm, net, tr_raw, va_raw, target, meta, seed, epochs=80, l1=0.0, l2=0.0, l3=0.0, tau=0.9):'
new='def train_v3(arm, net, tr_raw, va_raw, target, meta_tr, meta_va, seed, epochs=80, l1=0.0, l2=0.0, l3=0.0, tau=0.9):'
assert old in s, 'anchor missing'
s=s.replace(old,new)
p.write_text(s)
print('patched:', new in s)