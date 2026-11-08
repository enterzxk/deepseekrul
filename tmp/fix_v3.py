from pathlib import Path
p=Path('run_v3.py'); s=p.read_text()
s=s.replace('def train(arm, net,', 'def train_v3(arm, net,')
s=s.replace('        w = train(arm, net,', '        w = train_v3(arm, net,')
p.write_text(s); print('renamed')