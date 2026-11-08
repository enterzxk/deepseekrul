from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
if 'use_graph=graph' in s:
    print('already'); raise SystemExit
i=s.index('def build_net')
j=s.index('use_graph=False', i)
s=s[:j]+'use_graph=graph'+s[j+len('use_graph=False'):]
p.write_text(s); print('patched')
