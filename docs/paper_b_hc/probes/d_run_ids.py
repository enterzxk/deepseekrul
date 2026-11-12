"""Derive the canonical 48 Stage D run ids from the dispatcher's command list.

The ids must come from the same list the launcher consumes, not from a second
hand-typed table: a registry entry that does not match the directory the run
writes is exactly the kind of drift that makes a ledger unusable later.
"""
import re
from pathlib import Path

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/d')
cmds = (ROOT / 'matrix_commands.txt').read_text(encoding='utf-8').splitlines()
ids = []
for ln in cmds:
    if '--subset' not in ln:
        continue
    d = dict(re.findall(r'--(\w+) (\S+)', ln))
    ids.append('%s_%s_s%s' % (d['arm'], d['subset'], d['seed']))
out = ROOT / 'matrix_run_ids.txt'
out.write_text('\n'.join(ids) + '\n', encoding='utf-8')
dup = sorted({i for i in ids if ids.count(i) > 1})
print('commands with a cell: %d' % len(ids))
print('unique ids: %d   duplicates: %s' % (len(set(ids)), dup or 'none'))
print('first 3: %s' % ids[:3])
print('last 3:  %s' % ids[-3:])
print('record -> %s' % out)
