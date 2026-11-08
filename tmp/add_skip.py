from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
old="    set_seed(a.seed)"
new=('    set_seed(a.seed)\n'
     '    _out = ROOT/\'experiments\'/a.tag/a.subset\n'
     '    if (_out/f\'dspt_s{a.seed}.csv\').exists():\n'
     '        print(\'SKIP %s (exists)\' % a.tag, flush=True); return')
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('added skip')