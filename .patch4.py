from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
start = s.find('    if getattr(args, "cp_labels", False):')
end = s.find(')', s.find('src.data before enabling this flag.'))
if start == -1:
    print('anchor missing'); raise SystemExit
block_end = s.find('        )', start) + len('        )')
new = '    if getattr(args, "cp_labels", False):' + chr(10) + '        import dataset as _ds' + chr(10) + '        _ds.CP_LABELS = True'
s = s[:start] + new + s[block_end:]
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('run_candidate patched + syntax ok')