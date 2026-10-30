from pathlib import Path
p = Path('src/dataset.py')
s = p.read_text(encoding='utf-8')
start = s.find('    if LABEL_MODE == \'cp\':')
end = s.find('    return np.minimum(rul, float(max_rul)).astype(np.float32)', start)
if start == -1 or end == -1:
    print('pattern not found; nothing to revert')
else:
    s = s[:start] + s[end:]
    s = s.replace('LABEL_MODE = "cap"' + chr(10)*2 + chr(10), '')
    p.write_text(s, encoding='utf-8')
    print('reverted cp block')