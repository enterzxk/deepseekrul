import glob, os, re
hits = []
for f in glob.glob('scripts/*.py') + glob.glob('src/**/*.py', recursive=True):
    try: t = open(f, encoding='utf-8', errors='ignore').read()
    except Exception: continue
    for i, line in enumerate(t.splitlines(), 1):
        if re.search(r'test_FD00[1-4]|RUL_FD00[1-4]|/test/', line):
            hits.append((f, i, line.strip()[:110]))
print('script lines referencing official test/RUL files:', len(hits))
for f, i, l in hits[:25]: print(f'  {f}:{i}  {l}')
