import re
for f in ('src/dataset.py','scripts/run_candidate.py'):
    t = open(f, encoding='utf-8', errors='ignore').read()
    print('==', f, len(t))
    for m in re.finditer(r'^(def |class ).*', t, re.M):
        print('  ', m.group(0)[:90])
    for kw in ('window','cycle'):
        hits = [m.start() for m in re.finditer(kw, t)][:4]
        print(f'   {kw}: {len(hits)} hits')
