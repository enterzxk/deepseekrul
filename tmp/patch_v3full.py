from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
if '"FULL"' in s and 'PriorWrapper(body, "M1")' in s:
    print('already'); raise SystemExit
s=s.replace('''    if arm in ("M0", "DIFF", "PHASE", "GRAPH"):
        return ida.PriorWrapper(body, "M0")''',
'''    if arm in ("M0", "DIFF", "PHASE", "GRAPH"):
        return ida.PriorWrapper(body, "M0")
    if arm in ("OPS", "FULL"):
        return ida.PriorWrapper(body, "M1")
    if arm == "NORM":
        return ida.PriorWrapper(body, "M0", norm=True)''')
p.write_text(s); print('patched')
