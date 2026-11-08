from pathlib import Path
for f in ['run_full_fold.py','run_full_fold2.py']:
    p=Path(f); s=p.read_text()
    if 'tables["NORM"] = tables["FULL"]' in s:
        print(f,'already patched'); continue
    old='    tables["M0"] = tables["FULL"]'
    if old not in s:
        print(f,'anchor missing'); continue
    s=s.replace(old, old+'\n    tables["NORM"] = tables["FULL"]')
    p.write_text(s); print(f,'patched')
