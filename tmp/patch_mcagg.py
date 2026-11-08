from pathlib import Path
p=Path('mc_aggregate.py'); s=p.read_text()
if 'available folds' in s:
    print('already'); raise SystemExit
s=s.replace('''def load(subset, arm, seed, cond):
    blocks = []
    for k in range(5):
        p = HERE / "measurement" / subset / str(k) / f"{arm}_{seed}_{cond}.csv"
        if not p.is_file():
            return None
        blocks.append(pd.read_csv(p))
    d = pd.concat(blocks, ignore_index=True)''','''def load(subset, arm, seed, cond):
    blocks = []
    for k in range(5):   # available folds only
        p = HERE / "measurement" / subset / str(k) / f"{arm}_{seed}_{cond}.csv"
        if not p.is_file():
            continue
        blocks.append(pd.read_csv(p).assign(fold=k))
    if not blocks:
        return None
    d = pd.concat(blocks, ignore_index=True)''')
s=s.replace("""            rows.append(dict(subset=subset, arm=arm, condition=cond, **stats(d)))""",
"""            rows.append(dict(subset=subset, arm=arm, condition=cond, folds=int(d.fold.nunique()), **stats(d)))""")
p.write_text(s); print('patched')
