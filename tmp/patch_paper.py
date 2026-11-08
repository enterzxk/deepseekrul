from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
old = "A per-window oracle over\nthe variants would reduce the error by a further 15%, but none of six observable families we\ntested predicts which variant is better for a given observation (|rho| < 0.08)."
new = ("A per-window oracle over\nthe variants would reduce the error by a further 15%, but none of six observable families we\ntested predicts which variant is better for a given observation (|rho| < 0.08). Under a\nsimulated 20% random missing-sample condition the additive prior widens its advantage over\nthe plain backbone from 5.3% to 7.3%, while a one-standard-deviation offset on three channels\ndegrades both estimators by more than a factor of two.")
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor not found')
