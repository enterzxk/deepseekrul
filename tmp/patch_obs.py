from pathlib import Path
p=Path('PAPER_RESULTS.md'); s=p.read_text()
old = '## D. Identifiability of adaptive correction (FD002, 4 folds)'
new = ('The curve is computed on the two multi-condition fleets, where the operating-condition\n'
       'normalisation is the object of study. For the single-condition fleets the same prior\n'
       'reaches a rank correlation of 0.65 (FD001) and 0.68 (FD003) at a 100-cycle cutoff and\n'
       '0.92 and 0.90 at 150 cycles, with MAE 25.9/47.0 and 13.4/25.6 cycles respectively.\n\n'
       '## D. Identifiability of adaptive correction (FD002, 4 folds)')
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor missing')