from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
anchor='Two baselines are reported throughout.'
add=('We make no correction for multiplicity across the arms and fleets. The headline comparison is a single contrast specified before the experiments, and its interval lies far from zero, so a Bonferroni-style correction would not change the reading; the smaller differences in Table~\\ref{tab:main} should be read as descriptive. '
     'The single-fold screens in Table~\\ref{tab:neg} are not tested at all and are reported only to show direction.\n\n')
assert anchor in s
s=s.replace(anchor, add+anchor, 1)
p.write_text(s); print('ok')