import numpy as np
mde260 = 1.60; base = 13.5
strata = {
  'FD002+FD004 (multi-condition)': 260+249,
  'FD001+FD003 (single-condition)': 100+100,
  'all four (mixed)': 709,
}
print('stratum                            units   MDE     relative   needs')
for name, n in strata.items():
    mde = mde260*np.sqrt(260.0/n)
    need = 'no conditioning change' if 'FD002+FD004' in name or 'FD001+FD003' in name else 'conditioning design decision'
    print(f'{name:34s} {n:5d}  {mde:5.2f}  {100*mde/base:6.1f}%   {need}')
print()
print('5% target needs', int(260*(mde260/(0.05*base))**2), 'units')
