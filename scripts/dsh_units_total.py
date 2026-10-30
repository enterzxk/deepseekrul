import pandas as pd
from pathlib import Path
ROOT = Path('data/CMAPSS')
tot = 0
print('subset  split   engines  status')
for sub in ('FD001','FD002','FD003','FD004'):
    for split in ('train','test'):
        f = ROOT / f'{split}_{sub}.txt'
        try:
            d = pd.read_csv(f, sep=r'\s+', header=None)
            n = d[0].nunique()
            tot += n
            print(f'{sub}    {split:5s}  {n:6d}  readable')
        except PermissionError:
            print(f'{sub}    {split:5s}       -  SEALED (permission denied)')
        except FileNotFoundError:
            print(f'{sub}    {split:5s}       -  missing')
print()
print('readable trajectories total:', tot)
print('needed for 5% (paired estimate): ~1590')
