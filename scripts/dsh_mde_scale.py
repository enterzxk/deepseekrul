import numpy as np
mde260 = 1.60   # from the paired-difference estimate
base_rmse = 13.5
print(' n_units   MDE(RMSE)   relative')
for n, label in ((260,'FD002 train only (current)'),
                 (709,'all four subsets, TRAIN only'),
                 (1068,'plus readable FD001/FD002 test')):
    mde = mde260 * np.sqrt(260.0/n)
    print(f' {n:6d}   {mde:8.2f}   {100*mde/base_rmse:5.1f}%   {label}')
print()
target = 0.05*base_rmse
print(f'5% target = {target:.2f} RMSE')
for n in (709, 1068):
    mde = mde260*np.sqrt(260.0/n)
    print(f'  at n={n}: MDE {mde:.2f} -> target {"reachable" if target>=mde else "NOT reachable"} (gap {mde-target:+.2f})')
print()
print('units needed for MDE <= 0.65:', int(260*(mde260/target)**2))
