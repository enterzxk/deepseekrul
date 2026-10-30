import numpy as np
# SD of the paired difference in RMSE units, inferred from the CI
se = (0.623 - (-2.206)) / (2*1.96)
sd = se*np.sqrt(52)
print(f'implied paired-diff SD (RMSE units) = {sd:.2f}')
print()
print(' MDE for this specific pair by engine count:')
for n in (52, 260, 509, 709, 1460):
    mde = 2.8*sd/np.sqrt(n)
    print(f'  n={n:5d}: MDE={mde:5.2f} RMSE  -> effect 0.63 is {"detectable" if 0.63>mde else "NOT detectable"}')
