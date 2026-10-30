"""What would engine-level k-fold CV buy in power? (costed design)"""
import numpy as np
# empirical split-level SD from the census (per cell)
sd = {'M4_racdr': 3.76, 'M1_global': 2.15, 'M0_raw_tf': 1.51, 'M6_tfA': 1.07}
med = float(np.median(list(sd.values())))
print(f'median split SD = {med:.2f}')
print()
print(' folds  SE=SD/sqrt(k)   detectable (80% power)   cost (configs x seeds)')
for k in (3, 5, 8, 10):
    se = med / np.sqrt(k)
    mde = (1.96 + 0.84) * se
    print(f' {k:5d}  {se:12.2f}   {mde:22.2f}   {k}x3 = {k*3} units')
print()
print('note: with 260 engines, k-fold CV over engines gives k = 260/fold_size')
for fold in (52, 26, 20):
    print(f'  fold={fold}: k={260//fold} -> SE={med/np.sqrt(260//fold):.2f}, MDE={(1.96+0.84)*med/np.sqrt(260//fold):.2f}')
