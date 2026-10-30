"""How many validation engines would the 5% target need? (power arithmetic)"""
import numpy as np
split_sd = 1.83      # across-split SD of the cell mean, folds of 52 engines
fold_n = 52
sigma_engine = split_sd * np.sqrt(fold_n)   # per-engine spread implied
print(f'across-split SD (fold of {fold_n}) = {split_sd:.2f} RMSE')
print(f'implied per-engine SD            = {sigma_engine:.1f} RMSE')
print()
print(' target effect   N engines needed (80% power)   vs available 260')
for target, label in ((0.65,'5% relative (internal challenge)'), (1.0,'~8% relative'), (1.5,'large effect'), (2.3,'detectable with 5 folds')):
    n = ((1.96 + 0.84) * sigma_engine / target) ** 2
    print(f' {target:5.2f} RMSE  {n:16.0f}                {n/260:6.1f}x')
