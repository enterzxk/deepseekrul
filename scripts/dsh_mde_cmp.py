import numpy as np
# observed at n=52 from the run just completed
se52_late = 0.25 / (2*1.96)      # CI width 0.25 at n=52
se260 = se52_late / np.sqrt(260/52)
mde260_late = 2.8 * se260
base_late = 0.48
print(f'late-fraction: SE(52)={se52_late:.3f}  SE(260)={se260:.3f}  MDE(260)={mde260_late:.3f} = {100*mde260_late/base_late:.0f}% of baseline')
# RMSE from the earlier arithmetic
sigma = 13.2; mde_rmse = 2.8*sigma/np.sqrt(260)
print(f'RMSE:          MDE(260)={mde_rmse:.2f} = {100*mde_rmse/13.5:.0f}% of baseline')
print()
print('=> both metrics land near 17% relative MDE with all 260 engines')
print('=> the 5% target is ~3.4x beyond reach whatever the metric')
