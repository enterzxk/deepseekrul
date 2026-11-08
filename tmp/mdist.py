import sys; sys.path.insert(0,'/home/user/rulb/src')
from dataset import load_cmapss_train
import numpy as np
raw = load_cmapss_train('/home/user/rulb/data/CMAPSS','FD002')
L = raw.groupby('unit').size().to_numpy()
print('FD002 train engines:', len(L), ' length min/median/max:', L.min(), int(np.median(L)), L.max())
print()
print('Training window m = min(t+1,100) distribution:')
tot = L.sum()
for lo,hi in [(1,50),(50,60),(60,80),(80,99),(100,100)]:
    n = sum(max(0, min(l,hi) - lo + 1) for l in L)
    print('  m in [%3d,%3d]: %6d windows (%.1f%%)' % (lo,hi,n,100*n/tot))
print()
print('Official test m distribution (observed length, capped at 100):')
from dataset import load_cmapss_files
_, tf, _ = load_cmapss_files('/home/user/rulb/data/CMAPSS','FD002')
TL = tf.groupby('unit').size().to_numpy()
for lo,hi in [(1,50),(50,60),(60,80),(80,99),(100,999)]:
    n = ((TL>=lo)&(TL<=hi)).sum()
    print('  m in [%3d,%3d]: %3d engines (%.1f%%)' % (lo,hi,n,100*n/len(TL)))