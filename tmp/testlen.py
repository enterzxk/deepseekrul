import sys; sys.path.insert(0,'/home/user/rulb/src')
from dataset import load_cmapss_files
import numpy as np
for s in ['FD001','FD002','FD003','FD004']:
    _, tf, rul = load_cmapss_files('/home/user/rulb/data/CMAPSS', s)
    L = tf.groupby('unit').size().to_numpy()
    print('%s n=%d  observed length: min %d  p25 %d  median %d  p75 %d  max %d  |  <100 cycles: %d (%.0f%%)' % (
        s, len(L), L.min(), np.percentile(L,25), np.median(L), np.percentile(L,75), L.max(), (L<100).sum(), 100*(L<100).mean()))