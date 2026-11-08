import sys, numpy as np
from pathlib import Path
sys.path[:0] = ['/home/user/rulb/src', '/home/user/rulb']
from dataset import load_cmapss_train
import re
src = Path('/home/user/rulb/src/candidates/dspt.py').read_text()
for m in re.finditer(r'^(MAX_SPAN|SHORT_SPAN)\s*=.*$', src, re.M):
    print(m.group(0))
for s in ['FD001','FD002','FD003','FD004']:
    L = load_cmapss_train(Path('/home/user/rulb/data/CMAPSS'), s).groupby('unit').size().to_numpy()
    print('%-6s mean %3.0f  p10 %3.0f  p50 %3.0f  p90 %3.0f  max %4d  frac>100 %.2f  frac>150 %.2f'
          % (s, L.mean(), np.percentile(L,10), np.percentile(L,50), np.percentile(L,90), L.max(), (L>100).mean(), (L>150).mean()))