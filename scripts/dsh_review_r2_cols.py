import sys; sys.path.insert(0,'src')
from pathlib import Path
from dataset import load_cmapss_train
d = load_cmapss_train(Path('data/CMAPSS'), 'FD001')
print('columns:', list(d.columns))
