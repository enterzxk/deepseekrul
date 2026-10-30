"""Can FD003/FD004 TRAIN data actually be loaded and used?"""
import sys, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
cols = list(SELECTED_SENSOR_COLUMNS)
for sub in ('FD001','FD002','FD003','FD004'):
    try:
        df = load_cmapss_train(Path('data/CMAPSS'), sub)
    except Exception as e:
        print(f'{sub}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}'); continue
    missing = [c for c in cols if c not in df.columns]
    nan = int(df[cols].isna().sum().sum()) if not missing else -1
    lens = df.groupby('unit').size()
    print(f'{sub}: units={df.unit.nunique():4d} rows={len(df):6d} missing_cols={len(missing)} nan={nan} life[min/med/max]={lens.min()}/{int(lens.median())}/{lens.max()}')
