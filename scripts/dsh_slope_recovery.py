"""How much of the lifespan->RUL slope does the model recover?"""
import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train
life = load_cmapss_train(Path('data/CMAPSS'), 'FD002').groupby('unit')['cycle'].max()
fs = glob.glob('experiments/order03/M1_global/FD002/*/pseudo_test_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['p'] = np.clip(d.y_pred, 0, 125)
d['life'] = d.unit_id.map(life)
print(f"{'frac':>5s} {'n':>4s} {'true slope':>11s} {'pred slope':>11s} {'recovery':>9s}")
for f in sorted(d.cutoff.unique()):
    s = d[np.isclose(d.cutoff, f)]
    x = s.life.to_numpy(float)
    bt = np.polyfit(x, s.y_true.to_numpy(float), 1)[0]
    bp = np.polyfit(x, s.p.to_numpy(float), 1)[0]
    rec = bp / bt if abs(bt) > 1e-9 else float('nan')
    print(f'{f:5.1f} {len(s):4d} {bt:11.4f} {bp:11.4f} {rec:9.1%}')
