"""A vs BE (cycle vs no-cycle) on the frozen pipeline: summary + lifespan-stratified analysis."""
import json, glob, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,'src')
from dataset import load_cmapss_train
rows = []
for f in sorted(glob.glob('experiments/r3_cycle/*/FD002/*/metrics.json')):
    d = json.load(open(f)); tag = f.split('/')[2]
    var, seed = tag.split('_s')
    rows.append(dict(variant=var, seed=int(seed), val=d['best_val_rmse'],
                     pseudo=d.get('pseudo_test_rmse_mean'), params=d['params'], epochs=d['epochs_ran']))
t = pd.DataFrame(rows).sort_values(['variant','seed'])
print(t.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print()
p = t.pivot(index='seed', columns='variant', values='val')
print('val pivot:'); print(p.round(4).to_string())
d = (p['BE'] - p['A'])
print(f'BE - A (val): per-seed {[round(x,3) for x in d]} mean {d.mean():+.4f} sd {d.std():.4f}')
pp = t.pivot(index='seed', columns='variant', values='pseudo')
dp = (pp['BE'] - pp['A'])
print(f'BE - A (pseudo-test): per-seed {[round(x,3) for x in dp]} mean {dp.mean():+.4f}')
t.to_csv('results/dsh/review_r3/cycle_vs_nocycle_frozen.csv', index=False)

# lifespan-stratified analysis from per-engine predictions
life = load_cmapss_train(Path('data/CMAPSS'), 'FD002').groupby('unit')['cycle'].max()
recs = []
for f in sorted(glob.glob('experiments/r3_cycle/*/FD002/*/matched_cutoff_predictions.csv')):
    tag = f.split('/')[2]; var, seed = tag.split('_s')
    d2 = pd.read_csv(f)
    d2['variant'] = var; d2['seed'] = int(seed); d2['life'] = d2.unit_id.map(life)
    recs.append(d2)
if recs:
    m = pd.concat(recs)
    print(); print('per-engine prediction rows:', len(m), 'cols:', list(m.columns)[:6])
    m['e'] = m.y_pred - m.y_true
    q1, q2 = m.life.quantile([1/3, 2/3])
    m['band'] = np.where(m.life <= q1, 'short', np.where(m.life <= q2, 'mid', 'long'))
    g = m.groupby(['variant','band']).apply(lambda s: pd.Series({
        'n': len(s), 'rmse': float(np.sqrt((s.e**2).mean())), 'bias': float(s.e.mean()), 'late': float((s.e>0).mean())}))
    print(g.round(3).to_string())
    g.to_csv('results/dsh/review_r3/cycle_by_lifespan.csv')
else:
    print('no matched_cutoff_predictions.csv found')
