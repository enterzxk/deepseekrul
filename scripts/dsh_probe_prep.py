import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')
print('tuple len:', len(prep))
for i, item in enumerate(prep):
    t = type(item).__name__
    if isinstance(item, (list, tuple)): extra = f'len={len(item)}'
    elif hasattr(item, '__len__') and not isinstance(item, str): extra = f'len={len(item)}'
    else: extra = ''
    print(f'  [{i}] {t} {extra}')
vs = prep[1]
print('val_set attrs:', [a for a in dir(vs) if not a.startswith('_')][:12])
print('val_unit_ids sample:', list(prep[5])[:5] if isinstance(prep[5], (list, tuple)) else type(prep[5]))
