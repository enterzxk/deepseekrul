import json
import statistics

for tag, p in [
    ('B2 s101', '/home/user/sprint24/experiments/hc/_gate/HC-D-B2-FD001-s101-D1/history.json'),
    ('B3 s101', '/home/user/sprint24/experiments/hc/_gate/HC-D-B3-FD001-s101-D1/history.json'),
]:
    h = json.load(open(p))
    if isinstance(h, dict):
        print('%s: dict keys=%s' % (tag, sorted(h)[:12]))
        for k, v in list(h.items())[:3]:
            print('   %s: %s len=%s' % (k, type(v).__name__, len(v) if hasattr(v, '__len__') else '-'))
        # find the per-epoch val list
        vals = None
        for k in ('val_engine_mean_rmse', 'val', 'val_eng', 'history'):
            if k in h and isinstance(h[k], list):
                vals = h[k]
                key = k
                break
        if vals:
            print('   using %s: n=%d first=%.4f last=%.4f min=%.4f'
                  % (key, len(vals), vals[0], vals[-1], min(vals)))
    else:
        print('%s: list n=%d first=%s' % (tag, len(h), str(h[0])[:80]))
