"""Independently recompute the three口径 for B2 s101 and B3 s101 from history.json.

The +10.05 that lived in five places was a single-seed-vs-three-seed-mean
difference. The replacement numbers (best / ema_min / tail10_mean) are about to
be written into the paper, so they get the same independent treatment: read
history.json directly, no pipeline.

Definitions used here, stated so they can be checked:
  best        min over epochs of val_engine_mean_rmse
  tail10_mean mean of the last 10 epochs
  ema_min     min over epochs of an exponential moving average, reported for
              several alphas because the definition must match what produced
              25.9924 / 15.5597
"""
import json

RUNS = {
    'B2 s101': '/home/user/sprint24/experiments/hc/_gate/HC-D-B2-FD001-s101-D1/history.json',
    'B3 s101': '/home/user/sprint24/experiments/hc/_gate/HC-D-B3-FD001-s101-D1/history.json',
}

out = {}
for tag, p in RUNS.items():
    h = json.load(open(p))
    vals = [r['val_engine_mean_rmse'] for r in h]
    best = min(vals)
    tail10 = sum(vals[-10:]) / 10.0
    emas = {}
    for a in (0.1, 0.2, 0.3, 0.5):
        e = vals[0]
        mn = e
        for v in vals[1:]:
            e = a * v + (1 - a) * e
            mn = min(mn, e)
        emas[a] = mn
    out[tag] = dict(n=len(vals), best=best, tail10=tail10, emas=emas,
                    last=vals[-1], first=vals[0])
    print('%s: n=%d first=%.4f last=%.4f best=%.4f tail10=%.4f'
          % (tag, len(vals), vals[0], vals[-1], best, tail10))
    print('   ema_min by alpha: ' + '  '.join('a=%.1f:%.4f' % (a, v) for a, v in emas.items()))

print('')
print('== differences (B2 - B3) ==')
for key in ('best', 'tail10'):
    d = out['B2 s101'][key] - out['B3 s101'][key]
    print('%-8s %.4f - %.4f = %+.4f' % (key, out['B2 s101'][key], out['B3 s101'][key], d))
for a in (0.1, 0.2, 0.3, 0.5):
    d = out['B2 s101']['emas'][a] - out['B3 s101']['emas'][a]
    print('ema(a=%.1f) %.4f - %.4f = %+.4f'
          % (a, out['B2 s101']['emas'][a], out['B3 s101']['emas'][a], d))

print('')
print('== the withdrawn number ==')
b3_three = (14.6884 + 14.0142 + 14.6046) / 3.0
print('B3 three-seed mean = %.6f  (14.6884/14.0142/14.6046)' % b3_three)
print('24.4884 - %.6f = %.6f   <- this is where +10.05 came from' % (b3_three, 24.4884 - b3_three))
print('24.4884 - 14.6884 (same seed) = %.6f' % (24.4884 - 14.6884))
