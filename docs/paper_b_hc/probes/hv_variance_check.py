"""Independent check of H-V: is the variance claim's F test set up correctly?

Three arms, three seeds each. If the question is "is the M3B/M6 dispersion smaller
than B3's", the degrees of freedom are (df_num, df_den) = (2+2, 2) = (4, 2) when
the two small-variance arms are pooled, NOT (4, 8). The critical value differs by
a factor of five, so this decides whether the conclusion stands.

Everything is read from metrics.json.
"""
import glob
import json
import os

ROOTS = ['/home/user/sprint24/experiments/hc/_gate',
         '/home/user/sprint24/experiments/hc/d80',
         '/home/user/sprint24/experiments/hc/d']

cells = {}
for root in ROOTS:
    for mf in glob.glob(os.path.join(root, '*', 'metrics.json')):
        try:
            m = json.load(open(mf))
        except Exception:
            continue
        if m.get('subset') != 'FD001':
            continue
        ep = int((m.get('budget') or {}).get('epochs') or 0)
        if ep != 80:
            continue
        key = (m.get('arm'), m.get('seed'))
        if key in cells and '_gate' not in root:
            continue
        cells[key] = m.get('best_val_engine_mean_rmse')


def vals(arm):
    out = sorted((s, v) for (a, s), v in cells.items() if a == arm and v is not None)
    return out


def var(xs):
    n = len(xs)
    mu = sum(xs) / n
    return sum((x - mu) ** 2 for x in xs) / (n - 1)


for arm in ('B3', 'M3B', 'M6', 'B0', 'M2'):
    v = vals(arm)
    if v:
        xs = [x for _, x in v]
        print('%-4s n=%d  %s  mean=%.4f  var=%.6f  sd=%.4f'
              % (arm, len(xs), ['s%d=%.4f' % (s, x) for s, x in v],
                 sum(xs) / len(xs), var(xs), var(xs) ** 0.5))

m3b = [x for _, x in vals('M3B')]
m6 = [x for _, x in vals('M6')]
b3 = [x for _, x in vals('B3')]
if len(m3b) == 3 and len(m6) == 3 and len(b3) == 3:
    df_num = (len(m3b) - 1) + (len(m6) - 1)
    pooled = (var(m3b) * (len(m3b) - 1) + var(m6) * (len(m6) - 1)) / df_num
    df_den = len(b3) - 1
    F = pooled / var(b3)
    print('')
    print('pooled var(M3B,M6) = %.6f  df_num = %d' % (pooled, df_num))
    print('var(B3)            = %.6f  df_den = %d' % (var(b3), df_den))
    print('F = %.4f   sd ratio = %.4f' % (F, pooled ** 0.5 / var(b3) ** 0.5))
    print('')
    print('critical values at 0.95:')
    print('  F(4,2) = 19.25   <- the correct one for this design')
    print('  F(4,8) =  3.84   <- what a (4,8) reading would use')
    print('  F(2,2) = 19.00   <- if the arms were compared one at a time')
    print('')
    print('verdict with F(4,2): %s' % ('REJECT equal variances' if F > 19.25 else 'cannot reject'))

# p-value via the incomplete beta identity: P(F <= f) = I_{d1 f/(d1 f + d2)}(d1/2, d2/2)
d1, d2 = df_num, df_den
x = d1 * F / (d1 * F + d2)
# I_x(2,1) = x^2 exactly, which is this design's shape
pval = x ** (d1 / 2.0)
print('')
print('x = %.6f   p = P(F(4,2) <= %.4f) = %.4f' % (x, F, pval))
print('sanity: F(4,2) at 0.95 -> x=%.4f, x^2=%.4f (should be 0.95)'
      % (4 * 19.25 / (4 * 19.25 + 2), (4 * 19.25 / (4 * 19.25 + 2)) ** 2))
print('verdict at alpha=0.05: %s' % ('reject equal variances' if pval < 0.05 else 'CANNOT reject (p=%.3f)' % pval))
