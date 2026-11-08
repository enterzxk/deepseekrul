import glob, json, os
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        j = json.load(open(f))
        d[j['fold']] = j['best_val']
    return d
base = load('/home/user/rulb/experiments/M_FD003_s101_f*/FD003/*.json')
inorm = load('/home/user/rulb/experiments/IN_FD003_s101_f*/FD003/*.json')
print('fold  baseline  instnorm   delta')
bs, ins = [], []
for f in sorted(set(base) | set(inorm)):
    b = base.get(f); i = inorm.get(f)
    if b is not None: bs.append(b)
    if i is not None: ins.append(i)
    d = ('%+.2f%%' % (100*(i/b-1))) if (b and i) else '--'
    print('%4d  %8s  %8s  %8s' % (f, ('%.4f'%b) if b else '--', ('%.4f'%i) if i else '--', d))
if bs: print('mean baseline %.4f (n=%d)' % (sum(bs)/len(bs), len(bs)))
if ins: print('mean instnorm %.4f (n=%d)' % (sum(ins)/len(ins), len(ins)))