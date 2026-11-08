import glob, json
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        j = json.load(open(f)); d[j['fold']] = j['best_val']
    return d
b = load('/home/user/rulb/experiments/M_FD001_s101_f*/FD001/*.json')
c = load('/home/user/rulb/experiments/CAP_FD001_s101_f*/FD001/*.json')
print('fold  baseline  capacity  delta')
bs, cs = [], []
for f in sorted(b):
    bv = b[f]; cv = c.get(f)
    if bv: bs.append(bv)
    if cv: cs.append(cv)
    d = ('%+.2f%%' % (100*(cv/bv-1))) if cv else 'running'
    print('%4d  %8.4f  %8s  %8s' % (f, bv, ('%.4f'%cv) if cv else '--', d))
if cs: print('mean baseline %.4f  capacity %.4f  (%+.2f%%)' % (sum(bs[:len(cs)])/len(cs), sum(cs)/len(cs), 100*(sum(cs)/len(cs))/(sum(bs[:len(cs)])/len(cs))-100))