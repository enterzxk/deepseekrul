import glob, json
def load(pat, sub):
    d = {}
    for f in sorted(glob.glob(pat)):
        j = json.load(open(f)); d[j['fold']] = j['best_val']
    return d
for sub in ['FD001','FD002','FD004']:
    b = load('/home/user/rulb/experiments/M_%s_s101_f*/%s/*.json' % (sub, sub), sub)
    i = load('/home/user/rulb/experiments/IN_%s_s101_f*/%s/*.json' % (sub, sub), sub)
    if not i: continue
    print('---', sub)
    bs, ins = [], []
    for f in sorted(b):
        bv = b.get(f); iv = i.get(f)
        if bv: bs.append(bv)
        if iv: ins.append(iv)
        d = ('%+.2f%%' % (100*(iv/bv-1))) if (bv and iv) else 'running'
        print('  f%d  base %.4f  inorm %s  %s' % (f, bv, ('%.4f'%iv) if iv else '--', d))
    if ins: print('  mean base %.4f  mean inorm %.4f' % (sum(bs[:len(ins)])/len(ins), sum(ins)/len(ins)))