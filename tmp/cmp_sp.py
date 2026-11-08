import glob, json
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        j = json.load(open(f)); d[j['fold']] = j['best_val']
    return d
b = load('/home/user/rulb/experiments/M_FD003_s101_f*/FD003/*.json')
s = load('/home/user/rulb/experiments/SP200_FD003_s101_f*/FD003/*.json')
print('fold  baseline  span200   delta')
bs, ss = [], []
for f in sorted(b):
    bv = b[f]; sv = s.get(f)
    if bv: bs.append(bv)
    if sv: ss.append(sv)
    d = ('%+.2f%%' % (100*(sv/bv-1))) if sv else 'running'
    print('%4d  %8.4f  %8s  %8s' % (f, bv, ('%.4f'%sv) if sv else '--', d))
if ss: print('mean baseline %.4f  span200 %.4f  (%+.2f%%)' % (sum(bs[:len(ss)])/len(ss), sum(ss)/len(ss), 100*(sum(ss)/len(ss))/(sum(bs[:len(ss)])/len(ss))-100))