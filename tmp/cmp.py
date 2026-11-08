import glob, os, re
def best(p):
    try:
        s = open(p).read()
    except Exception:
        return None
    m = re.findall(r'best_val=([0-9.]+)', s)
    return float(m[-1]) if m else None
print('--- 2-layer ---')
for f in sorted(glob.glob('/home/user/rulb/logs/L2_*.log')):
    b = best(f)
    print('%-24s %s' % (os.path.basename(f)[:-4], '%.4f' % b if b else 'running'))
print('--- 1-layer (seed 101) ---')
for sub in ['FD001','FD003']:
    vals = []
    for f in sorted(glob.glob('/home/user/rulb/logs/M_%s_s101_*.log' % sub)):
        b = best(f)
        if b: vals.append(b)
    if vals:
        print('%s  n=%d  mean %.4f  min %.4f' % (sub, len(vals), sum(vals)/len(vals), min(vals)))