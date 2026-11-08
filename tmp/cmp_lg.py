import glob, os, json
print('--- LG (200ep, patience 40) FD003 ---')
for f in sorted(glob.glob('/home/user/rulb/experiments/LG_FD003_s101_f*/FD003/*.json')):
    d = json.load(open(f))
    print('f%d  best_val %.4f  ep %d' % (d['fold'], d['best_val'], d['best_ep']))
print('--- M (standard, 80ep/20) FD003 seed101 ---')
for f in sorted(glob.glob('/home/user/rulb/experiments/M_FD003_s101_f*/FD003/*.json')):
    d = json.load(open(f))
    print('f%d  best_val %.4f  ep %d' % (d['fold'], d['best_val'], d['best_ep']))