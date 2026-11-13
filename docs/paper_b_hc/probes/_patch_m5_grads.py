import io
import sys

p = 'paper_b/hc/train_d.py'
t = io.open(p, encoding='utf-8').read()

reps = [
("""def term_grads(net, arm, ebt, dev, K, cap, s0, lam_t, lam_s, n_eng=3,
               scr=False, lam_a=0.0):""",
 """def term_grads(net, arm, ebt, dev, K, cap, s0, lam_t, lam_s, n_eng=3,
               scr=False, lam_a=0.0, lam_dpc=0.0):"""),
("    w = {'rul': 1.0, 'terminal': lam_t, 'smooth': lam_s, 'alive': lam_a}",
 "    w = {'rul': 1.0, 'terminal': lam_t, 'smooth': lam_s, 'alive': lam_a,\n"
 "         'dpc': lam_dpc}"),
("""                         scr=(arm in SCR_ARMS), lam_a=(lam_a or 0.0))""",
 """                         scr=(arm in SCR_ARMS), lam_a=(lam_a or 0.0),
                         lam_dpc=(lam_dpc or 0.0))"""),
]

for old, new in reps:
    n = t.count(old)
    if n != 1:
        sys.exit('anchor count %d for %r' % (n, old[:60]))
    t = t.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('patched term_grads for dpc')
