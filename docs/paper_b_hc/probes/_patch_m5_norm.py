import io
import sys

p = 'paper_b/hc/train_d.py'
t = io.open(p, encoding='utf-8').read()

reps = [
("""    params = trainable(net, arm)
    nparam = module_counts(net, arm)""",
 """    params = trainable(net, arm)
    nparam = module_counts(net, arm)
    # M5: normalise the auxiliary by its own initial value, exactly like S0.
    # Without this, lam_dpc is an absolute weight on a term whose scale moves:
    # measured 0.0052 -> 0.0960 over 15 epochs (18x), which took the gradient
    # share from 15% to 100% and stopped being a test of M5.
    dpc0 = None
    if arm in DPC_ARMS:
        was_training = net.training
        net.eval()
        with torch.no_grad():
            vals = []
            for _i in range(min(8, len(ebt['lengths']))):
                _n = int(ebt['lengths'][_i])
                _fo = forward_engine(net, arm, ebt, _i, _n, dev, K)
                _d, _k = net.dpc.loss(_fo['z'])
                if _k:
                    vals.append(float(_d))
        net.train(was_training)
        dpc0 = max(1e-12, float(np.mean(vals))) if vals else 1.0
        log('  dpc0 (init, detached) = %.6f from %d engines' % (dpc0, len(vals)))"""),
("                total = total + lam_dpc * L['dpc']",
 "                total = total + lam_dpc * (L['dpc'] / dpc0)"),
("                         lam_dpc=(lam_dpc or 0.0))",
 "                         lam_dpc=((lam_dpc or 0.0) / (dpc0 or 1.0)))"),
]

for old, new in reps:
    n = t.count(old)
    if n != 1:
        sys.exit('anchor count %d for %r' % (n, old[:60]))
    t = t.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('patched dpc normalisation')
