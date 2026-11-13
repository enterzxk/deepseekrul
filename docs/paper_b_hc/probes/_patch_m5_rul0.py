import io
import sys

p = 'paper_b/hc/train_d.py'
t = io.open(p, encoding='utf-8').read()

reps = [
("""    dpc0 = None
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
        log('  dpc0 (init, detached) = %.6f from %d engines' % (dpc0, len(vals)))""",
 """    dpc0 = rul0 = None
    if arm in DPC_ARMS:
        was_training = net.training
        net.eval()
        with torch.no_grad():
            dvals, rvals = [], []
            for _i in range(min(8, len(ebt['lengths']))):
                _n = int(ebt['lengths'][_i])
                _fo = forward_engine(net, arm, ebt, _i, _n, dev, K)
                _d, _k = net.dpc.loss(_fo['z'])
                if _k:
                    dvals.append(float(_d))
                    rvals.append(float(rul_loss(_fo['pred'], _fo['r'], cap).mean()))
        net.train(was_training)
        dpc0 = max(1e-12, float(np.mean(dvals))) if dvals else 1.0
        rul0 = max(1e-12, float(np.mean(rvals))) if rvals else 1.0
        # Both scales are measured in the SAME init pass and detached. The loss
        # term is then lam_dpc * (L_dpc/dpc0) * rul0, so lam_dpc is literally
        # "fraction of the initial RUL loss" and does not drift when L_dpc moves.
        log('  dpc0=%.6f rul0=%.2f from %d engines -> lam_dpc is a SHARE of rul0'
            % (dpc0, rul0, len(dvals)))"""),
("                total = total + lam_dpc * (L['dpc'] / dpc0)",
 "                total = total + lam_dpc * (L['dpc'] / dpc0) * rul0"),
("                         lam_dpc=((lam_dpc or 0.0) / (dpc0 or 1.0)))",
 "                         lam_dpc=((lam_dpc or 0.0) * (rul0 or 1.0) / (dpc0 or 1.0)))"),
]

for old, new in reps:
    n = t.count(old)
    if n != 1:
        sys.exit('anchor count %d for %r' % (n, old[:60]))
    t = t.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('patched rul0 scaling')
