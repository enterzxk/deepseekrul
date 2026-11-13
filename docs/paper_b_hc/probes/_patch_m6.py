import io
import sys

p = 'paper_b/hc/train_d.py'
t = io.open(p, encoding='utf-8').read()

reps = [
("DPC_ARMS = ('M5',)",
 """DPC_ARMS = ('M5',)
# M6 (EFR): evidence-gated future degradation rate on the B3 backbone.
# nu_future = nu_age * exp(clamp(g_t * a_tk, -a, a)), g_t = sigmoid(gate(q_t)).
# The amplitude head is zero-init, so step 0 is exactly B3.
EFR_ARMS = ('M6',)"""),
("HI_ARMS = ('B1', 'B2', 'B3') + ANCHORED_ARMS + SCR_ARMS",
 "HI_ARMS = ('B1', 'B2', 'B3') + ANCHORED_ARMS + SCR_ARMS + EFR_ARMS"),
("ARRIVAL_ARMS = ('B2', 'B3') + ANCHORED_ARMS + SCR_ARMS",
 "ARRIVAL_ARMS = ('B2', 'B3') + ANCHORED_ARMS + SCR_ARMS + EFR_ARMS"),
("ALL_ARMS = ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS + CDE_ARMS + DPC_ARMS",
 "ALL_ARMS = (ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS + CDE_ARMS\n"
 "            + DPC_ARMS + EFR_ARMS)"),
("""    if arm in DPC_ARMS:
        from .dpc import DPCHead""",
 """    if arm in EFR_ARMS:
        from .cde import CDEEvidence
        from .efr import EFRCorrection
        net.cde = CDEEvidence(14, 32)
        net.efr = EFRCorrection(net.d_model, 32, clip=1.0)
    if arm in DPC_ARMS:
        from .dpc import DPCHead"""),
("""    elif arm == 'B1':
        out['pred'] = net.reg_head(z).reshape(-1)
    else:""",
 """    elif arm in EFR_ARMS:
        # B3's rate, multiplied by an evidence-gated correction that is 1 at
        # step 0. The arrival operator is the reviewed one, called as-is.
        q = net.cde(x, b, m)
        nu_age = net.head.future_rates(D, nu, z, K)
        out['pred'] = first_passage(D, net.efr.rates(nu_age, D, nu, z, q, K))
        out['q'] = q
    elif arm == 'B1':
        out['pred'] = net.reg_head(z).reshape(-1)
    else:"""),
]

for old, new in reps:
    n = t.count(old)
    if n != 1:
        sys.exit('anchor count %d for %r' % (n, old[:60]))
    t = t.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('registered M6')
