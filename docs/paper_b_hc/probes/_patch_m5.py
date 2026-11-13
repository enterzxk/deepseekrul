import io
import sys

p = 'paper_b/hc/train_d.py'
t = io.open(p, encoding='utf-8').read()

reps = [
("CDE_ARMS = ('M2',)",
 """CDE_ARMS = ('M2',)
# M5 (DPC): multi-span degradation consistency. Training-only auxiliary head;
# the prediction path stays B0's reg_head(z), so inference is unchanged and the
# only difference from B0 is the auxiliary loss acting on the shared encoder.
DPC_ARMS = ('M5',)"""),
("NOHEAD_ARMS = ('B0',) + CSR_ARMS + CDE_ARMS",
 "NOHEAD_ARMS = ('B0',) + CSR_ARMS + CDE_ARMS + DPC_ARMS"),
("ALL_ARMS = ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS + CDE_ARMS",
 "ALL_ARMS = ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS + CDE_ARMS + DPC_ARMS"),
("""    if arm in CDE_ARMS:
        from .cde import CDECorrection, CDEEvidence
        net.cde = CDEEvidence(14, 32)
        net.cde_head = CDECorrection(32)
    return net.to(dev), sorted(missing)""",
 """    if arm in CDE_ARMS:
        from .cde import CDECorrection, CDEEvidence
        net.cde = CDEEvidence(14, 32)
        net.cde_head = CDECorrection(32)
    if arm in DPC_ARMS:
        from .dpc import DPCHead
        rh = net.reg_head
        dim = int(rh.in_features if hasattr(rh, 'in_features') else rh[0].in_features)
        net.dpc = DPCHead(dim)
    return net.to(dev), sorted(missing)"""),
("""    if arm == 'B0':
        return dict(pred=net.reg_head(z).reshape(-1), r=r, n=n)""",
 """    if arm in DPC_ARMS:
        # B0's prediction path plus the latent sequence the auxiliary needs.
        return dict(pred=net.reg_head(z).reshape(-1), r=r, n=n, z=z)
    if arm == 'B0':
        return dict(pred=net.reg_head(z).reshape(-1), r=r, n=n)"""),
("""                out['alive_n'] = torch.tensor(float(over.numel()), device=over.device)
    return out""",
 """                out['alive_n'] = torch.tensor(float(over.numel()), device=over.device)
    if arm in DPC_ARMS and 'z' in fo:
        d, k = net.dpc.loss(fo['z'])
        out['dpc_sum'] = d
        out['dpc_n'] = torch.tensor(1.0, device=d.device)
    return out"""),
("""    if 'alive_sum' in acc:
        L['alive'] = acc['alive_sum'] / acc['alive_n'].clamp_min(1.0)
    return L""",
 """    if 'alive_sum' in acc:
        L['alive'] = acc['alive_sum'] / acc['alive_n'].clamp_min(1.0)
    if 'dpc_sum' in acc:
        L['dpc'] = acc['dpc_sum'] / acc['dpc_n'].clamp_min(1.0)
    return L"""),
("""            if 'alive' in L:
                total = total + lam_a * L['alive']
                diag['alive'] = float(L['alive'].detach())""",
 """            if 'alive' in L:
                total = total + lam_a * L['alive']
                diag['alive'] = float(L['alive'].detach())
            if 'dpc' in L:
                total = total + lam_dpc * L['dpc']
                diag['dpc'] = float(L['dpc'].detach())"""),
("    lam_a = float(C.RUL_CAP ** 2 if lam_a is None else lam_a)",
 "    lam_a = float(C.RUL_CAP ** 2 if lam_a is None else lam_a)\n"
 "    lam_dpc = float(1.0 if lam_dpc is None else lam_dpc)"),
("budget_override=None, scr=False, lam_a=None):",
 "budget_override=None, scr=False, lam_a=None, lam_dpc=None):"),
("    ap.add_argument('--lam-a', type=float, default=None)",
 """    ap.add_argument('--lam-a', type=float, default=None)
    ap.add_argument('--lam-dpc', type=float, default=None,
                    help='M5: weight on the multi-span latent consistency loss')"""),
("                  scr=a.scr, lam_a=a.lam_a)",
 "                  scr=a.scr, lam_a=a.lam_a, lam_dpc=a.lam_dpc)"),
("""            log('  ep%03d loss=%.4f val_eng=%.4f alive=%s g_ratio=%s'
                % (ep, tl, ev['val_engine_mean_rmse'],
                   ('%.3e' % diag['alive']) if 'alive' in diag else '-',""",
 """            log('  ep%03d loss=%.4f val_eng=%.4f alive=%s dpc=%s g_ratio=%s'
                % (ep, tl, ev['val_engine_mean_rmse'],
                   ('%.3e' % diag['alive']) if 'alive' in diag else '-',
                   ('%.4f' % diag['dpc']) if 'dpc' in diag else '-',"""),
]

for old, new in reps:
    n = t.count(old)
    if n != 1:
        sys.exit('anchor count %d for %r' % (n, old[:60]))
    t = t.replace(old, new)

io.open(p, 'w', encoding='utf-8', newline='\n').write(t)
print('patched M5 into train_d.py')
