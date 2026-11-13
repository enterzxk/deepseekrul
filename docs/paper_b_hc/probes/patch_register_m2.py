"""One-shot patch: register M2 (CDE) in train_d.py.

Written as a script because the same edit has now been lost twice to concurrent
rewrites of train_d.py: the file is edited by three agents and a plain string
replace through PowerShell cannot express the quotes safely. Idempotent -- if M2
is already registered it says so and changes nothing.
"""
from pathlib import Path

P = Path('paper_b/hc/train_d.py')
t = P.read_text(encoding='utf-8')
if "'M2'" in t:
    print('already registered, nothing to do')
    raise SystemExit(0)

pairs = [
    (
        "CSR_ARMS = ('M4',)\n",
        "CSR_ARMS = ('M4',)\n"
        "# M2 (CDE) hangs on the B0 backbone on purpose: B0 is the only arm proven\n"
        "# strong, so a gain measured on it is a real gain. Hanging a module on B3\n"
        "# would confound 'module useless' with 'backbone already bad'.\n"
        "CDE_ARMS = ('M2',)\n",
    ),
    (
        "NOHEAD_ARMS = ('B0',) + CSR_ARMS\n",
        "NOHEAD_ARMS = ('B0',) + CSR_ARMS + CDE_ARMS\n",
    ),
    (
        "ALL_ARMS = ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS\n",
        "ALL_ARMS = ARMS + ANCHORED_ARMS + SPRINT_ARMS + SCR_ARMS + CSR_ARMS + CDE_ARMS\n",
    ),
    (
        "    if arm in CSR_ARMS:\n"
        "        net = DSPT_CSR(csr_mode='cond', **kw)\n"
        "    else:\n"
        "        net = DSPT_HC_Structured(**kw)\n",
        "    if arm in CSR_ARMS:\n"
        "        net = DSPT_CSR(csr_mode='cond', **kw)\n"
        "    else:\n"
        "        net = DSPT_HC_Structured(**kw)\n",
    ),
    (
        "    return net.to(dev), sorted(missing)\n",
        "    if arm in CDE_ARMS:\n"
        "        from .cde import CDECorrection, CDEEvidence\n"
        "        net.cde = CDEEvidence(14, 32)\n"
        "        net.cde_head = CDECorrection(32)\n"
        "    return net.to(dev), sorted(missing)\n",
    ),
    (
        "    if arm == 'B1':\n"
        "        out['pred'] = net.reg_head(z).reshape(-1)\n"
        "    else:\n"
        "        out['pred'] = first_passage(D, net.head.future_rates(D, nu, z, K))\n",
        "    if arm in CDE_ARMS:\n"
        "        # M2 = B0 plus an explicit condition-matched evidence correction.\n"
        "        # The correction head is zero-init, so step 0 is bit-identical to B0\n"
        "        # and any later difference is attributable to the CDE evidence.\n"
        "        q = net.cde(x, b, m)\n"
        "        out['pred'] = net.reg_head(z).reshape(-1) + net.cde_head(q)\n"
        "        out['q'] = q\n"
        "    elif arm == 'B1':\n"
        "        out['pred'] = net.reg_head(z).reshape(-1)\n"
        "    else:\n"
        "        out['pred'] = first_passage(D, net.head.future_rates(D, nu, z, K))\n",
    ),
]

for old, new in pairs:
    if old not in t:
        raise SystemExit('anchor not found (file changed again):\n%r' % old[:90])
    t = t.replace(old, new, 1)

P.write_text(t, encoding='utf-8')
print('patched; M2 occurrences: %d' % t.count("'M2'"))
