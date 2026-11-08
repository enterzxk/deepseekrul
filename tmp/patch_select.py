from pathlib import Path
p=Path('run_v3b.py'); s=p.read_text()

# 1) signature
s=s.replace("target_mode='rul', mixup=0.0, mixup_mode='all'):",
            "target_mode='rul', mixup=0.0, mixup_mode='all', select='final'):",1)

# 2) add validation loader before the epoch loop
anchor='    for _ in range(epochs):\n        net.train()'
prep=('    va = [base.device_array(a) for a in va_raw[:4]]\n'
      '    def val_rmse():\n'
      '        net.eval()\n'
      '        with torch.no_grad():\n'
      '            r = torch.cat([fwd(va, i) for i in torch.arange(len(va[0]), device="cuda").split(BATCH)])\n'
      '        return float(torch.sqrt(torch.mean((r - va[1]) ** 2)))\n'
      '    best_state, best_val, best_ep = None, float("inf"), -1\n'
      '    for _ep in range(epochs):\n        net.train()')
assert anchor in s, 'loop anchor'
s=s.replace(anchor, prep, 1)

# 3) after sch.step(), record best
old='        sch.step()\n    va = [base.device_array(a) for a in va_raw[:4]]'
new=('        sch.step()\n'
     '        if select == "best":\n'
     '            v = val_rmse()\n'
     '            if v < best_val - 1e-4:\n'
     '                best_val, best_ep = v, _ep\n'
     '                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}\n'
     '    if select == "best" and best_state is not None:\n'
     '        net.load_state_dict(best_state)\n'
     '    va = [base.device_array(a) for a in va_raw[:4]]')
assert old in s, 'step anchor'
s=s.replace(old,new,1)

# 4) CLI flag
a='ap.add_argument("--mixup-mode", default="all")'
if a in s:
    s=s.replace(a, a+'\nap.add_argument("--select", default="final", choices=["final","best"])',1)
else:
    s=s.replace('ap.add_argument("--mixup"', 'ap.add_argument("--select", default="final", choices=["final","best"])\nap.add_argument("--mixup"',1)
p.write_text(s); print('patched select')