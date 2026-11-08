from pathlib import Path
p=Path('run_v3b.py'); s=p.read_text()
s=s.replace("target_mode='rul', mixup=0.0):", "target_mode='rul', mixup=0.0, mixup_mode='all'):")
old='''                pred = net(lam * xi + (1 - lam) * xj, None, lam * ei + (1 - lam) * ej,
                           lam * pi + (1 - lam) * pj)['prediction'].reshape(-1)'''
new='''                xmix = lam * xi + (1 - lam) * xj
                if mixup_mode == 'x':
                    pred = net(xmix, None, ei, pi)['prediction'].reshape(-1)
                else:
                    pred = net(xmix, None, lam * ei + (1 - lam) * ej,
                               lam * pi + (1 - lam) * pj)['prediction'].reshape(-1)'''
assert old in s, 'mix block'
s=s.replace(old,new)
s=s.replace("ap.add_argument('--mixup', type=float, default=0.0)",
            "ap.add_argument('--mixup', type=float, default=0.0)\n    ap.add_argument('--mixup-mode', default='all')")
s=s.replace('life_tr, life_va, a.target_mode, a.mixup)',
            'life_tr, life_va, a.target_mode, a.mixup, a.mixup_mode)')
p.write_text(s); print('patched')