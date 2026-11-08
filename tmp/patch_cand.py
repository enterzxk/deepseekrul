from pathlib import Path
p=Path('scripts/run_candidate.py'); s=p.read_text()
a='    parser.add_argument("--quantile", type=float, default=None)'
b='    parser.add_argument("--quantile", type=float, default=None)\n    parser.add_argument("--mixup", type=float, default=0.0)'
assert a in s, 'arg anchor'
s=s.replace(a,b,1)
old='''            x, y, cond, extras = x.to(device), y.to(device), cond.to(device), extras.to(device)
            optimiser.zero_grad(set_to_none=True)'''
new='''            x, y, cond, extras = x.to(device), y.to(device), cond.to(device), extras.to(device)
            if args.mixup > 0:
                _lam = float(np.random.beta(args.mixup, args.mixup))
                _perm = torch.randperm(x.shape[0], device=device)
                x = _lam * x + (1 - _lam) * x[_perm]
                y = _lam * y + (1 - _lam) * y[_perm]
                cond = _lam * cond + (1 - _lam) * cond[_perm]
                extras = _lam * extras + (1 - _lam) * extras[_perm]
            optimiser.zero_grad(set_to_none=True)'''
assert old in s, 'loop anchor'
s=s.replace(old,new,1)
p.write_text(s); print('patched mixup')