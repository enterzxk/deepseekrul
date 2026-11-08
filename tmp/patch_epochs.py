from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
if '--epochs' in s:
    print('already'); raise SystemExit
s=s.replace('def train_arm(arm, net, train_data, valid_data, target, metadata, seed, save_path=None, weights=None):',
            'def train_arm(arm, net, train_data, valid_data, target, metadata, seed, save_path=None, weights=None, epochs=80):')
s=s.replace('    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)',
            '    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)')
s=s.replace('    for _ in range(EPOCHS):\n        net.train()', '    for _ in range(epochs):\n        net.train()')
s=s.replace('    ap.add_argument("--tag", default="")', '    ap.add_argument("--tag", default="")\n    ap.add_argument("--epochs", type=int, default=80)')
s=s.replace('dest = HERE / (f"runs3_w{args.window}" + (args.tag and "_" + args.tag)) / args.subset / str(args.fold)',
            'dest = HERE / (f"runs3_w{args.window}" + (args.tag and "_" + args.tag)) / args.subset / str(args.fold)')
s=s.replace('wrote = train_arm(arm, net, tx, vx, target, meta, args.seed, dest / f"{arm}_{args.seed}.pt", weights)',
            'wrote = train_arm(arm, net, tx, vx, target, meta, args.seed, dest / f"{arm}_{args.seed}.pt", weights, args.epochs)')
p.write_text(s); print('patched')
