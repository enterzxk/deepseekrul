from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
if 'weighting' in s:
    print('already patched'); raise SystemExit
s=s.replace(
"def train_arm(arm, net, train_data, valid_data, target, metadata, seed, save_path=None):",
"def train_arm(arm, net, train_data, valid_data, target, metadata, seed, save_path=None, weights=None):")
s=s.replace(
"""    for _ in range(EPOCHS):
        net.train()
        for idx in torch.randperm(len(tr[0]), generator=g, device="cuda").split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            error = forward(tr, idx) - tr[1][idx]
            loss = error.square().mean() + .5 * torch.where(error > 0, 1.8, 1.).mul(error.square()).mean()""",
"""    w_all = None if weights is None else base.device_array(weights)
    for _ in range(EPOCHS):
        net.train()
        for idx in torch.randperm(len(tr[0]), generator=g, device="cuda").split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            error = forward(tr, idx) - tr[1][idx]
            if w_all is None:
                loss = error.square().mean() + .5 * torch.where(error > 0, 1.8, 1.).mul(error.square()).mean()
            else:
                w = w_all[idx]
                denom = w.sum()
                loss = (w * error.square()).sum() / denom + .5 * (w * torch.where(error > 0, 1.8, 1.) * error.square()).sum() / denom""")
s=s.replace(
"""    ap.add_argument("--arms", default="M0,DIFF,PHASE")""",
"""    ap.add_argument("--arms", default="M0,DIFF,PHASE")
    ap.add_argument("--weighting", default="window", choices=["window", "engine"])
    ap.add_argument("--tag", default="")""")
s=s.replace(
"""    dest = HERE / f"runs3_w{args.window}" / args.subset / str(args.fold)""",
"""    dest = HERE / (f"runs3_w{args.window}" + (args.tag and "_" + args.tag)) / args.subset / str(args.fold)""")
s=s.replace(
"""        target = dest / f"{arm}_{args.seed}.csv"
        wrote = train_arm(arm, net, tx, vx, target, meta, args.seed, dest / f"{arm}_{args.seed}.pt")""",
"""        target = dest / f"{arm}_{args.seed}.csv"
        weights = None
        if args.weighting == "engine":
            counts = pd.Series(mtr[:, 0]).value_counts()
            weights = (1.0 / pd.Series(mtr[:, 0]).map(counts).to_numpy()).astype(np.float32)
        wrote = train_arm(arm, net, tx, vx, target, meta, args.seed, dest / f"{arm}_{args.seed}.pt", weights)""")
s=s.replace(
"""        subset=args.subset, fold=args.fold, seed=args.seed, window=args.window,""",
"""        subset=args.subset, fold=args.fold, seed=args.seed, window=args.window, weighting=args.weighting,""")
p.write_text(s); print('patched')
