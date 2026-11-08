from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
if 'GRAPH' in s:
    print('already'); raise SystemExit
s=s.replace("def build_net(arm, window):", "def build_net(arm, window, graph=False):")
s=s.replace("use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)",
            "use_cycle=True, use_pe=True, use_graph=graph, use_dual_head=False)")
s=s.replace('''    if arm == "M0" or arm == "DIFF" or arm == "PHASE":
        return ida.PriorWrapper(body, "M0")''',
'''    if arm in ("M0", "DIFF", "PHASE", "GRAPH"):
        return ida.PriorWrapper(body, "M0")''')
s=s.replace('    ap.add_argument("--epochs", type=int, default=80)',
            '    ap.add_argument("--epochs", type=int, default=80)\n    ap.add_argument("--graph", action="store_true")')
s=s.replace('        net = build_net(arm, args.window)', '        net = build_net(arm, args.window, args.graph)')
s=s.replace('        subset=args.subset, fold=args.fold, seed=args.seed, window=args.window, weighting=args.weighting,',
            '        subset=args.subset, fold=args.fold, seed=args.seed, window=args.window, weighting=args.weighting, graph=args.graph,')
p.write_text(s); print('patched')
