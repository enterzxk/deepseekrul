from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
if '--subsets' in s:
    print('already supported'); raise SystemExit
# 1) CLI
s = s.replace('    parser.add_argument("--subset", required=True)',
              '    parser.add_argument("--subset")\n    parser.add_argument("--subsets", default=None,' + chr(10) +
              '                        help="comma-separated subsets to pool, e.g. FD002,FD004")')
# 2) prepare signature already takes subset str; add pooling inside
old = '    raw_train = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)'
new = '''    if "," in subset:
        parts = [x.strip() for x in subset.split(",") if x.strip()]
        frames = []
        for i, sub in enumerate(parts):
            f = load_cmapss_train(ROOT / "data" / "CMAPSS", sub)
            f = f.copy()
            f["unit"] = f["unit"].astype(int) + i * 10000   # keep ids globally unique
            f["source_subset"] = sub
            frames.append(f)
        raw_train = pd.concat(frames, ignore_index=True)
    else:
        raw_train = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)'''
assert old in s
s = s.replace(old, new, 1)
# 3) resolve --subsets into subset string
old2 = '        args.subset, args.split_seed, args.window, 125.0,'
new2 = '        (args.subsets or args.subset), args.split_seed, args.window, 125.0,'
assert old2 in s
s = s.replace(old2, new2, 1)
s = s.replace('condition_aware=args.subset in {"FD002", "FD004"}, n_cond=6,',
              'condition_aware=any(x in (args.subsets or args.subset) for x in ("FD002", "FD004")), n_cond=6,')
s = s.replace('out_dir = ROOT / args.out / args.subset / f"{args.variant}_s{args.seed}"',
              'out_dir = ROOT / args.out / (args.subsets or args.subset).replace(",", "+") / f"{args.variant}_s{args.seed}"')
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('--subsets implemented; syntax ok')
