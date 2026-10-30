from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
if 'quantile' in s:
    print('already has quantile'); raise SystemExit
old = '            loss = nn.functional.mse_loss(pred, y)'
new = '''            if args.quantile is not None:
                # Pinball loss: trains a conditional quantile instead of the mean,
                # so late risk is controlled during training rather than post hoc.
                r = y - pred
                loss = torch.maximum(args.quantile * r, (args.quantile - 1.0) * r).mean()
            else:
                loss = nn.functional.mse_loss(pred, y)'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('    parser.add_argument("--cp-labels", action="store_true")',
              '    parser.add_argument("--cp-labels", action="store_true")\n    parser.add_argument("--quantile", type=float, default=None)')
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s); print('quantile loss added; syntax ok')
