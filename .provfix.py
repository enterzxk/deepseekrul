from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
if 'run_hyperparameters' in s:
    print('already patched'); raise SystemExit
anchor = '        "config": cfg, "params": params, "best_val_rmse": best,'
assert anchor in s
ins = '''        "run_hyperparameters": {
            "late_weight": args.late_weight,
            "quantile": args.quantile,
            "dropout": args.dropout,
            "lr": args.lr,
            "window": args.window,
            "epochs_requested": args.epochs,
        },
'''
s = s.replace(anchor, ins + anchor, 1)
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s)
print('patched run_hyperparameters into metrics.json')
