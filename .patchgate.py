from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
if 'AG_ones' in s:
    print('already present'); raise SystemExit
anchor = '    "AB": dict(use_baseline=True, use_ema_features=True, use_cycle=True,'
assert anchor in s, 'anchor not found'
new = (
    "AG_ones": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                    use_graph=True, use_dual_head=False, use_model_ema=False,
                    gate_mode="ones", cls_weight=0.0, nasa_weight=0.0),
    "AG_const": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                     use_graph=True, use_dual_head=False, use_model_ema=False,
                     gate_mode="const", cls_weight=0.0, nasa_weight=0.0),
)
s = s.replace(anchor, new + chr(10) + anchor, 1)
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s)
print('added AG_ones and AG_const; syntax ok')
