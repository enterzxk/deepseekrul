from pathlib import Path
p = Path('scripts/run_candidate.py')
s = p.read_text(encoding='utf-8')
before = s.count('use_reliability=True, gate_mode')
# add use_reliability=True to the two AG variants
s = s.replace('''    "AG_ones": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                    use_graph=True, use_dual_head=False, use_model_ema=False,
                    gate_mode="ones", cls_weight=0.0, nasa_weight=0.0),''',
'''    "AG_ones": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                    use_graph=True, use_dual_head=False, use_model_ema=False,
                    use_reliability=True, gate_mode="ones", cls_weight=0.0, nasa_weight=0.0),''')
s = s.replace('''    "AG_const": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                     use_graph=True, use_dual_head=False, use_model_ema=False,
                     gate_mode="const", cls_weight=0.0, nasa_weight=0.0),''',
'''    "AG_const": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                     use_graph=True, use_dual_head=False, use_model_ema=False,
                     use_reliability=True, gate_mode="const", cls_weight=0.0, nasa_weight=0.0),''')
# also add an evidence-mode reference variant
if 'AG_evidence' not in s:
    anchor = '    "AG_ones"'
    ins = '''    "AG_evidence": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                        use_graph=True, use_dual_head=False, use_model_ema=False,
                        use_reliability=True, gate_mode="evidence", cls_weight=0.0, nasa_weight=0.0),
'''
    s = s.replace(anchor, ins + anchor, 1)
p.write_text(s, encoding='utf-8')
import ast; ast.parse(s)
print('patched; use_reliability=True count now:', s.count('use_reliability=True, gate_mode'))
