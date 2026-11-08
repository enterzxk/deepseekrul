from pathlib import Path
p=Path('run_full_fold3.py'); s=p.read_text()
old='def build_net(arm, window, graph=False, d_model=64):\n    body = ida.CandidateModel(num_sensors=14, window_size=window, d_model=d_model, num_layers=1,'
new='def build_net(arm, window, graph=False, d_model=64, layers=1):\n    body = ida.CandidateModel(num_sensors=14, window_size=window, d_model=d_model, num_layers=layers,'
assert old in s, 'anchor1'
s=s.replace(old,new)
s=s.replace('    ap.add_argument("--d-model", type=int, default=64)', '    ap.add_argument("--d-model", type=int, default=64)\n    ap.add_argument("--layers", type=int, default=1)')
s=s.replace('net = build_net(arm, args.window, args.graph, args.d_model)', 'net = build_net(arm, args.window, args.graph, args.d_model, args.layers)')
p.write_text(s); print('patched')