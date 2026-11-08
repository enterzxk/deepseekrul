from pathlib import Path
NL = chr(10)
for name, argline, buildline in [
    ('scripts/run_dspt.py',
     "    ap.add_argument('--mode', default='dual', choices=['dual','short','long','repeat','slot'])",
     'net = build(14, a.d_model, a.layers, a.dropout).to(dev)'),
    ('scripts/eval_dspt.py', None, 'net = build(14, a.d_model, a.layers, 0.2)'),
]:
    p = Path('/home/user/rulb') / name
    s = p.read_text()
    if 'instnorm' in s:
        print(name, 'already'); continue
    if argline:
        assert s.count(argline) == 1, (name, 'arg')
        s = s.replace(argline, argline + NL + "    ap.add_argument('--instnorm', action='store_true')")
    assert s.count(buildline) == 1, (name, 'build')
    new = buildline.replace('a.dropout)', 'a.dropout, instnorm=getattr(a, \'instnorm\', False))').replace('0.2)', '0.2, instnorm=getattr(a, \'instnorm\', False))')
    s = s.replace(buildline, new)
    p.write_text(s)
    import ast; ast.parse(s)
    print(name, 'patched')