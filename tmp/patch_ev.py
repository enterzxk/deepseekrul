from pathlib import Path
NL = chr(10)
p = Path('/home/user/rulb/scripts/eval_dspt.py')
s = p.read_text()
if "'--instnorm'" in s:
    print('already')
else:
    a = "    ap.add_argument('--tta', type=int, default=1)"
    if a not in s:
        a = "    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)"
    assert s.count(a) == 1, ('anchor', s.count(a))
    s = s.replace(a, a + NL + "    ap.add_argument('--instnorm', action='store_true')")
    p.write_text(s)
    import ast; ast.parse(s)
    print('patched')