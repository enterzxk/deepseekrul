from pathlib import Path
NL = chr(10)

# --- dspt.py: build() and DSPT() take max_span ---
p = Path('/home/user/rulb/src/candidates/dspt.py')
s = p.read_text()
if 'span=' not in s.split('def build')[1][:200]:
    a = 'def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, instnorm=False):'
    assert s.count(a) == 1
    s = s.replace(a, 'def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, instnorm=False, max_span=MAX_SPAN):')
    a2 = '    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers, dropout=dropout, instnorm=instnorm)'
    assert s.count(a2) == 1
    s = s.replace(a2, '    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers, dropout=dropout, instnorm=instnorm, max_span=max_span)')
    p.write_text(s)
    import ast; ast.parse(s); print('dspt ok')

# --- run_dspt.py: --span flag, pass through ---
p = Path('/home/user/rulb/scripts/run_dspt.py')
s = p.read_text()
if "'--span'" not in s:
    a = "    ap.add_argument('--wshift', action='store_true')"
    assert s.count(a) == 1
    s = s.replace(a, a + NL + "    ap.add_argument('--span', type=int, default=100)")
    a2 = 'def windows_for(frame, values, labels, pop_ref, tau=2.0, max_rul=125.0):'
    assert s.count(a2) == 1
    s = s.replace(a2, 'def windows_for(frame, values, labels, pop_ref, tau=2.0, max_rul=125.0, span=MAX_SPAN):')
    s = s.replace('TR = windows_for(tr, tr_s, lt, pop); VA = windows_for(va, va_s, lv, pop)',
                  'TR = windows_for(tr, tr_s, lt, pop, span=a.span); VA = windows_for(va, va_s, lv, pop, span=a.span)')
    s = s.replace('net = build(14, a.d_model, a.layers, a.dropout, instnorm=getattr(a, \'instnorm\', False)).to(dev)',
                  'net = build(14, a.d_model, a.layers, a.dropout, instnorm=getattr(a, \'instnorm\', False), max_span=a.span).to(dev)')
    # MAX_SPAN usage inside windows_for -> span
    s = s.replace('            end = t + 1; m = min(end, MAX_SPAN)', '            end = t + 1; m = min(end, span)')
    s = s.replace('            pad = MAX_SPAN - m', '            pad = span - m')
    s = s.replace('            x = np.zeros((MAX_SPAN, seq.shape[1]), np.float32)', '            x = np.zeros((span, seq.shape[1]), np.float32)')
    s = s.replace('            mask = np.zeros(MAX_SPAN, np.float32); mask[pad:] = 1.0', '            mask = np.zeros(span, np.float32); mask[pad:] = 1.0')
    p.write_text(s)
    import ast; ast.parse(s); print('run ok')