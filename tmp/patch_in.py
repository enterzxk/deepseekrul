from pathlib import Path
p = Path('/home/user/rulb/src/candidates/dspt.py')
s = p.read_text()
NL = chr(10)
if 'instnorm' in s:
    print('already patched'); raise SystemExit

# 1) build() signature
a = 'def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2):'
assert s.count(a) == 1
s = s.replace(a, 'def build(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, instnorm=False):')
a2 = '    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers, dropout=dropout)'
assert s.count(a2) == 1
s = s.replace(a2, '    return DSPT(num_sensors=num_sensors, d_model=d_model, num_layers=num_layers, dropout=dropout, instnorm=instnorm)')

# 2) __init__ signature + sensor_in
a3 = '                 pe_periods=(25.0, 50.0, 125.0, 250.0), max_span=MAX_SPAN):'
assert s.count(a3) == 1
s = s.replace(a3, '                 pe_periods=(25.0, 50.0, 125.0, 250.0), max_span=MAX_SPAN, instnorm=False):')
a4 = '        self.sensor_in = nn.Linear(3, d_model)'
assert s.count(a4) == 1
s = s.replace(a4, '        self.instnorm = bool(instnorm)' + NL + '        self.n_feat = 4 if self.instnorm else 3' + NL + '        self.sensor_in = nn.Linear(self.n_feat, d_model)')

# 3) _features
old = ('        trend = torch.stack(out, dim=1)' + NL +
       '        return torch.stack([x, dev, trend], dim=-1)  # (B,L,N,3)')
assert s.count(old) == 1, 'features anchor'
new = ('        trend = torch.stack(out, dim=1)' + NL +
       '        if not self.instnorm:' + NL +
       '            return torch.stack([x, dev, trend], dim=-1)  # (B,L,N,3)' + NL +
       '        # per-engine instance normalisation: remove this engine scale as well as its offset' + NL +
       '        m = mask.unsqueeze(-1).to(x.dtype)                       # (B,L,1)' + NL +
       '        den = m.sum(dim=1).clamp_min(1.0)                        # (B,1)' + NL +
       '        mu = (x * m).sum(dim=1) / den                            # (B,N)' + NL +
       '        var = (((x - mu.unsqueeze(1)) ** 2) * m).sum(dim=1) / den' + NL +
       '        sc = var.sqrt().clamp_min(1e-3).unsqueeze(1)             # (B,1,N)' + NL +
       '        norm = dev / sc' + NL +
       '        return torch.stack([x, dev, norm, trend], dim=-1)  # (B,L,N,4)')
s = s.replace(old, new)
p.write_text(s)
import ast; ast.parse(s)
print('dspt patched')