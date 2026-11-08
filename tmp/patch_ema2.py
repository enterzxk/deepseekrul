from pathlib import Path
p = Path('/home/user/rulb/scripts/run_dspt.py')
s = p.read_text()
NL = chr(10)
if 'ema_state' in s:
    print('already patched'); raise SystemExit
a1 = "    best, best_state, best_ep = float('inf'), None, -1"
assert s.count(a1) == 1, ('a1', s.count(a1))
s = s.replace(a1, a1 + NL +
  '    ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()}' + NL +
  '    EMA_DECAY = 0.995')
a2 = '            opt.step()'
assert s.count(a2) == 1, ('a2', s.count(a2))
s = s.replace(a2, a2 + NL +
  '            with torch.no_grad():' + NL +
  '                for _k, _v in net.state_dict().items():' + NL +
  '                    if _v.dtype.is_floating_point:' + NL +
  '                        ema_state[_k].mul_(EMA_DECAY).add_(_v.detach(), alpha=1.0-EMA_DECAY)' + NL +
  '                    else:' + NL +
  '                        ema_state[_k].copy_(_v)')
a3 = '    if best_state is not None: net.load_state_dict(best_state)'
assert s.count(a3) == 1, ('a3', s.count(a3))
rep = (a3 + NL +
  '    _bs = {k: t.detach().clone() for k, t in net.state_dict().items()}' + NL +
  '    _vb = evaluate(net, VA, dev)[0]' + NL +
  '    net.load_state_dict(ema_state)' + NL +
  '    _ve = evaluate(net, VA, dev)[0]' + NL +
  '    if _ve < _vb:' + NL +
  '        print("  using EMA weights (%.4f < %.4f)" % (_ve, _vb), flush=True)' + NL +
  '    else:' + NL +
  '        net.load_state_dict(_bs)')
s = s.replace(a3, rep)
p.write_text(s)
import ast; ast.parse(s)
print('patched, syntax ok')