from pathlib import Path
p = Path('/home/user/rulb/scripts/run_dspt.py')
s = p.read_text()
if 'ema_state' in s:
    print('already patched'); raise SystemExit
# 1) init ema right before the epoch loop
anchor = '            opt.step()'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + chr(10) +
    '            with torch.no_grad():' + chr(10) +
    '                for _k, _v in net.state_dict().items():' + chr(10) +
    '                    if _v.dtype.is_floating_point:' + chr(10) +
    '                        ema_state[_k].mul_(EMA_DECAY).add_(_v.detach(), alpha=1.0-EMA_DECAY)' + chr(10) +
    '                    else:' + chr(10) +
    '                        ema_state[_k].copy_(_v)')
# 2) create ema_state before training starts
a2 = '    best_state = None'
assert s.count(a2) == 1, s.count(a2)
s = s.replace(a2, a2 + chr(10) +
    '    ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()}' + chr(10) +
    '    EMA_DECAY = 0.995')
# 3) after loading best_state, also try EMA and keep the better one on validation
a3 = '    if best_state is not None: net.load_state_dict(best_state)'
assert s.count(a3) == 1
s = s.replace(a3, a3 + chr(10) +
    '    net.load_state_dict(best_state)' + chr(10) +
    '    _vr_best = evaluate(net, VA, dev)[0]' + chr(10) +
    '    net.load_state_dict(ema_state)' + chr(10) +
    '    _vr_ema = evaluate(net, VA, dev)[0]' + chr(10) +
    '    if _vr_ema < _vr_best:' + chr(10) +
    '        print(\"  using EMA weights (%.4f < %.4f)\" % (_vr_ema, _vr_best), flush=True)' + chr(10) +
    '        used_ema = True' + chr(10) +
    '    else:' + chr(10) +
    '        net.load_state_dict(best_state)' + chr(10) +
    '        used_ema = False')
p.write_text(s)
print('patched ok')