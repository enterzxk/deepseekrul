"""Why does B3 miss the threshold while B2 does not? Measure both heads' scale.

B2 (fixed rate) and B3 (learned rate) share the same increment head and the same
b0, so the difference must be inside ``future_rates``. This probe prints the
pre-activation and output statistics of both heads at init, plus the arrival
time, on one engine. Read-only; no training.
"""
import argparse
import json

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch, train_lifetime_mean
from paper_b.hc.hi_d import DSPT_HC_Structured

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
TMP = ROOT + '/experiments/hc/tmp'


def build(dev, l_mean, K, head_seed=101):
    cfg = dict(C.hc_config('A1', 'FD001', 101))
    bb = cfg['backbone']
    net = DSPT_HC_Structured(
        fixed_rate=False, K=K, cycle_mode='none', num_sensors=14,
        d_model=cfg['d_model'], num_layers=cfg['layers'], dropout=cfg['dropout'],
        max_span=cfg['span'], ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
        hi_source=bb['hi_source'], hi_mono=bb['hi_mono'], use_pad_mask=True,
        n_cond=cfg['n_cond'], l_mean=l_mean, head_seed=head_seed)
    ck = torch.load(TMP + '/wave1_init_A1_s101.pt', map_location='cpu', weights_only=True)
    sd = dict(ck['state_dict'] if 'state_dict' in ck else ck)
    # The init checkpoint is a B0 file, so the damage head is legitimately absent.
    # Everything else must be present: a bare strict=False would also accept a
    # checkpoint that lost the backbone, and the rate scale would then be measured
    # on a random network with nothing to show for it.
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith('head.')]
    if bad or unexpected:
        raise SystemExit('init mismatch: missing=%s unexpected=%s (only head.* may '
                         'be absent from a B0 init)' % (bad, unexpected))
    return net.to(dev).eval()


def z_of(net, eb, i, n, dev):
    x = torch.from_numpy(np.asarray(eb['x'][i, :n], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][i, :n], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][i, :n], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][i, :n], np.int64)).to(dev)
    return net.pooled_feature(x, m, b, torch.zeros(n, device=dev), cd)


def stats(t, name):
    t = t.detach().float()
    return dict(name=name, min=float(t.min()), median=float(t.median()),
                max=float(t.max()), mean=float(t.mean()), std=float(t.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--K', type=int, default=600)
    ap.add_argument('--engine', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    lm = train_lifetime_mean(a.subset, DATA)['mean']
    net = build(dev, lm, a.K)
    eb = engine_batch(a.subset, DATA, C.SPAN, C.VAL_CUTS, split='val')
    n = int(eb['lengths'][a.engine])
    with torch.no_grad():
        z = z_of(net, eb, a.engine, n, dev)
        delta = net.head.increment(z.unsqueeze(0)).reshape(-1)
        D = torch.cumsum(delta, 0)
        nu = net.current_rate(delta.unsqueeze(0)).reshape(-1)
        out = [stats(delta, 'delta_out'),
               stats(D, 'D_along'),
               stats(nu, 'nu_ema')]
        # increment head internals
        pre_d = net.head.delta[2](net.head.delta[1](net.head.delta[0](z)))
        out += [stats(pre_d, 'delta_pre_softplus')]
        # rate head internals, one prediction point in the middle of life
        t = n // 2
        K = net.K
        k = torch.arange(1, K + 1, device=dev, dtype=z.dtype)
        kk = torch.stack([k, torch.log1p(k)], -1).unsqueeze(0)
        base = torch.cat([z[t].unsqueeze(0), D[t].reshape(1, 1), nu[t].reshape(1, 1)], -1)
        inp = torch.cat([base.unsqueeze(1).expand(-1, K, -1), kk], -1)
        h1 = net.head.rate[0](inp)
        a1 = net.head.rate[1](h1)
        pre_r = net.head.rate[2](a1)
        out_r = torch.nn.functional.softplus(pre_r)
        out += [stats(inp, 'rate_input_all'), stats(inp[..., -2], 'rate_input_k'),
                stats(inp[..., -1], 'rate_input_log1pk'),
                stats(h1, 'rate_hidden1_pre'), stats(pre_r, 'rate_pre_softplus'),
                stats(out_r, 'rate_out')]
        T = net.arrival(D[t].reshape(1), nu[t].reshape(1), z[t].unsqueeze(0))
        out += [dict(name='arrival_at_t', t=int(t), D=float(D[t]), nu=float(nu[t]),
                     T=float(T), K=K)]
    print(json.dumps(dict(subset=a.subset, K=a.K, engine=int(eb['engine_id'][a.engine]),
                          n=n, l_mean=lm, b0=net.head.init_b0,
                          stats=out), indent=1))


if __name__ == '__main__':
    main()
