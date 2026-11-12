"""D4.5 at initialization, every subset, every arm that has auxiliary terms.

The one-time lambda reweighting the spec allows must be decided from the INIT
state, not from a post-training epoch: after one epoch the smoothness gradient
has already collapsed by three orders of magnitude, so a c_S fitted there would
be fitting the wrong object. This probe measures only the init state -- it never
optimises a step.

Reports both readings (see train_d.term_grads for why both):
  unweighted  ||grad L_i||            D4.5's literal wording
  weighted    ||grad lambda_i L_i||   what the optimiser actually sees
"""
import argparse
import json
from pathlib import Path

import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import build_d, term_grads, trainable

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'
SHARED = ROOT + '/experiments/hc/d'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    ap.add_argument('--arms', default='B1,B2,B3')
    ap.add_argument('--engines', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = []
    for subset in [s for s in a.subsets.split(',') if s]:
        sh = json.loads((Path(SHARED) / ('%s_shared.json' % subset)).read_text())
        lam = float(C.RUL_CAP) ** 2
        ebt = engine_batch(subset, DATA, C.SPAN, C.VAL_CUTS, split='train')
        for arm in [s for s in a.arms.split(',') if s]:
            net, _ = build_d(arm, subset, int(sh['K']), float(sh['l_mean']),
                             int(sh['head_seed']), dev, INIT)
            trainable(net, arm)
            net.eval()  # dropout off: this is the init, not a sample of it
            tg = term_grads(net, arm, ebt, dev, int(sh['K']), float(C.RUL_CAP),
                            float(sh['s0']['used']), lam, lam, n_eng=a.engines)
            w, u = tg['weighted'], tg['unweighted']
            mx = max(w.values())
            rec = dict(subset=subset, arm=arm, lam=lam,
                       s0=float(sh['s0']['used']),
                       unweighted=u, weighted=w,
                       ratio={k: v / mx for k, v in w.items()},
                       implied_c={k: (w['rul'] / v if v > 0 else None)
                                  for k, v in w.items()})
            out.append(rec)
            print('%-6s %s  ratio rul=%.3e term=%s smooth=%s   implied_c_term=%s '
                  'implied_c_smooth=%s'
                  % (subset, arm, rec['ratio']['rul'],
                     ('%.3e' % rec['ratio'].get('terminal')) if 'terminal' in rec['ratio'] else '-',
                     ('%.3e' % rec['ratio'].get('smooth')) if 'smooth' in rec['ratio'] else '-',
                     ('%.3f' % rec['implied_c'].get('terminal')) if rec['implied_c'].get('terminal') else '-',
                     ('%.4f' % rec['implied_c'].get('smooth')) if rec['implied_c'].get('smooth') else '-'))
    p = Path(SHARED) / 'd45_init.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('\nrecord -> %s' % p)


if __name__ == '__main__':
    main()
