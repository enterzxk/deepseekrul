"""Trained-damage distribution: does the model naturally keep D < 1 while alive?

This is the discriminator between two readings of M7's ``alive -> 0``:

  A  the SCR penalty was overwhelmed by the terminal anchor and the smoothness
     term, so the constraint had nothing left to act on;
  B  training made D correct, and a correct D does not cross 1 during life, so
     the constraint was satisfied rather than suppressed.

Both show alive -> 0. The way to tell them apart is to look at an arm WITHOUT
the SCR term: if its trained damage also stays below 1 during life, then the
loss alone produces that property and SCR is redundant (B). If it crosses and
M7 does not, SCR is doing work (A).

This probe runs on any checkpoint and reports, per engine and in aggregate, the
fraction of ALIVE cycles (r > 0) with D > 1, plus the distribution of D.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import build_d, forward_engine, trainable

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'
SHARED = ROOT + '/experiments/hc/d'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = torch.device(a.device)
    sh = json.loads(open('%s/%s_shared.json' % (SHARED, a.subset)).read())
    net, _ = build_d(a.arm, a.subset, int(sh['K']), float(sh['l_mean']),
                     int(sh['head_seed']), dev, INIT)
    trainable(net, a.arm)
    ck = Path(a.run_dir) / 'ckpt' / 'best.pt'
    if not ck.exists():
        raise SystemExit('no checkpoint: %s' % ck)
    net.load_state_dict(torch.load(ck, map_location=dev), strict=False)
    net.eval()

    eb = engine_batch(a.subset, DATA, C.SPAN, C.VAL_CUTS, split=a.split)
    viol_eng, alive_all, over_all, dend, dmax = 0, 0, 0, [], []
    per = []
    with torch.no_grad():
        for i in range(len(eb['lengths'])):
            n = int(eb['lengths'][i])
            fo = forward_engine(net, a.arm, eb, i, n, dev, int(sh['K']))
            if 'D' not in fo:
                raise SystemExit('%s has no damage path' % a.arm)
            D = fo['D'].cpu().numpy()
            r = fo['r'].cpu().numpy()
            alive = r > 0
            n_alive = int(alive.sum())
            n_over = int((D[alive] > 1.0).sum())
            alive_all += n_alive
            over_all += n_over
            dend.append(float(D[-1]))
            dmax.append(float(D.max()))
            if n_over:
                viol_eng += 1
                per.append(dict(engine=int(eb['engine_id'][i]), n=n,
                                alive_cycles=n_alive, over=n_over,
                                max_D=float(D.max()),
                                max_over=float(D[alive].max() - 1.0)))
    rec = dict(arm=a.arm, subset=a.subset, split=a.split, run_dir=a.run_dir,
               n_engines=int(len(eb['lengths'])),
               engines_with_violation=viol_eng,
               frac_engines_with_violation=viol_eng / len(eb['lengths']),
               alive_cycles=alive_all, alive_cycles_over=over_all,
               frac_alive_cycles_over=(over_all / alive_all if alive_all else None),
               D_end=dict(median=float(np.median(dend)), min=float(min(dend)),
                          max=float(max(dend))),
               D_max=dict(median=float(np.median(dmax)), max=float(max(dmax))),
               worst=per[:8])
    print(json.dumps(rec, indent=1))
    out = Path(ROOT) / 'experiments/hc/d' / ('scr_%s_%s_%s.json'
                                             % (a.arm, a.subset, a.split))
    out.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('record -> %s' % out)


if __name__ == '__main__':
    main()
