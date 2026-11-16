#!/usr/bin/env python3
"""Is the E5 branch actually awake? Measured as the perturbation it applies to H.

The last layer of `g` is zero-initialised (E5_PREREG.md §1) so the branch starts as
an exact identity. If it is still near zero after training, the branch never learned
-- and H1's 5..14 spectrum clause then reports "indistinguishable" for a reason that
has nothing to do with the rank constraint being useless.

Definition (E5_PREREG.md §9.6): the branch is dormant iff the relative perturbation
it applies to H is below the float-discriminable level.

    rel = || lr(H) - H || / || H ||

Weight norms are only a proxy and are reported as diagnostics only. Two ways that
proxy misleads, both measured before any E5 arm checkpoint existed:

  * `lr.g.0.weight` is randomly initialised (norm ~4.6), so a max over all
    `lr.g.*.weight` reports a completely untrained model as awake;
  * a large final-layer norm can still be dormant, because the branch scales by
    lam (init 1e-2): ||W_last|| = 1.295 with lam = 1e-2 gives rel = 3.4e-4.

So: hook the branch, measure rel directly.

Usage
-----
    python scripts/audit_e5_activation.py --runs <runs> --data-dir <dir> \
        --seeds 101,202,303
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.sensor_aug.evaluation import _predict  # noqa: E402
from paper_b.training.data import prepare_train  # noqa: E402
from paper_b.training.train import _build  # noqa: E402
from scripts._cells import try_resolve  # noqa: E402

ARMS = ('E5lr4', 'E5lr14', 'E5mlp')
DORMANT_BELOW = 1e-6


class BranchTap:
    """Capture the branch's input and output on the same forward pass."""

    def __init__(self):
        self.h_in = None
        self.h_out = None
        self.handles = []

    def attach(self, net):
        if getattr(net, 'lr', None) is None:
            return False
        self.handles.append(net.lr.register_forward_pre_hook(
            lambda m, inp: setattr(self, 'h_in', inp[0].detach().float())))
        self.handles.append(net.lr.register_forward_hook(
            lambda m, inp, out: setattr(self, 'h_out', out.detach().float())))
        return True

    def close(self):
        for h in self.handles:
            h.remove()

    def rel(self):
        if self.h_in is None or self.h_out is None:
            return None
        num = (self.h_out - self.h_in).norm().item()
        den = self.h_in.norm().item()
        return num / den if den > 0 else None


def last_g_weight(sd):
    keys = [k for k in sd if k.startswith('lr.g.') and k.endswith('.weight')]
    if not keys:
        return None
    key = max(keys, key=lambda k: int(k.split('.')[2]))
    return key, float(sd[key].norm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--seeds', default='101,202,303')
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    args = ap.parse_args()

    runs = Path(args.runs)
    seeds = [int(s) for s in args.seeds.split(',')]
    subsets = args.subsets.split(',')

    print(f'{"arm":7} {"subset":7} {"seed":>5} {"|lam|":>10} {"||W_last||":>11} '
          f'{"rel=||dH||/||H||":>18}  verdict')
    dormant = awake = 0
    for arm in ARMS:
        for subset in subsets:
            for seed in seeds:
                cell = try_resolve(runs, arm, subset, seed, 17)
                if cell is None:
                    continue
                cell = Path(cell)
                cfg = json.loads((cell / 'config.json').read_text())
                net = _build(cfg)
                sd = torch.load(cell / 'best.pt', map_location='cpu', weights_only=True)
                net.load_state_dict(sd)
                net.eval()
                lam = abs(float(sd['lr.lam']))
                wkey, wn = last_g_weight(sd)

                tap = BranchTap()
                if not tap.attach(net):
                    print(f'{arm:7} {subset:7} {seed:>5}  no branch module')
                    continue
                ds = prepare_train(cfg['subset'], Path(args.data_dir), span=cfg['span'],
                                   split_seed=cfg['split_seed'], fold=cfg['fold'],
                                   val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                                   n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                                   legacy_rul=cfg['legacy_rul'],
                                   global_hi=cfg['global_hi'])
                _predict(net, ds['val'], 'cpu', cfg)
                tap.close()
                rel = tap.rel()
                if rel is None:
                    print(f'{arm:7} {subset:7} {seed:>5} {lam:>10.3e} {wn:>11.3e}  '
                          f'{"(no capture)":>18}  UNKNOWN')
                    continue
                is_dormant = rel < DORMANT_BELOW
                dormant += int(is_dormant)
                awake += int(not is_dormant)
                print(f'{arm:7} {subset:7} {seed:>5} {lam:>10.3e} {wn:>11.3e} '
                      f'{rel:>18.3e}  {"DORMANT" if is_dormant else "AWAKE"}')

    print()
    print(f'awake {awake}, dormant {dormant} (dormant threshold rel < {DORMANT_BELOW:g})')
    if dormant:
        print('A dormant cell must be reported as "the branch did not learn" '
              '(E5_PREREG.md §9.6); its H1 spectrum clause is void, and it must not '
              'be written up as "the rank constraint has no effect".')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
