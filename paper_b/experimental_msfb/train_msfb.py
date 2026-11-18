"""Train MSFB under the frozen Paper B protocol (same loop as DSPT)."""
import argparse
from pathlib import Path

import paper_b.training.train as T
from .model_msfb import build_msfb

_ORIG_BUILD = T._build


def _build(cfg):
    if cfg.get('model') == 'msfb':
        return build_msfb(
            num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
            dropout=cfg['dropout'], max_span=cfg['span'],
            instnorm=cfg['instnorm'], ms_mode=cfg['ms_mode'],
            sa_mode=cfg['sa_mode'], hi_source=cfg['hi_source'],
            hi_mono=cfg['hi_mono'], use_pad_mask=cfg['use_pad_mask'],
            use_cycle=cfg['use_cycle'], cycle_pos=cfg['cycle_pos'],
            n_cond=cfg['n_cond'], legacy_ms_pad=cfg['legacy_ms_pad'],
            legacy_ema=cfg['legacy_ema'], control=cfg.get('control', False))
    return _ORIG_BUILD(cfg)


T._build = _build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--out-root', default=str(Path.home() / 'sprint24/paper_b/results/msfb'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fp32', action='store_true', default=True)
    ap.add_argument('--sa-mode', default='basic')
    ap.add_argument('--ms-mode', default='full')
    ap.add_argument('--hi-source', default='hf')
    ap.add_argument('--tag', default='msfb')
    ap.add_argument('--control', action='store_true',
                    help='capacity control: same params, extra feature channels zeroed')
    a = ap.parse_args()
    cfg = dict(model='msfb', subset=a.subset, seed=a.seed, data_dir=a.data_dir,
               epochs=a.epochs, patience=a.patience, batch=512, lr=1e-3,
               weight_decay=1e-4, d_model=64, layers=2, dropout=0.2,
               mode='dual', ms_mode=a.ms_mode, sa_mode=a.sa_mode,
               hi_source=a.hi_source, hi_mono='none', use_cycle=True,
               cycle_pos=None, fp32=True, use_pad_mask=True, aux_dual=True,
               instnorm=False, tag=a.tag, control=a.control)
    m = T.train_one(cfg, device=a.device, out_root=a.out_root)
    print('DONE', a.subset, a.seed, m.get('val_engine_mean_rmse'), m.get('params'))


if __name__ == '__main__':
    main()
