"""Numerical check that the hc training loop matches the frozen trainer.

B4 of the audit forbids keeping an unverified "already verified" claim, so this
file is the verification. It feeds the *same synthetic tensors* to
``paper_b.training.train.train_one`` and to ``paper_b.hc.train_hc.train_hc``
for the one variant whose architecture is identical to the frozen model (A1:
sensor-only, no cycle, no HI head), then compares the per-epoch losses and the
final parameters.

What it proves: the transcription of the loop (optimiser, schedule, loss terms,
gradient clipping, batch order, early-stopping rule) is faithful.
What it does not prove: that the cycle variants behave like anything else —
they differ by construction.

Run: ``python -m paper_b.hc.verify_loop``
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

N_ENG, CYCLES, SPAN, NS = 6, 12, 8, 14
VAL_CUTS = (0.4, 0.55, 0.7, 0.85, 0.95)


def synth(seed=0):
    rng = np.random.default_rng(seed)
    n_tr = N_ENG * CYCLES
    tr = dict(
        x=rng.normal(size=(n_tr, SPAN, NS)).astype(np.float32),
        mask=np.ones((n_tr, SPAN), np.float32),
        baseline=rng.normal(size=(n_tr, NS)).astype(np.float32),
        cycle=rng.uniform(0, 1, n_tr).astype(np.float32),
        cycle_raw=rng.uniform(1, 250, n_tr).astype(np.float32),
        cond=rng.integers(0, 3, n_tr).astype(np.int64),
        hi=rng.uniform(0, 1, n_tr).astype(np.float32),
        y=rng.uniform(0, 125, n_tr).astype(np.float32),
    )
    n_va = N_ENG * len(VAL_CUTS)
    va = dict(
        x=rng.normal(size=(n_va, SPAN, NS)).astype(np.float32),
        mask=np.ones((n_va, SPAN), np.float32),
        baseline=rng.normal(size=(n_va, NS)).astype(np.float32),
        cycle=rng.uniform(0, 1, n_va).astype(np.float32),
        cycle_raw=rng.uniform(1, 250, n_va).astype(np.float32),
        cond=rng.integers(0, 3, n_va).astype(np.int64),
        y=rng.uniform(0, 125, n_va).astype(np.float32),
        unit=np.repeat(np.arange(1, N_ENG + 1), len(VAL_CUTS)).astype(np.int64),
        cut=np.tile(np.arange(5, 5 + len(VAL_CUTS)), N_ENG).astype(np.int64),
    )
    return dict(train=tr, val=va,
                train_units=list(range(1, N_ENG + 1)),
                val_units=list(range(1, N_ENG + 1)))


def main():
    from paper_b.training import train as FT
    from paper_b.hc import data_hc as DH
    from paper_b.hc import train_hc as HT
    from paper_b.hc.config_hc import hc_config

    ds = synth(0)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tmp = Path(tempfile.mkdtemp(prefix='verify_loop_'))
    try:
        # --- frozen trainer, sensor-only DSPT -------------------------------
        FT.prepare_train = lambda *a, **k: ds
        fcfg = dict(model='dspt', subset='FD001', seed=101, span=SPAN,
                    epochs=2, patience=99, batch=16, lr=1e-3, weight_decay=1e-4,
                    d_model=32, layers=1, dropout=0.0, ms_mode='full',
                    sa_mode='basic', hi_source='none', hi_mono='none',
                    hi_lambda=0.0, hi_mono_coef=0.0, use_cycle=False,
                    aux_dual=True, n_cond=3, tag='verify', data_dir='x',
                    cuts=VAL_CUTS)
        FT.train_one(fcfg, device=dev, out_root=tmp / 'frozen')
        sd_f = torch.load(tmp / 'frozen' / 'verify' / 'FD001' / 's101' / 'model.pt',
                          map_location='cpu', weights_only=True)

        # --- hc trainer, same architecture (A1) -----------------------------
        DH.build = lambda *a, **k: ds
        cfg = hc_config('A1', 'FD001', 101, span=SPAN)
        cfg.update(epochs=2, patience=99, batch=16, lr=1e-3, weight_decay=1e-4,
                   d_model=32, layers=1, dropout=0.0, data_dir='x',
                   aux_dual=True, n_cond=3,
                   budget_override='verify_loop: 2 epochs on synthetic data')
        HT.train_hc(cfg, device=dev, out_dir=tmp / 'hc', log=lambda *a: None)
        sd_h = torch.load(tmp / 'hc' / 'ckpt' / 'best.pt', map_location='cpu',
                          weights_only=True)

        h_f = [h['train_loss'] for h in json.loads(
            (tmp / 'frozen' / 'verify' / 'FD001' / 's101' / 'history.json').read_text())]
        h_h = [h['train_loss'] for h in json.loads(
            (tmp / 'hc' / 'history.json').read_text())]

        print(f'frozen epoch losses: {[round(x, 6) for x in h_f]}')
        print(f'hc     epoch losses: {[round(x, 6) for x in h_h]}')
        keys = sorted(set(sd_f) & set(sd_h))
        print(f'param tensors compared: {len(keys)} (frozen {len(sd_f)}, hc {len(sd_h)})')
        max_diff = 0.0
        worst = None
        for k in keys:
            d = float((sd_f[k].float() - sd_h[k].float()).abs().max())
            if d > max_diff:
                max_diff, worst = d, k
        loss_diff = (max(abs(a - b) for a, b in zip(h_f, h_h))
                     if len(h_f) == len(h_h) else float('inf'))
        print(f'max |loss diff| = {loss_diff:.3e}')
        print(f'max |param diff| = {max_diff:.3e}  (worst: {worst})')
        ok = loss_diff < 1e-6 and max_diff < 1e-6 and set(sd_f) == set(sd_h)
        print('VERIFY_LOOP', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
