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
import math
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


def check_a3p_equivalence(seed=0):
    """A3p must be the frozen model, not merely similar to it.

    Builds ``DSPT_HC_Frozen`` and the frozen ``build_dspt(use_cycle=True)`` with
    identical shapes, copies one state dict into the other, feeds the same
    batch and the same cycle scalar, and compares the predictions. A3p's whole
    job is to be the pre-norm reference, so any difference here means the
    injection-point control is measuring something else.
    """
    from paper_b.models.dspt import build_dspt
    from paper_b.hc.model_hc import DSPT_HC_Frozen

    torch.manual_seed(seed)
    common = dict(num_sensors=NS, d_model=32, num_layers=1, dropout=0.0,
                  max_span=SPAN, ms_mode='full', sa_mode='basic',
                  hi_source='none', hi_mono='none', use_pad_mask=True, n_cond=3)
    ref = build_dspt(use_cycle=True, **common).eval()
    hc = DSPT_HC_Frozen(**common).eval()
    missing, unexpected = hc.load_state_dict(ref.state_dict(), strict=True) \
        if False else (None, None)
    hc.load_state_dict(ref.state_dict(), strict=True)

    g = torch.Generator().manual_seed(seed + 1)
    n = 8
    x = torch.randn(n, SPAN, NS, generator=g)
    mask = torch.ones(n, SPAN)
    base = torch.randn(n, NS, generator=g)
    cycle_cycles = torch.rand(n, generator=g) * 250.0   # cycles, 0..250
    cond = torch.randint(0, 3, (n,), generator=g)

    with torch.no_grad():
        p_ref = ref(x, mask=mask, baseline=base,
                    cycle=torch.clamp(cycle_cycles, 0, 250) / 250.0, cond=cond)['prediction']
        p_hc = hc(x, mask=mask, baseline=base, cycle=cycle_cycles, cond=cond)['prediction']
    diff = float((p_ref - p_hc).abs().max())
    print(f'A3p vs frozen DSPT: max |pred diff| = {diff:.3e}  params '
          f'{sum(p.numel() for p in hc.parameters())} vs '
          f'{sum(p.numel() for p in ref.parameters())}')
    return diff < 1e-6


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

        # --- A3a / A3p control, asserted before any training -----------------
        # Shared init: identical backbone tensors and an identical Fourier term.
        # The full encode is expected to differ (pre-norm vs post-norm); that
        # difference IS the treatment, so it is printed, not asserted to zero.
        from paper_b.hc.make_control_init import build_pair
        a3p, a3a, sd_p, sd_a, n_shared = build_pair(101, 'FD001', SPAN)
        a3p.eval(); a3a.eval()
        backbone_diff = max(float((sd_p[k] - sd_a[k]).abs().max())
                            for k in sd_p if k in sd_a)
        x2 = torch.randn(2, SPAN, 14)
        m2 = torch.ones(2, SPAN)
        b2 = x2[:, 0]
        cyc2 = torch.tensor([40.0, 250.0])
        ang = (2.0 * math.pi) * cyc2.unsqueeze(1) / a3p.pe_periods.view(1, -1)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        with torch.no_grad():
            term_p = a3p.pe_proj(pe)
            term_a = a3a.cycle_enc(cyc2)
            full_p = a3p.encode(x2, b2, m2, cyc2, None)
            full_a = a3a.encode(x2, b2, m2, cyc2, None)
        term_diff = float((term_p - term_a).abs().max())
        full_diff = float((full_p - full_a).abs().max())
        print(f'A3p/A3a shared tensors = {n_shared}, backbone max|diff| = {backbone_diff:.3e}')
        print(f'A3p/A3a Fourier term max|diff| = {term_diff:.3e}')
        print(f'A3p/A3a full encode max|diff| = {full_diff:.3e} (treatment, expected > 0)')

        ok = (loss_diff < 1e-6 and max_diff < 1e-6 and set(sd_f) == set(sd_h)
              and backbone_diff < 1e-6 and term_diff < 1e-6)
        print('VERIFY_LOOP', 'PASS' if ok else 'FAIL')
        # A3p must be the frozen model, not merely similar to it.
        eq = check_a3p_equivalence()
        print('A3P_EQUIVALENCE', 'PASS' if eq else 'FAIL')
        return 0 if (ok and eq) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
