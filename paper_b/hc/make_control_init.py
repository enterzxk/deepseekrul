"""Shared initialisation for the A3a / A3p injection-point control.

A3p builds ``DSPT`` with ``use_cycle=True``, so ``pe_proj`` (Linear(8,64)) is
created at ``dspt.py:119`` -- before layers / sensor_attn / cond_emb / reg_head.
A3a builds the same backbone with ``use_cycle=False`` and creates
``cycle_enc.proj`` last, so even with the same seed the two runs would start
from *different* backbone weights and the A3a-A3p difference would mix the
treatment (injection point) with initialisation noise.

This module removes that confound: build A3p first, copy every shared tensor
into A3a, and copy ``pe_proj.{weight,bias}`` into ``cycle_enc.proj`` (the same
``Linear(8,64)`` applied to ``cat([sin, cos])``). The two runs then differ only
in where the Fourier term enters the encoder.

Usage::

    python -m paper_b.hc.make_control_init --out-dir experiments/hc/tmp
"""
import argparse
from pathlib import Path

import torch

from .config_hc import hc_config
from .model_hc import build_hc


def build_pair(seed=101, subset='FD001', span=70):
    """Return (a3p, a3a, sd_p, sd_a, n_shared) with a shared backbone."""
    torch.manual_seed(seed)
    a3p = build_hc(hc_config('A3p', subset, seed, span=span))
    sd_p = {k: v.clone() for k, v in a3p.state_dict().items()}
    torch.manual_seed(seed)
    a3a = build_hc(hc_config('A3a', subset, seed, span=span))
    sd_a = {k: v.clone() for k, v in a3a.state_dict().items()}
    n_shared = 0
    for k, v in sd_p.items():
        if k in sd_a and sd_a[k].shape == v.shape:
            sd_a[k] = v.clone()
            n_shared += 1
    for src, dst in (('pe_proj.weight', 'cycle_enc.proj.weight'),
                     ('pe_proj.bias', 'cycle_enc.proj.bias')):
        if src in sd_p and dst in sd_a:
            sd_a[dst] = sd_p[src].clone()
    a3a.load_state_dict(sd_a, strict=True)
    return a3p, a3a, sd_p, sd_a, n_shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--span', type=int, default=70)
    ap.add_argument('--out-dir', default='experiments/hc/tmp')
    a = ap.parse_args()
    a3p, a3a, sd_p, sd_a, n_shared = build_pair(a.seed, a.subset, a.span)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p1 = out / f'a3p_init_s{a.seed}.pt'
    p2 = out / f'a3a_init_s{a.seed}.pt'
    torch.save(sd_p, p1)
    torch.save(sd_a, p2)
    print('shared backbone tensors:', n_shared)
    print('wrote', p1)
    print('wrote', p2)


if __name__ == '__main__':
    main()
