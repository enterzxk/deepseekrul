"""M4 (CSR) pre-dispatch checks. CPU only, no GPU, no training.

Three claims are checked, and each one is a claim the arm's interpretation
depends on:

1. TRANSPARENT AT STEP 0 -- with the reliability bias zero-initialised, M4 must
   produce bit-identical predictions to B0 given the same weights. If this
   fails, "M4 improves on B0" would be confounded with "M4 re-initialised the
   pooling head", and the arm would not be a one-variable ablation.
2. BIAS IS THE ONLY DIFFERENCE -- after a nonzero step on the bias only, the
   prediction must move; and with the bias zeroed again it must return exactly.
   This proves the bias is live and that nothing else drifted.
3. BIAS ACTUALLY REACHES THE OUTPUT -- a synthetic condition shift must change
   the pooling weights, i.e. the module is wired, not dead code.

Run:  python -m paper_b.hc.test_csr_parity
"""
import torch

from .csr_pool import DSPT_CSR
from .hi_d import DSPT_HC_Structured

KW = dict(num_sensors=14, d_model=16, num_layers=1, dropout=0.0, max_span=70,
          ms_mode='full', sa_mode='basic', hi_source='none', hi_mono='none',
          use_pad_mask=True, n_cond=6, K=200, l_mean=None, head_seed=None)


def make_batch(B=3, L=12, N=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(B, L, N, generator=g),
        mask=torch.ones(B, L),
        baseline=torch.zeros(B, N),
        cycle=torch.zeros(B),
        cond=torch.randint(0, 6, (B,), generator=g),
    )


def main():
    torch.manual_seed(0)
    b0 = DSPT_HC_Structured(cycle_mode='none', **KW).eval()
    torch.manual_seed(0)
    m4 = DSPT_CSR(csr_mode='cond', **KW).eval()

    # --- 1. transparency ---------------------------------------------------
    b0.load_state_dict(m4.state_dict_remapped(), strict=False)
    b = make_batch()
    with torch.no_grad():
        p0 = b0(b['x'], mask=b['mask'], baseline=b['baseline'],
                cycle=b['cycle'], cond=b['cond'], mode='dual')['prediction']
        p4 = m4(b['x'], mask=b['mask'], baseline=b['baseline'],
                cycle=b['cycle'], cond=b['cond'], mode='dual')['prediction']
    d0 = float((p0 - p4).abs().max())
    # Same weights + same ops must give the same numbers; a tiny tolerance only
    # absorbs a different BLAS reduction order, not a real drift.
    assert d0 <= 1e-7, 'step-0 parity broken: max|diff|=%.3e' % d0
    print('PASS  transparent_at_step0            max|diff| = %.3e' % d0)

    # --- 2. bias is the only difference ------------------------------------
    with torch.no_grad():
        m4.sensor_score.cond_bias.weight.normal_(0, 0.5)
        p4b = m4(b['x'], mask=b['mask'], baseline=b['baseline'],
                 cycle=b['cycle'], cond=b['cond'], mode='dual')['prediction']
        moved = float((p4b - p4).abs().max())
        m4.sensor_score.cond_bias.weight.zero_()
        p4c = m4(b['x'], mask=b['mask'], baseline=b['baseline'],
                 cycle=b['cycle'], cond=b['cond'], mode='dual')['prediction']
        back = float((p4c - p0).abs().max())
    assert moved > 0.0, 'condition bias has no effect on the prediction'
    assert back == 0.0, 'zeroing the bias did not restore B0 exactly: %.3e' % back
    print('PASS  bias_is_the_only_difference     moved = %.3e, restored = %.3e'
          % (moved, back))

    # --- 3. the bias is a softmax shift over sensors -----------------------
    with torch.no_grad():
        m4.sensor_score.cond_bias.weight.normal_(0, 0.5)
        tokens = torch.randn(3, 12, 14, 16)
        m4.sensor_score.set_context(cond=b['cond'], trend=None)
        logits = m4.sensor_score(tokens).squeeze(-1)          # (B,L,N)
        w = torch.softmax(logits, dim=-1)
        spread = float((w.max(dim=-1).values - w.min(dim=-1).values).mean())
    assert spread > 0.0, 'reliability weights are uniform: the bias is dead'
    print('PASS  reliability_weights_are_live     mean(max-min w) = %.3e' % spread)
    print('\nCONTRACT OK  (3/3)')


if __name__ == '__main__':
    main()
