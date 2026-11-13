"""M4 (CSR) pre-dispatch gate -- goes through ``build_d``, the REAL entry point.

Why this file was rewritten
---------------------------
The first version built ``DSPT_CSR`` directly and passed 3/3, while the real run
died instantly with

    TypeError: DSPT_HC_Structured.__init__() got multiple values for keyword
               argument 'cycle_mode'

``build_d`` puts ``cycle_mode='none'`` into its kw dict, so a second explicit
``cycle_mode=`` raised. The gate tested a private construction path and the
dispatch used another -- a gate that passes while the real path is broken is
worse than no gate. Everything below therefore goes through ``build_d``.

What is checked
---------------
1. BOTH ARMS BUILD through ``build_d`` (the path the runner uses).
2. TRANSPARENT AT STEP 0 -- B0 and M4 are built independently from the SAME
   shared init, so their predictions must agree while ``cond_bias`` is zero.
3. BIAS IS THE ONLY DIFFERENCE -- perturbing ``cond_bias`` must move the
   prediction, and zeroing it must return exactly to B0.
4. THE BIAS IS ZERO-INITIALISED -- it must appear in ``build_d``'s ``missing``
   list, which is the proof that the random init never touched it.

Run (on 3090, where the init file exists):

    python -m paper_b.hc.test_csr_parity \
        --init-from experiments/hc/tmp/wave1_init_A1_s101.pt \
        --subset FD001 --device cpu
"""
import argparse

import torch

from .train_d import build_d, ALL_ARMS, CSR_ARMS


def make_batch(subset, seed=0, B=2, L=10):
    from . import config_hc as C
    cfg = C.hc_config('A1', subset, seed)
    g = torch.Generator().manual_seed(seed)
    n_sensors, span = 14, int(cfg['span'])
    return dict(
        x=torch.randn(B, L, n_sensors, generator=g),
        mask=torch.ones(B, L),
        baseline=torch.zeros(B, n_sensors),
        cycle=torch.zeros(B),
        cond=torch.randint(0, int(cfg['n_cond']), (B,), generator=g),
    )


def pred(net, b):
    return net(b['x'], mask=b['mask'], baseline=b['baseline'],
               cycle=b['cycle'], cond=b['cond'], mode='dual')['prediction']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-from', required=True,
                    help='shared wave1 init .pt (both arms load the same file)')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tol', type=float, default=1e-6)
    a = ap.parse_args()

    # --- 0. registration, checked separately from construction -------------
    # build_d does NOT validate arm membership: an unregistered name silently
    # builds a plain structured model and returns. Only argparse rejects it. So
    # "build_d('M4') works" proves nothing about dispatchability -- both checks
    # are required, and this one has to come first.
    assert 'M4' in ALL_ARMS, "M4 missing from ALL_ARMS: argparse will reject --arm M4"
    assert 'M4' in CSR_ARMS, "M4 missing from CSR_ARMS: build_d will silently build B0"
    print('PASS  registered_in_ALL_ARMS        M4 in ALL_ARMS and CSR_ARMS')

    # --- 1. both arms build through the real entry point -------------------
    b0, _ = build_d('B0', a.subset, 200, None, None, a.device, a.init_from)
    m4, missing = build_d('M4', a.subset, 200, None, None, a.device, a.init_from)
    # A silent B0 fallback would build cleanly and even pass the parity test
    # below -- assert the class, not just "it built".
    assert type(m4).__name__ == 'DSPT_CSR', (
        'build_d("M4") returned %s, not DSPT_CSR: the arm fell back to the '
        'plain backbone' % type(m4).__name__)
    b0.eval(); m4.eval()
    print('PASS  build_d_accepts_both_arms     B0 + M4 built on %s' % a.device)

    # --- 4. the bias must be zero-initialised ------------------------------
    assert 'sensor_score.cond_bias.weight' in missing, (
        'cond_bias was NOT in the missing list, so it was loaded from the init '
        'instead of being zero-initialised: %s' % sorted(missing))
    w = m4.sensor_score.cond_bias.weight
    assert float(w.abs().max()) == 0.0, 'cond_bias is not zero at build time'
    print('PASS  bias_is_zero_initialised      max|w| = %.3e' % float(w.abs().max()))

    b = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in make_batch(a.subset).items()}

    # --- 2. transparent at step 0 -----------------------------------------
    with torch.no_grad():
        p0 = pred(b0, b)
        p4 = pred(m4, b)
    d0 = float((p0 - p4).abs().max())
    assert d0 <= a.tol, 'step-0 parity broken: max|diff|=%.3e' % d0
    print('PASS  transparent_at_step0          max|diff| = %.3e' % d0)

    # --- 3. bias is the only difference -----------------------------------
    with torch.no_grad():
        m4.sensor_score.cond_bias.weight.normal_(0, 0.5)
        moved = float((pred(m4, b) - p4).abs().max())
        m4.sensor_score.cond_bias.weight.zero_()
        back = float((pred(m4, b) - p0).abs().max())
    assert moved > 0.0, 'condition bias has no effect on the prediction'
    assert back <= a.tol, 'zeroing the bias did not restore B0: %.3e' % back
    print('PASS  bias_is_the_only_difference   moved = %.3e, restored = %.3e'
          % (moved, back))

    # --- the bias is a live softmax shift over sensors ---------------------
    with torch.no_grad():
        m4.sensor_score.cond_bias.weight.normal_(0, 0.5)
        tokens = torch.randn(2, 10, 14, 64, device=a.device)
        m4.sensor_score.set_context(cond=b['cond'], trend=None)
        wgt = torch.softmax(m4.sensor_score(tokens).squeeze(-1), dim=-1)
        spread = float((wgt.max(dim=-1).values - wgt.min(dim=-1).values).mean())
    assert spread > 0.0, 'reliability weights are uniform: the bias is dead'
    print('PASS  reliability_weights_are_live  mean(max-min w) = %.3e' % spread)

    # --- 5. anti-collapse sentinel ----------------------------------------
    # The 6 step-0 checks cannot see post-training saturation: M4 passed them
    # 5/5 and then froze at val_eng = 69.5273 for three sampled epochs. The
    # failure mode is "pooling collapses to one sensor", so test that mode
    # directly instead of hoping one epoch reveals it: drive cond_bias to an
    # adversarial magnitude and require the prediction to still vary across
    # DIFFERENT inputs. A collapsed model returns the same number for all of
    # them, which is exactly what val_eng = const means.
    with torch.no_grad():
        m4.sensor_score.cond_bias.weight.normal_(0, 1000.0)   # adversarial
        xs = []
        for k in range(4):
            bb = {kk: (v.to(a.device) if torch.is_tensor(v) else v)
                  for kk, v in make_batch(a.subset, seed=100 + k).items()}
            xs.append(pred(m4, bb))
        stacked = torch.cat(xs)
        spread_pred = float(stacked.max() - stacked.min())
        m4.sensor_score.cond_bias.weight.zero_()
    assert spread_pred > 1e-3, (
        'pooling collapsed under an adversarial bias: the prediction is the '
        'same for every input (spread=%.3e). This is the M4 freeze mode.' % spread_pred)
    print('PASS  no_collapse_under_adversarial_bias  pred spread = %.3e' % spread_pred)

    print('\nCONTRACT OK  (6/6, through build_d)')


if __name__ == '__main__':
    main()
