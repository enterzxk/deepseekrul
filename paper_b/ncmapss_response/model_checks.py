"""CPU-only contracts. Run as python -m paper_b.ncmapss_response.model_checks.

No NASA files, optimizer updates, GPU allocation, or official TEST inference.
These checks establish software identities, not a useful degradation mechanism.
"""
from __future__ import annotations

import torch
from .models import (ResponseEncoder, MODES, LATENT, common_contrast,
                     psd_parallel_sum, identified_inverse, direction_weighted_contrast)


def check_unused_common_exact(raw, prior, support):
    """Compare the optimized path against the old full diagnostic computation.

    CPU FP32/FP64, inference and auxiliary-loss paths; all functional outputs,
    input/parameter gradients and RNG states must be bitwise equal. No step.
    """
    for dtype in (torch.float32, torch.float64):
        for mode in ('raw', 'fe', 'delta'):
            for block in (None, 1):
                fast = ResponseEncoder(mode).to(dtype=dtype)
                full = ResponseEncoder(mode).to(dtype=dtype)
                full.load_state_dict(fast.state_dict(), strict=True)
                x_fast = raw.to(dtype=dtype).detach().clone().requires_grad_()
                x_full = raw.to(dtype=dtype).detach().clone().requires_grad_()
                rng = torch.get_rng_state().clone()
                a = fast(x_fast, prior, support, support, block=block)
                after_fast = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                b = full(x_full, prior, support, support, block=block, audit_full_common=True)
                assert torch.equal(after_fast, torch.get_rng_state()) and torch.equal(rng, after_fast)
                assert a['diagnostics']['common_diagnostics_status'] == 'not_computed'
                assert b['diagnostics']['common_diagnostics_status'] == 'computed'
                assert 'parallel_sum' not in a['diagnostics']
                for key in ('embedding', 'embedding_without_reference',
                            'reconstruction_loss', 'reconstruction_per_flight'):
                    assert torch.equal(a[key], b[key]), (mode, dtype, block, key)
                for key in ('coverage', 'level_code', 'innovation', 'aux_context_latent'):
                    assert torch.equal(a['diagnostics'][key], b['diagnostics'][key])
                loss_a = (a['embedding'].square().mean() +
                          .3 * a['embedding_without_reference'].square().mean() +
                          .1 * a['reconstruction_loss'])
                loss_b = (b['embedding'].square().mean() +
                          .3 * b['embedding_without_reference'].square().mean() +
                          .1 * b['reconstruction_loss'])
                loss_a.backward()
                loss_b.backward()
                assert torch.equal(x_fast.grad, x_full.grad), (mode, dtype, block, 'input gradient')
                for (name, p), (other_name, q) in zip(fast.named_parameters(), full.named_parameters()):
                    assert name == other_name and torch.equal(p, q)
                    assert (p.grad is None) == (q.grad is None), (mode, dtype, block, name)
                    if p.grad is not None:
                        assert torch.equal(p.grad, q.grad), (mode, dtype, block, name)
                assert torch.equal(after_fast, torch.get_rng_state())


def run_checks():
    torch.set_default_device('cpu')
    torch.manual_seed(20261113)
    torch.set_num_threads(2)
    checks = []
    f, t = 3, 512
    raw = torch.randn(f, t, 18, dtype=torch.float64)
    prior = torch.tensor([0, 0, 1])
    support = torch.ones(f, t, dtype=torch.bool)

    states = []
    coverages = []
    for mode in MODES:
        torch.manual_seed(7)
        net = ResponseEncoder(mode).double()
        states.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        out = net(raw, prior, support, support, block=1)
        assert out['embedding'].shape == (f, 64)
        assert out['embedding_without_reference'].shape == (f, 64)
        assert out['reconstruction_per_flight'].shape == (f,)
        assert torch.equal(out['reconstruction_loss'], out['reconstruction_per_flight'].mean())
        assert all(torch.isfinite(x).all() for x in [out['embedding'], out['embedding_without_reference'], out['reconstruction_per_flight']])
        # Same projection, exactly the innovation/coverage columns removed.
        info = out['diagnostics']
        removed = (info['innovation'] @ net.embedding.weight[:, 68:72].T +
                   info['coverage'][:, None] * net.embedding.weight[:, 72])
        assert torch.allclose(out['embedding'] - out['embedding_without_reference'], removed, atol=1e-12, rtol=1e-12)
        assert not torch.equal(out['embedding'], out['embedding_without_reference'])
        coverages.append(out['diagnostics']['coverage'])
        (out['embedding'].square().mean() + .1 * out['reconstruction_loss']).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        for module in [net.background, net.basis, net.response_body]:
            gradients = [p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        # Holdout sensors affect the target/loss, never the context code.
        changed = raw.clone()
        changed[:, 128:256, 4:] += 10
        again = net(changed, prior, support, support, block=1)
        assert torch.equal(out['diagnostics']['aux_context_latent'], again['diagnostics']['aux_context_latent'])
        checks.append(mode + ':finite_gradient_b_A_and_holdout_context')
        empty = torch.zeros_like(support)
        no_common = net(raw, prior, empty, empty)
        if mode in ['raw', 'fe', 'supported', 'common']:
            assert torch.count_nonzero(no_common['diagnostics']['innovation']) == 0
        if mode in ['supported', 'common']:
            assert torch.count_nonzero(no_common['diagnostics']['common_projector']) == 0
            assert torch.isfinite(no_common['embedding']).all()
        if mode in ['raw', 'fe', 'supported', 'common']:
            assert torch.equal(no_common['embedding'], no_common['embedding_without_reference'])
        try:
            net(raw, torch.tensor([1, 0, 1]), support, support)
        except ValueError:
            pass
        else:
            raise AssertionError('future prior accepted')
        report = net.parameter_report()
        assert 0 < report['structurally_effective_parameters'] <= report['total_parameters']
    for state in states[1:]:
        assert all(torch.equal(states[0][k], state[k]) for k in state)
    assert all(torch.equal(coverages[0], c) for c in coverages[1:])
    check_unused_common_exact(raw, prior, support)
    checks.append('unused_spectral_skip_bitwise_equal_outputs_input_parameter_gradients_and_RNG_FP32_FP64')
    checks.append('identical_initial_state_and_shared_coverage')
    checks.append('history_boundary_view_removes_exactly_innovation_and_coverage')
    supported_net = ResponseEncoder('supported').double()
    supported_net.load_state_dict(net.state_dict(), strict=True)
    supported_out = supported_net(raw, prior, support, support, block=1)
    assert torch.equal(supported_out['embedding_without_reference'], out['embedding_without_reference'])
    assert torch.equal(supported_out['reconstruction_per_flight'], out['reconstruction_per_flight'])
    assert torch.equal(supported_out['diagnostics']['coverage'], out['diagnostics']['coverage'])
    expected_difference = ((out['diagnostics']['weighted_innovation'] -
                            out['diagnostics']['supported_innovation']) @ net.embedding.weight[:, 68:72].T)
    assert torch.allclose(out['embedding'] - supported_out['embedding'], expected_difference, atol=1e-12, rtol=1e-12)
    assert out['diagnostics']['weighting_change_norm'].max() > 0
    checks.append('supported_vs_common_only_direction_weight_changes')

    qc = torch.diag(torch.tensor([1., 0., 0., 0.], dtype=torch.float64))[None]
    qr = torch.diag(torch.tensor([0., 1., 0., 0.], dtype=torch.float64))[None]
    h, projection, info = psd_parallel_sum(qc, qr)
    assert torch.count_nonzero(h) == 0 and torch.count_nonzero(projection) == 0
    assert info['common_rank'].item() == 0
    checks.append('orthogonal_excitation_parallel_sum_exact_zero')

    probe = torch.tensor([[1., -2., .5, 3.]], dtype=torch.float64)
    rank = torch.tensor([4])
    iso = 2 * torch.eye(4, dtype=torch.float64)[None]
    assert torch.equal(direction_weighted_contrast(probe, iso, rank), probe)
    # Identity is on the retained range, including when r differs from k=4.
    partial = torch.diag(torch.tensor([2., 2., 0., 0.], dtype=torch.float64))[None]
    retained_probe = torch.tensor([[1., -2., 0., 0.]], dtype=torch.float64)
    assert torch.equal(direction_weighted_contrast(retained_probe, partial, torch.tensor([2])), retained_probe)
    assert torch.count_nonzero(direction_weighted_contrast(probe, iso, torch.tensor([0]))) == 0
    strengths = torch.tensor([[.5, 1., 2., 4.]], dtype=torch.float64, requires_grad=True)
    vector = probe.clone().requires_grad_()
    anisotropic = direction_weighted_contrast(vector, torch.diag_embed(strengths), rank)
    assert not torch.allclose(anisotropic, vector)
    anisotropic.square().sum().backward()
    assert torch.isfinite(vector.grad).all() and torch.isfinite(strengths.grad).all()
    assert strengths.grad.abs().sum() > 0
    checks.append('common_direction_weight_isotropic_identity_anisotropic_effect_finite_gradient')

    # Repeated eigenvalues and zero spectra must not cause NaN gradients.
    matrices = torch.stack([torch.zeros(4, 4), torch.eye(4), torch.diag(torch.tensor([1., 1., 0., 0.]))]).double().requires_grad_()
    inv, _, cleaned, _ = identified_inverse(matrices)
    (inv.square().sum() + cleaned.square().sum()).backward()
    assert torch.isfinite(matrices.grad).all()
    checks.append('repeated_and_zero_spectra_finite_gradient')

    # Exact known responses under different design Grams; no ridge in contrast.
    torch.manual_seed(19)
    a = torch.randn(2, t, 14, LATENT, dtype=torch.float64)
    a[1] = a[1] * torch.tensor([.2, .5, 2., 4.], dtype=torch.float64)
    htrue = torch.tensor([.7, -.4, .2, .8], dtype=torch.float64)
    residual = torch.einsum('ftsk,k->fts', a, htrue)
    same, info = common_contrast(a, residual, torch.tensor([0, 0]), support[:2], support[:2])
    assert info['common_rank'][1] == 4
    assert same[1].abs().max() < 1e-10
    same_weighted = direction_weighted_contrast(same, info['parallel_sum'], info['common_rank'], info['common_supported'])
    assert same_weighted[1].abs().max() < 1e-10
    change = torch.tensor([.2, .1, -.3, .4], dtype=torch.float64)
    residual[1] = torch.einsum('tsk,k->ts', a[1], htrue + change)
    changed, info = common_contrast(a, residual, torch.tensor([0, 0]), support[:2], support[:2])
    assert torch.allclose(changed[1], change, atol=1e-10, rtol=1e-10)
    changed_weighted = direction_weighted_contrast(changed, info['parallel_sum'], info['common_rank'], info['common_supported'])
    assert changed_weighted[1].norm() > 0
    checks.append('same_response_different_Gram_zero_and_visible_change_retained')

    # Fewer than 32 matched points is not usable common excitation.
    short = torch.zeros(2, t, dtype=torch.bool)
    short[:, :31] = True
    unsupported, info = common_contrast(a, residual, torch.tensor([0, 0]), short, support[:2])
    assert torch.count_nonzero(unsupported) == 0
    assert torch.count_nonzero(direction_weighted_contrast(unsupported, info['parallel_sum'], info['common_rank'], info['common_supported'])) == 0
    checks.append('minimum_support_gate_exact_zero')
    print('CPU_MODEL_CONTRACTS_PASS', len(checks), checks)
    return checks


if __name__ == '__main__':
    run_checks()
