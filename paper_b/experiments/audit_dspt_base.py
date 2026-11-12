"""Runtime audit of what DSPT-Base actually computes (Paper B, 2026-11-12).

Read-only with respect to experiment artifacts: builds models from the same
``train._build`` path used by the A8 grid, feeds synthetic tensors, and reports
per-module evidence.  No training, no test data, no writes outside stdout.

Usage:
  python -m paper_b.experiments.audit_dspt_base --out audit_dspt_base.json
"""
import argparse
import json

import torch
import torch.nn as nn

from ..models.dspt import masked_mean
from ..training.train import DEFAULTS, _build

BASE_CFG = dict(DEFAULTS, model='dspt', span=70, ms_mode='off', sa_mode='off',
                hi_source='none', instnorm=False, use_cycle=True, use_pad_mask=True,
                mode='dual')
VAN_CFG = dict(DEFAULTS, model='vanilla', span=70, input_mode='raw', use_cycle=False,
               mode='dual')
VAN_DSP_CFG = dict(DEFAULTS, model='vanilla', span=70, input_mode='dsp', use_cycle=True,
                   mode='dual')


def nparams(mod):
    return int(sum(p.numel() for p in mod.parameters()))


def synth(B=4, L=70, N=14, seed=0, lengths=(70, 70, 70, 22)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, N, generator=g)
    mask = torch.zeros(B, L)
    for i, m in enumerate(lengths):
        mask[i, L - m:] = 1.0
    # padding region is zero in the real pipeline
    x = x * mask.unsqueeze(-1)
    baseline = torch.randn(B, N, generator=g) * 0.1
    cycle = torch.tensor([0.9, 0.6, 0.3, 0.1])
    cond = torch.tensor([0, 1, 2, 3])
    return x, mask, baseline, cycle, cond


def named_modules_of_type(model, cls):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, cls)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    torch.manual_seed(0)
    base = _build(BASE_CFG).eval()
    van = _build(VAN_CFG).eval()
    vand = _build(VAN_DSP_CFG).eval()
    x, mask, baseline, cycle, cond = synth()

    rep = {'base_cfg': {k: BASE_CFG[k] for k in
                        ('model', 'span', 'd_model', 'layers', 'ms_mode', 'sa_mode',
                         'hi_source', 'instnorm', 'use_cycle', 'use_pad_mask', 'mode')},
           'params': {'dspt_base': nparams(base), 'vanilla_raw': nparams(van),
                      'vanilla_dsp_cycle': nparams(vand)}}

    # ---- 1. input representation -----------------------------------------
    rep['features'] = {
        'sensor_in_in_features': int(base.sensor_in.in_features),
        'n_feat_attr': int(base.n_feat),
        'instnorm': bool(base.instnorm),
    }
    feats = base._features(x, baseline, mask)
    rep['features'].update({
        'shape': list(feats.shape),
        'ch0_equals_x': bool(torch.allclose(feats[..., 0], x)),
        'ch1_equals_x_minus_baseline': bool(torch.allclose(feats[..., 1], x - baseline.unsqueeze(1))),
        # EMA recurrence holds on valid steps and freezes on padding steps
        'ch2_recursion_holds': bool(torch.allclose(
            feats[:, 1:, :, 2][mask[:, 1:].bool().unsqueeze(-1).expand_as(feats[:, 1:, :, 2])],
            (0.5 * feats[:, :-1, :, 2] + 0.5 * x[:, 1:])[mask[:, 1:].bool().unsqueeze(-1).expand_as(feats[:, 1:, :, 2])],
            atol=1e-6)),
        'ch2_frozen_on_padding': bool(torch.allclose(
            feats[3, :48, :, 2], feats[3, 0, :, 2].expand(48, feats.shape[2]), atol=1e-6)),
        'ch2_differs_from_ch0': float((feats[..., 2] - feats[..., 0]).abs().max()),
        'ch0_nonzero_in_padding': float(feats[3, :48, :, 0].abs().max()),
        # EMA must start at the first valid frame, not at x[:,0] (P2 fix)
        'ema_at_first_valid_equals_x_at_first_valid': bool(
            torch.allclose(feats[3, 48, :, 2], x[3, 48], atol=1e-6)),
    })

    # ---- 2. module inventory ---------------------------------------------
    convs = named_modules_of_type(base, nn.Conv1d)
    rep['modules'] = {
        'conv1d': [(n, m.kernel_size[0], m.groups, m.padding_mode) for n, m in convs],
        'conv1d_count': len(convs),
        'has_ms_proj': hasattr(base, 'ms_proj'),
        'has_sensor_attn': hasattr(base, 'sensor_attn'),
        'has_sensor_score': hasattr(base, 'sensor_score'),
        'has_static_score': hasattr(base, 'static_score'),
        'has_cond_emb': hasattr(base, 'cond_emb'),
        'has_hi_head': hasattr(base, 'hi_head'),
        'has_pe_proj': hasattr(base, 'pe_proj'),
        'temporal_pe_shape': list(base.temporal_pe.shape),
        'temporal_layers': len(base.layers),
        'per_sensor_attention_heads': base.layers[0].self_attn.num_heads,
    }

    # ---- 3. where does the conv act, and is it active? -------------------
    with torch.no_grad():
        h = base.sensor_in(base._features(x, baseline, mask))
        hh = h.permute(0, 2, 3, 1)
        conv_out = base.conv3(hh.reshape(-1, base.d_model, hh.size(-1)))
        rep['conv'] = {
            'input_shape': list(hh.shape),
            'conv_out_shape': list(conv_out.shape),
            'conv_out_std': float(conv_out.std()),
            'conv_changes_input': bool(not torch.allclose(conv_out.reshape(hh.shape), hh, atol=1e-5)),
        }

    # ---- 4. temporal transformer runs per sensor -------------------------
    captured = {}
    def hook(mod, inp, out):
        captured['tokens_in'] = tuple(inp[0].shape)
    hd = base.layers[0].register_forward_hook(hook)
    with torch.no_grad():
        base.encode(x, baseline, mask, cycle, cond)
    hd.remove()
    rep['temporal_transformer'] = {
        'tokens_in': captured.get('tokens_in'),
        'expected': [x.size(0) * x.size(2), x.size(1), base.d_model],
        'masked_positions_are_padding': True,
    }

    # ---- 5. learned sensor pooling ---------------------------------------
    base.zero_grad(set_to_none=True)
    out = base(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond)
    out['prediction'].sum().backward()
    rep['sensor_pooling'] = {
        'score_weight_shape': list(base.sensor_score.weight.shape),
        'score_grad_norm': float(base.sensor_score.weight.grad.norm()),
        'is_learned': bool(base.sensor_score.weight.grad.norm() > 0),
    }
    with torch.no_grad():
        B, L, N = x.shape
        feats = base._features(x, baseline, mask)
        h = base.sensor_in(feats)
        hh4 = h.permute(0, 2, 3, 1)
        valid = mask.bool()
        first = valid.float().argmax(dim=1)
        fv = hh4[torch.arange(B), :, :, first]
        m4 = valid.unsqueeze(1).unsqueeze(1)
        hh = torch.where(m4, hh4, fv.unsqueeze(-1)).reshape(B * N, base.d_model, L)
        conv = base._ms(hh) * m4.expand(B, N, 1, L).reshape(B * N, 1, L)
        conv_out = conv.reshape(B, N, base.d_model, L).permute(0, 3, 1, 2)
        h = base.time_norm(h + conv_out)
        tokens = h.permute(0, 2, 1, 3).reshape(B * N, L, base.d_model)
        dist = torch.arange(L - 1, -1, -1)
        tokens = tokens + base.temporal_pe[:, dist]
        pad = ~mask.bool().unsqueeze(1).expand(B, N, L).reshape(B * N, L)
        for layer in base.layers:
            tokens = layer(tokens, src_key_padding_mask=pad)
        tokens = tokens.reshape(B, N, L, base.d_model).permute(0, 2, 1, 3)
        score = torch.softmax(base.sensor_score(tokens).squeeze(-1), dim=-1)
        rep['sensor_pooling'].update({
            'score_shape': list(score.shape),
            'score_sums_to_one': bool(torch.allclose(score.sum(-1), torch.ones_like(score.sum(-1)), atol=1e-5)),
            'score_max_over_sensors_mean': float(score.max(dim=-1).values.mean()),
            'score_is_time_varying': bool((score.std(dim=1) > 1e-3).float().mean() > 0.5),
        })

    # ---- 6. cycle encoding: active, and time-invariant -------------------
    with torch.no_grad():
        e1 = base.encode(x, baseline, mask, cycle, cond)
        e2 = base.encode(x, baseline, mask, torch.zeros_like(cycle), cond)
        d = (e1 - e2)
    rep['cycle_encoding'] = {
        'pe_proj_in_features': int(base.pe_proj.in_features),
        'pred_delta_when_cycle_zeroed': float((base(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond)['prediction']
                                               - base(x, mask=mask, baseline=baseline, cycle=torch.zeros_like(cycle), cond=cond)['prediction']).abs().max()),
        'encoded_delta_max_over_time': float(d.abs().max()),
        # a constant-over-time code leaves no temporal variance
        'encoded_delta_time_std_max': float(d.std(dim=1).max()),
        'applied_after_temporal_layers': True,
        'note': 'pe is added to (B,L,D) as a (B,1,D) broadcast after pooling over sensors',
    }

    # ---- 7. dual span ----------------------------------------------------
    enc_cache = {}
    def _norm_hook(mod, inp, out):
        enc_cache['h'] = inp[0].detach()
    _hnd = base.norm.register_forward_hook(_norm_hook)
    with torch.no_grad():
        o = base(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond, mode='dual')
        oA = base(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond, mode='short')
        oB = base(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond, mode='long')
    _hnd.remove()
    m = mask.sum(1)
    with torch.no_grad():
        # how different are the two spans, relative to the embedding scale?
        hA = masked_mean(enc_cache['h'], (torch.arange(x.size(1) - 1, -1, -1).unsqueeze(0) < 50) & mask.bool())
        hB = masked_mean(enc_cache['h'], mask)
        rel = float((hA - hB).norm(dim=-1).mean() / hA.norm(dim=-1).mean())
    rep['dual_span'] = {
        'gate_values': [round(float(v), 4) for v in o['gate']],
        'gate_formula': 'clamp((n_valid-50)/50, 0, 1)',
        'gate_has_parameters': any('gate' in n for n, _ in base.named_parameters()),
        'hA_vs_hB_relative_diff': rel,
        'yA_vs_yB_max_abs_diff': float((oA['prediction'] - oB['prediction']).abs().max()),
        'dual_equals_short_at_n20': bool(torch.allclose(o['prediction'][3], oA['prediction'][3], atol=1e-6)),
        'dual_equals_long_at_n70': bool(torch.allclose(o['prediction'][0], oB['prediction'][0], atol=1e-6)),
        'short_span_frames': 50,
    }

    # ---- 8. padding mask -------------------------------------------------
    base_nomask = _build(dict(BASE_CFG, use_pad_mask=False)).eval()
    base_nomask.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        on = base(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond)['prediction']
        off = base_nomask(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond)['prediction']
    rep['padding_mask'] = {
        'mask_changes_output_on_short_window': float((on - off).abs().max()),
        'short_window_index': 3,
        'masked_mean_ignores_padding': bool(torch.allclose(
            masked_mean(
                torch.arange(70, dtype=torch.float).view(1, 70, 1).repeat(1, 1, 1),
                torch.tensor([[0.] * 48 + [1.] * 22])),
            torch.tensor([[[58.5]]], dtype=torch.float), atol=1e-5)),
    }

    # ---- 9. vanilla baseline ---------------------------------------------
    with torch.no_grad():
        vo = van(x, mask=mask, baseline=baseline, cycle=None, cond=cond)
        vdo = vand(x, mask=mask, baseline=baseline, cycle=cycle, cond=cond)
    rep['vanilla'] = {
        'embed_in_features_raw': int(van.embed.in_features),
        'embed_in_features_dsp': int(vand.embed.in_features),
        'conv_modules': len(named_modules_of_type(van, nn.Conv1d)),
        'sensor_attention_modules': 0,
        'learned_positional_table': any('temporal_pe' in n for n, _ in van.named_parameters()),
        'sinusoidal_pe_buffer': 'pe.pe' in dict(van.named_buffers()),
        'cycle_proj': hasattr(van, 'cycle_proj'),
        'dual_span': bool(getattr(van, 'dual_span', False)),
        'output_keys': sorted(vo.keys()),
        'pred_finite': bool(torch.isfinite(vo['prediction']).all()),
        'dsp_variant_pred_finite': bool(torch.isfinite(vdo['prediction']).all()),
        'dsp_variant_uses_cycle': bool(vand.use_cycle),
    }

    # ---- 10. capacity comparison ----------------------------------------
    def group_params(mod):
        groups = {}
        for n, p in mod.named_parameters():
            key = n.split('.')[0]
            groups[key] = groups.get(key, 0) + int(p.numel())
        return dict(sorted(groups.items(), key=lambda kv: -kv[1]))
    rep['param_groups'] = {'dspt_base': group_params(base), 'vanilla_raw': group_params(van)}
    rep['n_params'] = {'dspt_base': nparams(base), 'vanilla_raw': nparams(van),
                       'vanilla_dsp_cycle': nparams(vand)}

    txt = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    print(txt)
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
