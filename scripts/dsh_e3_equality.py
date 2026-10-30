"""E3 equality harness: prove that use_pe=False is bit-identical to the frozen model.

Modes
  --capture  run BEFORE the patch: save a fixed batch, the seed-2027 init state,
             the forward output and a 3-step loss trace.
  --check    run AFTER the patch: rebuild with use_pe=False, load the saved state,
             and compare keys / forward / loss trace.

CPU only, so the result does not depend on who is using the GPU.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel

CFG = rc.CONFIGS['A']


def build(model_seed=2027, with_flag=False):
    """with_flag=False reproduces the pre-patch constructor (capture mode)."""
    rc.set_seed(model_seed)
    kw = dict(
        num_sensors=14, window_size=50, d_model=64, num_layers=2, dropout=0.15,
        use_baseline=CFG['use_baseline'], use_ema_features=CFG['use_ema_features'],
        use_cycle=CFG['use_cycle'], use_graph=CFG['use_graph'],
        use_dual_head=CFG['use_dual_head'],
    )
    if with_flag:
        kw['use_pe'] = False
    return CandidateModel(**kw)


def fixed_batch():
    prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='cond')
    train_set = prep[0]
    loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=0)
    x, y, cond, extras = next(iter(loader))
    return x, y, cond, extras


def loss_of(model, x, y, cond, extras):
    pred = model(x, cond=cond, extras=extras)['prediction']
    d = pred - y.unsqueeze(1)
    late = torch.where(d > 0, 1.8 * d.square(), d.square())
    return torch.nn.functional.mse_loss(pred, y.unsqueeze(1)) + 0.5 * late.mean()


def three_steps(model, x, y, cond, extras, steps=3):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_of(model, x, y, cond, extras)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['capture', 'check'], required=True)
    ap.add_argument('--dir', default='results/dsh/e3/equality')
    a = ap.parse_args()
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)

    if a.mode == 'capture':
        x, y, cond, extras = fixed_batch()
        model = build().eval()
        with torch.no_grad():
            pred = model(x, cond=cond, extras=extras)['prediction']
        losses = three_steps(build(), x, y, cond, extras)
        np.savez(d / 'batch.npz', x=x.numpy(), y=y.numpy(), cond=cond.numpy(), extras=extras.numpy(),
                 pred=pred.numpy())
        torch.save(build().state_dict(), d / 'init_state.pt')
        np.savez(d / 'loss_trace.npz', losses=np.array(losses))
        out = dict(params=sum(p.numel() for p in build().parameters()), losses=losses,
                   pred_sha=str(abs(float(pred.sum()))) )
        print(json.dumps(out, indent=2))
        return

    b = np.load(d / 'batch.npz')
    x, y = torch.from_numpy(b['x']), torch.from_numpy(b['y'])
    cond, extras = torch.from_numpy(b['cond']), torch.from_numpy(b['extras'])
    ref = torch.from_numpy(b['pred'])
    saved = torch.load(d / 'init_state.pt', map_location='cpu')
    model = build(with_flag=True)
    fresh = build(with_flag=True)
    keys_saved, keys_new = set(saved.keys()), set(fresh.state_dict().keys())
    same_keys = keys_saved == keys_new
    missing = sorted(keys_new - keys_saved); unexpected = sorted(keys_saved - keys_new)
    model.load_state_dict(saved, strict=True)
    model.eval()
    with torch.no_grad():
        pred = model(x, cond=cond, extras=extras)['prediction']
    fwd = float((pred - ref).abs().max())
    ref_losses = np.load(d / 'loss_trace.npz')['losses']
    losses = three_steps(build(with_flag=True), x, y, cond, extras)
    loss_diff = float(np.max(np.abs(np.array(losses) - ref_losses)))
    out = dict(same_keys=same_keys, missing=missing, unexpected=unexpected,
               forward_max_abs_diff=fwd, loss_max_abs_diff=loss_diff,
               losses_now=losses, losses_ref=ref_losses.tolist(),
               params_now=sum(p.numel() for p in fresh.parameters()),
               pass_=bool(same_keys and fwd <= 1e-6 and loss_diff <= 1e-6))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
