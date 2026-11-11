"""Real-data, validation-only architecture/gradient and split checks.

These are fidelity checks, not trained baseline results.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from .adapters import UpstreamAdapter, module
from .common import seed_everything, write_json
from .data import prepare
from .unified import configuration


def check(model, subset, data_dir):
    seed_everything(101)
    cfg = configuration(model, subset, 101)
    data = prepare(subset, Path(data_dir), cfg)
    tr = data["train"]
    assert set(tr["unit"]) == set(data["train_units"])
    assert set(data["val"]["unit"]) == set(data["val_units"])
    assert not set(tr["unit"]) & set(data["val"]["unit"])
    assert np.max(tr["y"]) <= 125 and np.min(tr["y"]) >= 0
    net = UpstreamAdapter(cfg).cuda()
    x, age = torch.from_numpy(tr["x"][:4]).cuda(), torch.from_numpy(tr["age"][:4]).cuda()
    y = torch.from_numpy(tr["y"][:4]).cuda()
    net.eval()
    with torch.no_grad():
        torch.manual_seed(173)
        torch.cuda.manual_seed_all(173)
        adapted = net(x, age)
        torch.manual_seed(173)
        torch.cuda.manual_seed_all(173)
        if model == "aidgn":
            direct = net.net(x, age)[1].reshape(-1) * 125
        elif model == "ffve":
            mu, lv = net.encoder(x)
            direct = net.decoder(net.variational.reparameterize(mu, lv)).reshape(-1)
        elif model == "fcstgnn":
            direct = net.net(x.reshape(4, net.time_length, net.patch, 14).transpose(2, 3).contiguous()).reshape(-1) * 125
        else:
            direct = net.net(x).reshape(-1) * net.output_scale
        assert torch.equal(adapted, direct), model
    net.train()
    if model == "dual_mixer":
        cmapss = module(model, "dataset/cmapss.py", "check_cmapss")
        ds = cmapss.Cmapss(tr["x"], tr["unit"], tr["y"] / 125)
        cmapss.CmapssGaussianNegativeSampler(ds, 5, thresh=0.5, std=0.3)
        sampled = [ds[i] for i in range(4)]
        x = torch.from_numpy(np.stack([item[0] for item in sampled])).cuda()
        y = torch.from_numpy(np.stack([item[1] for item in sampled])).cuda()
    loss = net.loss(x, y, age)
    loss.backward()
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and gradients and all(torch.isfinite(g).all() for g in gradients)
    if model == "two_phase":
        pre = net.pre_module.Transformer_pre(17, 16, 45, 32, 32, 1 if subset in ["FD001", "FD003"] else 2,
                                             2, 0.1 if subset in ["FD001", "FD003"] else 0).cuda()
        out, domain = pre(x)
        preloss = (out.reshape(-1) - y).square().mean().sqrt() + torch.nn.functional.cross_entropy(domain, torch.tensor([0, 1, 0, 1]).cuda())
        preloss.backward()
        assert torch.isfinite(preloss)
        transfer = net.net.encoder.load_state_dict(pre.encoder.state_dict(), strict=True)
        assert not transfer.missing_keys and not transfer.unexpected_keys
    return {"model": model, "subset": subset, "window": cfg["window"], "params": sum(p.numel() for p in net.parameters()),
            "loss": float(loss.detach()), "direct_forward_equal": True, "finite_backward": True,
            "n_training_engines": len(data["train_units"]), "n_validation_engines": len(data["val_units"]),
            "n_training_windows": len(tr["y"]), "n_validation_cuts": len(data["val"]["y"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn"])
    a = ap.parse_args()
    results = []
    for model in a.models:
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            results.append(check(model, subset, a.data_dir))
            print("PASS", results[-1], flush=True)
    write_json(a.out, results)
