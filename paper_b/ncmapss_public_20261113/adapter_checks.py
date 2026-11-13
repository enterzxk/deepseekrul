"""Synthetic software contracts only: no data files, training or TEST inference.

Run explicitly on the intended host, e.g. python -B -m
paper_b.ncmapss_public_20261113.adapter_checks --author-file ... --device cuda:0
All five shape forwards use B=1, including 70*512=35840. The controller owns
any B=64 feasibility probe. This module never launches an optimizer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path


def _implementation():
    # Lazy so --help is usable without installing/importing torch.
    import torch
    from . import aidgn_adapter as api
    return torch, api


def _reject(function):
    try:
        function()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError("invalid input or provenance was accepted")


def check_manifest_rejection(author_file, manifest_path=None):
    _, api = _implementation()
    author_file = Path(author_file).resolve()
    manifest_path = Path(manifest_path) if manifest_path is not None else author_file.parent.parent / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = author_file.read_bytes()
    api._verify_bytes(source, manifest)
    bad = copy.deepcopy(manifest)
    next(x for x in bad["included"] if x["path"] == "Model/AIDGN.py")["sha256"] = "0" * 64
    _reject(lambda: api._verify_bytes(source, bad))
    _reject(lambda: api._verify_bytes(source + b" ", manifest))
    bad = copy.deepcopy(manifest); bad["full_commit_sha"] = "0" * 40
    _reject(lambda: api._verify_bytes(source, bad))
    return {"manifest_mismatch_rejected": True, "source_mismatch_rejected": True,
            "wrong_commit_rejected": True, "source_sha256": hashlib.sha256(source).hexdigest()}


def check_pack_contract(device="cpu"):
    torch, api = _implementation()
    device = torch.device(device)
    f = 8
    raw = torch.arange(f*512*18, device=device, dtype=torch.float32).reshape(f, 512, 18)
    age = torch.arange(1, f+1, device=device, dtype=torch.float32) / f
    windows = torch.full((f, 70), -1, device=device, dtype=torch.long)
    for query in range(f):
        windows[query, -query-1:] = torch.arange(query+1, device=device)

    class Guard(dict):
        def __getitem__(self, key):
            if key not in {"raw", "age", "windows"}:
                raise AssertionError("forbidden bundle read: " + key)
            return super().__getitem__(key)

    guarded = Guard(raw=raw, age=age, windows=windows,
                    target=object(), y=object(), unit=object(), cycle=object())
    shapes = []
    for history, points in api.WINDOW_CATALOG:
        query = torch.tensor([5], device=device)
        x, a = api.pack_window(guarded, query, history, points)
        assert x.shape == (1, history*points, 18) and a.shape == (1, history*points, 1)
        assert torch.equal(x[0, -1], raw[5, -1])
        assert a[0, -1, 0] == age[5]
        chosen = windows[5, -history:]
        offset = int((chosen < 0).sum()) * points
        assert not bool(x[0, :offset].any()) and not bool(a[0, :offset].any())
        ids = torch.linspace(0, 511, points, dtype=torch.float64, device=device).round().long()
        valid = chosen[chosen >= 0]
        expected = raw[valid[:, None], ids[None, :], :].reshape(-1, 18)
        assert torch.equal(x[0, offset:], expected)
        # Alter all later observations; no selected/padded value may change.
        changed_raw, changed_age = raw.clone(), age.clone()
        changed_raw[6:] = float("nan"); changed_age[6:] = float("nan")
        altered = Guard(raw=changed_raw, age=changed_age, windows=windows)
        y, b = api.pack_window(altered, query, history, points)
        assert torch.equal(x, y) and torch.equal(a, b)
        shapes.append({"history": history, "points": points, "batch": 1, "T": history*points})
    x, a = api.pack_window(guarded, [0], 70, 4)
    assert not bool(x[0, :-4].any()) and not bool(a[0, :-4].any())
    future = windows.clone(); future[5, -2] = 6
    _reject(lambda: api.pack_window(Guard(raw=raw, age=age, windows=future), [5], 70, 4))
    _reject(lambda: api.pack_window(guarded, [5.1], 1, 50))
    _reject(lambda: api.pack_window(guarded, [5], 2, 50))
    return {"registered_shapes": shapes, "only_raw_age_windows_read": True,
            "past_only_and_zero_padding": True, "future_values_irrelevant": True,
            "unit_identity_scope": "validated by bundle caller; pack validates index-causality only"}


def _toy_input(torch, length, device, batch=1):
    x = torch.arange(batch*length*18, dtype=torch.float32, device=device).reshape(batch, length, 18)
    x = torch.sin(x / 97.)
    age = torch.linspace(.01, 1., length, dtype=torch.float32, device=device)[None, :, None].expand(batch, -1, -1).clone()
    return x, age


def check_constructor_api_and_grad(author_file, manifest_path=None, device="cpu"):
    torch, api = _implementation()
    device = torch.device(device)
    adapter = api.AIDGNAdapter(author_file, 1, 50, manifest_path).to(device).eval()
    original = adapter.author_class(api.author_arguments(1, 50)).to(device).eval()
    assert type(adapter.author_model) is type(original)
    assert Path(inspect.getsourcefile(type(original))).resolve() == Path(author_file).resolve()
    assert inspect.signature(type(original).forward).parameters.keys() == {"self", "x", "OCC"}
    original.load_state_dict(adapter.author_model.state_dict(), strict=True)
    assert set(adapter.state_dict()) == {"author_model." + k for k in original.state_dict()}
    assert sum(p.numel() for p in adapter.parameters()) == sum(p.numel() for p in original.parameters())
    args = adapter.author_args
    assert (args.hidden_dim, args.fc_layer_dim, args.feature_fc_layer_dim, args.AATE_dim,
            args.nlayer, args.hop, args.fc_dropout, args.outlayer) == (128, 512, 72, 16, 1, 1, .1, "Linear")
    x, age = _toy_input(torch, 50, device)
    with torch.no_grad():
        original.device = device
        expected = original(x, age)[1].reshape(1)
        actual = adapter(x, age)
    assert torch.equal(actual, expected)
    assert actual.shape == (1,) and actual.dtype == torch.float32 and bool(torch.isfinite(actual).all())
    _reject(lambda: adapter(x[..., :17], age))
    _reject(lambda: adapter(x, age[:, :-1]))
    _reject(lambda: adapter(x.double(), age.double()))
    x, age = _toy_input(torch, 50, device, batch=2)
    x.requires_grad_(); age.requires_grad_()
    parameters = tuple(adapter.parameters())
    before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    output = adapter(x, age)
    gradients = torch.autograd.grad(output.square().mean(), parameters + (x, age), allow_unused=True)
    present = [g for g in gradients if g is not None]
    assert present and all(g.dtype == torch.float32 and bool(torch.isfinite(g).all()) for g in present)
    assert any(g is not None and bool((g != 0).any()) for g in gradients[:len(parameters)])
    assert gradients[-1] is not None and gradients[-2] is not None
    assert all(torch.equal(before[k], v) for k, v in adapter.state_dict().items())
    return {"exact_author_class": True, "exact_eval_output_and_state_api": True,
            "parameters": sum(p.numel() for p in adapter.parameters()),
            "finite_fp32_gradients": True, "nonzero_parameter_gradient": True,
            "observed_nonzero_age_gradient": bool((gradients[-1] != 0).any()),
            "parameters_without_gradient": sum(g is None for g in gradients[:len(parameters)]),
            "optimization_steps": 0, "provenance": adapter.provenance}


def check_five_model_shapes(author_file, manifest_path=None, device="cpu"):
    torch, api = _implementation()
    device = torch.device(device)
    rows = []
    for history, points in api.WINDOW_CATALOG:
        # Isolated software probe: B=1 only, never B=64 or an optimizer step.
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        adapter = api.AIDGNAdapter(author_file, history, points, manifest_path).to(device).eval()
        x, age = _toy_input(torch, history*points, device, batch=1)
        with torch.no_grad():
            prediction = adapter(x, age)
        assert prediction.shape == (1,) and prediction.dtype == torch.float32
        assert bool(torch.isfinite(prediction).all())
        rows.append({"history": history, "points": points, "T": history*points, "batch": 1,
                     "parameters": sum(p.numel() for p in adapter.parameters()),
                     "finite_fp32_output": True,
                     "peak_allocated_MiB": torch.cuda.max_memory_allocated(device)/1024**2
                     if device.type == "cuda" else None})
        del adapter, x, age, prediction
    return rows


def run_checks(author_file, manifest_path=None, device="cpu"):
    before = hashlib.sha256(Path(author_file).read_bytes()).hexdigest()
    report = {"software_contracts_only": True, "device": device, "real_data_read": False,
              "official_test_read": False, "training_performed": False, "optimization_steps": 0,
              "manifest": check_manifest_rejection(author_file, manifest_path),
              "packing": check_pack_contract(device),
              "author_api_and_grad": check_constructor_api_and_grad(author_file, manifest_path, device),
              "five_B1_shapes": check_five_model_shapes(author_file, manifest_path, device)}
    assert hashlib.sha256(Path(author_file).read_bytes()).hexdigest() == before
    report["author_source_unchanged"] = True
    report["status"] = "PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_checks(args.author_file, args.manifest, args.device), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

