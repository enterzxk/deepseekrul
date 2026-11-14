"""Explicit software contracts; nothing runs on import and no optimizer exists.

The implementation author has NOT run forward/gradient checks. The controller
may execute them in an isolated environment with --forward. No real data is used.
CLI without --forward verifies source and toy packing/metadata contracts only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def check_source(author_file):
    from .bayes_adapter import verify_author_sources
    source, provenance = verify_author_sources(author_file)
    manifest = json.loads((Path(author_file).resolve().parents[3] / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(manifest)
    changed["full_commit_sha"] = "0" * 40
    try:
        verify_author_sources(author_file, manifest_override=changed)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong commit was accepted")
    changed = copy.deepcopy(manifest)
    next(e for e in changed["included"] if e["path"].endswith("models/nets/inception.py"))["sha256"] = "0" * 64
    try:
        verify_author_sources(author_file, manifest_override=changed)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong source digest was accepted")
    return {"status": "PASS", "verified_files": len(source), "provenance": provenance,
            "wrong_commit_rejected": True, "wrong_sha_rejected": True}


def check_packing(device="cpu"):
    import torch
    from .bayes_adapter import BayesHNNAdapter, WINDOW_CATALOG, pack_window
    class Guard(dict):
        def __getitem__(self, key):
            if key not in ("raw", "age", "windows"):
                raise AssertionError("forbidden bundle read: " + key)
            return super().__getitem__(key)
    raw = torch.arange(5 * 512 * 18, device=device, dtype=torch.float32).reshape(5, 512, 18) / 1000
    age = torch.arange(1, 6, device=device, dtype=torch.float32) / 2
    windows = torch.full((5, 70), -1, device=device, dtype=torch.long)
    for q in range(5):
        windows[q, -(q + 1):] = torch.arange(q + 1, device=device)
    bundle = Guard(raw=raw, age=age, windows=windows)
    result = []
    for history, points in WINDOW_CATALOG:
        x, a = pack_window(bundle, [1], history, points)
        assert x.shape == (1, history * points, 18)
        assert a.shape == (1, history * points, 1)
        assert torch.equal(x[0, -1], raw[1, -1]) and a[0, -1, 0] == age[1]
        padding = max(0, history - 2) * points
        assert not bool(x[:, :padding].any()) and not bool(a[:, :padding].any())
        corrupted = Guard(raw=raw.clone(), age=age.clone(), windows=windows.clone())
        corrupted["raw"][2:] = float("nan")
        corrupted["age"][2:] = float("nan")
        xx, aa = pack_window(corrupted, [1], history, points)
        assert torch.equal(x, xx) and torch.equal(a, aa)
        # Exercise metadata packing without constructing/forwarding a network.
        shell = type("ShapeOnly", (), {"physical_sequence_len": history * points})()
        encoded = BayesHNNAdapter.encode_input(shell, x, a)
        assert encoded.shape == (1, history * points + 1, 18)
        assert encoded[0, 0, 0] == age[1] and encoded[0, 0, 1] == 1
        assert not bool(encoded[:, 0, 2:].any())
        assert torch.equal(encoded[:, 1:], x)
        a_changed = a.clone()
        a_changed[:, -1, 0] += 1
        encoded_changed = BayesHNNAdapter.encode_input(shell, x, a_changed)
        assert encoded_changed[0, 0, 0] == encoded[0, 0, 0] + 1
        assert torch.equal(encoded_changed[:, 1:], encoded[:, 1:])
        result.append({"history": history, "points": points, "physical_T": history * points,
                       "author_T": history * points + 1})
    invalid = Guard(raw=raw, age=age, windows=windows.clone())
    invalid["windows"][1, -2] = 2
    try:
        pack_window(invalid, [1], 20, 16)
    except ValueError:
        pass
    else:
        raise AssertionError("future flight accepted")
    return {"status": "PASS", "catalog": result, "forbidden_reads": 0,
            "future_independence": True, "future_index_rejected": True,
            "zero_padding": True, "metadata_age_transmission": True}


def check_network(author_file, device="cpu"):
    """Actual forward/backward, only when called explicitly by the controller."""
    import torch
    from torch.nn import functional as F
    from .bayes_adapter import BayesHNNAdapter, WINDOW_CATALOG, load_author_api
    api = load_author_api(author_file)
    records = []
    gpu_devices = [] if torch.device(device).type == "cpu" else [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=gpu_devices):
        for history, points in WINDOW_CATALOG:
            torch.manual_seed(101)
            model = BayesHNNAdapter(author_file, history, points, device=device)
            torch.manual_seed(101)
            direct = type(model.author_model)(**model.original_args)
            direct.apply(api["weights_init"])
            direct = direct.to(device=device)
            original_state, adapter_state = direct.state_dict(), model.author_model.state_dict()
            assert original_state.keys() == adapter_state.keys()
            assert all(torch.equal(original_state[k], adapter_state[k]) for k in original_state)
            assert sum(p.numel() for p in direct.parameters()) == model.parameter_count
            assert model.original_args["n_features"] == 18 and model.original_args["out_size"] == 2
            try:
                type(model.author_model)(**{**model.original_args, "n_features": 17})
            except AssertionError:
                pass
            else:
                raise AssertionError("original strict 18-feature assertion was lost")
            model.eval()
            direct.eval()
            x = torch.randn(1, history * points, 18, device=device)
            age = torch.full((1, history * points, 1), 0.7, device=device)
            got = model(x, age)
            with torch.no_grad():
                expected = direct(model.encode_input(x, age))
            assert torch.equal(got["mu_norm"], expected[:, 0])
            assert torch.equal(got["sigma_norm"], expected[:, 1])
            assert got["mu_norm"].shape == got["sigma_norm"].shape == (1,)
            assert bool((expected >= 1e-9).all()) and bool(torch.isfinite(expected).all())
            got["sigma_norm"].retain_grad()
            target = torch.tensor([0.5], device=device)
            loss = model.nll(got["mu_norm"], target, got["sigma_norm"])
            reference = F.gaussian_nll_loss(got["mu_norm"], target, got["sigma_norm"].square())
            assert torch.equal(loss, reference)
            loss.backward()
            grads = [p.grad for p in model.parameters()]
            assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            assert any(bool(g.ne(0).any()) for g in grads)
            assert bool(torch.isfinite(got["sigma_norm"].grad).all())
            assert bool(got["sigma_norm"].grad.ne(0).any())
            records.append({"history": history, "points": points,
                            "parameter_count": model.parameter_count,
                            "exact_initialized_state_and_output": True,
                            "original_18_feature_assertion": True,
                            "positive_mu_sigma": True, "native_nll_equal": True,
                            "finite_gradients": True, "nonzero_sigma_gradient": True})
    return {"status": "PASS", "device": str(device), "toy_only": True,
            "optimization_steps": 0, "windows": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-file", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward", action="store_true", help="explicitly execute toy network forward/backward")
    args = parser.parse_args(argv)
    results = {"source": check_source(args.author_file), "packing": check_packing(args.device)}
    results["network"] = check_network(args.author_file, args.device) if args.forward else {"status": "NOT_RUN"}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
