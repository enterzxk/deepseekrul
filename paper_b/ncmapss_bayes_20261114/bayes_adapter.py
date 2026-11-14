"""Full BayesRUL Inception-HNN; unified-protocol migration, not native reproduction.

Native recipe: ncmapss_hnn.yaml (lines 12-33), model/{hnn,nn,inception}.yaml.
Adam lr=.001574, weight_decay=.001; batch=250; max_epochs=500; patience=50,
min_delta=0, min nll/val. Native net: relu, dropout=0, bias='True' (the
author's literal string), out_size=2; HNN mc_samples=0 and p_dropout=0.
frequentist.py:28 applies the original miscellaneous.py:53 weights_init.
frequentist.py:38-48 uses GaussianNLL(mu, target, sigma**2) without kwargs.
Both outputs remain the original softplus + Threshold(1e-9, 1e-9).

This adapter adds one explicitly NONPHYSICAL metadata row before the physical
sequence: [current observed cycle / train-q95, 1, 0, ..., 0]. No custom temporal
head, source edit, label input, or physical-feature deletion is used. Physical
scaling belongs to the controller, fitted on training engines only. Targets
must be min(raw_RUL,125)/125; outputs here are NORMALIZED, not cycles. Multiply
both mu_norm and sigma_norm by 125 for cycle units. No z-score target is valid.

Author native DS01-05 joint training, hs-dependent target, random-window val,
and pooled StandardScalerTransform differ and have NOT been reproduced here.
See NATIVE_RECIPE.json for the exact pinned source and protocol distinctions.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

AUTHOR_REPOSITORY = "lbasora/bayesrul"
AUTHOR_COMMIT = "2943f4d85a1bb71910c100c956b32c3f6b595ceb"
MODEL_REL = "bayesrul/models/nets/inception.py"
HNN_REL = "bayesrul/models/frequentist.py"
INIT_REL = "bayesrul/utils/miscellaneous.py"
CORE_SHA256 = {
    MODEL_REL: "83ea189b83264ef53a004d4aea2848f011a8f316bc7b0bc693aca45fcee052f6",
    HNN_REL: "1ccc7bb336a5f121bee4e20e93e72d56772a787c4db4c78868f248d332f60ee3",
    INIT_REL: "f9910d82f6ec4214f192bb5513b892314edafb1806e7423875e79f53197b8f74",
}
WINDOW_CATALOG = ((1, 30), (1, 50), (1, 512), (20, 16), (70, 4))
NATIVE_NET_ARGS = {"activation": "relu", "bias": "True", "dropout": 0, "out_size": 2}
NATIVE_TRAIN_ARGS = {
    "optimizer": "torch.optim.Adam", "lr": 0.001574, "weight_decay": 0.001,
    "batch_size": 250, "min_epochs": 1, "max_epochs": 500,
    "patience": 50, "min_delta": 0.0, "monitor": "nll/val", "mode": "min",
    "seed": 12345, "mc_samples": 0, "p_dropout": 0,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _length(history, points):
    _require(type(history) is int and type(points) is int, "integer window dimensions required")
    _require((history, points) in WINDOW_CATALOG, "window is outside the preregistered catalog")
    return history * points


def verify_author_sources(author_file, manifest_path=None, *, manifest_override=None):
    """Hash all recipe dependencies before executing ANY author source.

    An override is provided solely for in-memory rejection tests; it cannot
    bypass the three independently pinned core hashes above.
    """
    author_file = Path(author_file).resolve()
    root = author_file.parents[3]
    _require(author_file == root / MODEL_REL, "author_file must end in " + MODEL_REL)
    manifest_path = Path(manifest_path) if manifest_path else root / "SOURCE_MANIFEST.json"
    manifest = manifest_override if manifest_override is not None else json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest["repository"] == AUTHOR_REPOSITORY, "repository mismatch")
    _require(manifest["full_commit_sha"] == AUTHOR_COMMIT, "commit mismatch")
    entries = {entry["path"]: entry for entry in manifest["included"]}
    recipe = json.loads(Path(__file__).with_name("NATIVE_RECIPE.json").read_text(encoding="utf-8"))
    expected = dict(recipe["verified_source_sha256"])
    expected.update(CORE_SHA256)
    source = {}
    for rel, digest in expected.items():
        content = (root / rel).read_bytes()
        entry = entries.get(rel)
        _require(entry is not None, "missing source manifest entry: " + rel)
        _require(hashlib.sha256(content).hexdigest() == digest == entry["sha256"], "source SHA256 mismatch: " + rel)
        _require(len(content) == entry["bytes"], "source length mismatch: " + rel)
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        _require(blob == entry["git_blob_sha1"], "Git blob mismatch: " + rel)
        source[rel] = content
    provenance = {
        "repository": AUTHOR_REPOSITORY, "commit": AUTHOR_COMMIT,
        "author_file": str(author_file), "author_sha256": CORE_SHA256[MODEL_REL],
        "verified_sources": expected, "author_code_modified": False,
        "source_url": f"https://github.com/{AUTHOR_REPOSITORY}/blob/{AUTHOR_COMMIT}/{MODEL_REL}",
        "protocol": "complete_author_model_unified_protocol_migration",
        "initialization": "exact extracted original weights_init, applied once after constructor",
        "loss": "exact extracted original HNN.step; logging-only shim replaces Lightning",
        "age_migration": "prepend [current_observed_cycle/train_q95,1,0*16] metadata row",
        "output_units": "mu_norm and sigma_norm; multiply BOTH by 125 for cycles",
        "native_protocol_reproduced": False,
    }
    return source, provenance


class _VerifiedLoader(importlib.machinery.SourceFileLoader):
    """Execute already verified bytes, never stale pyc or a modified reread."""
    def __init__(self, name, path, source):
        super().__init__(name, path)
        self._source = source

    def get_code(self, fullname):
        return compile(self._source, self.path, "exec", dont_inherit=True)


def _extract_function(source, filename, name, class_name=None):
    """Compile the COMPLETE original definition with only its required globals."""
    tree = ast.parse(source, filename=filename)
    nodes = tree.body
    if class_name is not None:
        nodes = next(n.body for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "nn": nn, "F": F, "__name__": "_verified_author_hnn_functions"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def load_author_api(author_file, manifest_path=None):
    source, provenance = verify_author_sources(author_file, manifest_path)
    # The author omits these kwargs. Reject semantic changes in a future runtime.
    defaults = inspect.signature(F.gaussian_nll_loss).parameters
    _require(defaults["full"].default is False and defaults["eps"].default == 1e-6
             and defaults["reduction"].default == "mean", "runtime GaussianNLL defaults differ from native torch 1.12")
    name = "_bayesrul_inception_" + CORE_SHA256[MODEL_REL][:16]
    loader = _VerifiedLoader(name, str(Path(author_file).resolve()), source[MODEL_REL])
    spec = importlib.util.spec_from_file_location(name, author_file, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)  # Real torchinfo dependency is required; no fake module.
    init = _extract_function(source[INIT_REL], INIT_REL, "weights_init")
    step = _extract_function(source[HNN_REL], HNN_REL, "step", "HNN")
    return {"module": module, "weights_init": init, "hnn_step": step, "provenance": provenance}


def pack_window(data, query, history, points):
    """Causal physical sequence; reads ONLY raw/age/windows, never labels/units.

    Caller must enforce engine identity and train-only physical/age scalers.
    Windows are left-padded and end at the query flight. Sampling is the same
    deterministic endpoint-inclusive linspace/round rule as the AIDGN adapter.
    """
    length = _length(history, points)
    raw, age, windows = data["raw"], data["age"], data["windows"]
    _require(raw.ndim == 3 and raw.shape[1:] == (512, 18) and raw.dtype == torch.float32, "expected raw [F,512,18] FP32")
    _require(age.shape == (len(raw),) and age.dtype == torch.float32, "expected age [F] FP32")
    _require(windows.shape == (len(raw), 70) and windows.dtype in (torch.int32, torch.int64), "expected integer windows [F,70]")
    _require(raw.device == age.device == windows.device, "bundle device mismatch")
    query = torch.as_tensor(query, device=raw.device)
    _require(query.ndim == 1 and query.numel() > 0 and query.dtype in (torch.int32, torch.int64), "nonempty integer query required")
    query = query.long()
    _require(bool(((query >= 0) & (query < len(raw))).all()), "query out of bounds")
    chosen = windows[query, -history:].long()
    valid = chosen >= 0
    _require(bool(((chosen >= -1) & (chosen <= query[:, None])).all()), "future or invalid flight")
    _require(bool((chosen[:, -1] == query).all()), "query must be the last physical flight")
    _require(not bool((valid[:, :-1] & ~valid[:, 1:]).any()), "padding must be a left prefix")
    _require(not bool(((chosen[:, 1:] <= chosen[:, :-1]) & valid[:, 1:] & valid[:, :-1]).any()), "history must strictly increase")
    positions = torch.linspace(0, 511, points, dtype=torch.float64, device=raw.device).round().long()
    safe = chosen.clamp_min(0)
    selected = raw[safe[:, :, None], positions[None, None, :], :]
    selected_age = age[safe, None, None].expand(-1, -1, points, 1)
    mask = valid[:, :, None, None]
    packed = torch.where(mask, selected, torch.zeros_like(selected)).reshape(len(query), length, 18)
    packed_age = torch.where(mask, selected_age, torch.zeros_like(selected_age)).reshape(len(query), length, 1)
    _require(bool(torch.isfinite(packed).all()) and bool(torch.isfinite(packed_age).all()), "nonfinite selected data")
    return packed, packed_age


class BayesHNNAdapter(nn.Module):
    """Full original model, original initialization and loss; no optimizer here.

    Seed the entire pipeline BEFORE construction. ``device`` is optional; .to()
    is also supported. Input T=history*points; author T=T+1 after metadata.
    """
    def __init__(self, author_file, history, points, manifest_path=None, device=None):
        super().__init__()
        self.history, self.points = history, points
        self.physical_sequence_len = _length(history, points)
        self.sequence_len = self.physical_sequence_len + 1
        api = load_author_api(author_file, manifest_path)
        self.provenance = api["provenance"]
        self.author_sha256 = self.provenance["author_sha256"]
        self.author_commit = AUTHOR_COMMIT
        self.original_args = {"win_length": self.sequence_len, "n_features": 18, **NATIVE_NET_ARGS}
        self.author_model = api["module"].Inception(**self.original_args)
        self.author_model.apply(api["weights_init"])
        self._hnn_step = api["hnn_step"]
        if device is not None:
            self.to(device=device)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def encode_input(self, raw_window, age_window):
        _require(raw_window.ndim == 3 and raw_window.shape[1:] == (self.physical_sequence_len, 18), "wrong physical input shape")
        _require(age_window.shape == (*raw_window.shape[:2], 1), "wrong age input shape")
        _require(raw_window.dtype == age_window.dtype == torch.float32, "FP32 inputs required")
        _require(raw_window.device == age_window.device, "input device mismatch")
        _require(bool(torch.isfinite(raw_window).all()) and bool(torch.isfinite(age_window).all()), "nonfinite input")
        _require(bool((age_window >= 0).all()), "observed age must be nonnegative")
        metadata = raw_window.new_zeros((raw_window.shape[0], 1, 18))
        metadata[:, 0, 0] = age_window[:, -1, 0]
        metadata[:, 0, 1] = 1.0
        return torch.cat((metadata, raw_window), dim=1)

    def forward(self, raw_window, age_window):
        output = self.author_model(self.encode_input(raw_window, age_window))
        return {"mu_norm": output[:, 0], "sigma_norm": output[:, 1]}

    def nll(self, mu_norm, target_norm, sigma_norm):
        """Execute original HNN.step on already computed predictions (no reforward).

        Targets are required in [0,1] (min(raw_RUL,125)/125). No target scaling
        occurs here. The tiny shell replaces ONLY Lightning's logger/net call.
        """
        _require(mu_norm.ndim == 1 and target_norm.shape == sigma_norm.shape == mu_norm.shape, "NLL vector shape mismatch")
        _require(bool(torch.isfinite(target_norm).all()) and bool(((target_norm >= 0) & (target_norm <= 1)).all()), "expected capped normalized RUL target")
        output = torch.stack((mu_norm, sigma_norm), dim=1)
        class _LossOnlyShell:
            def net(self, unused):
                return output

            def log(self, *args, **kwargs):
                pass

        return self._hnn_step(_LossOnlyShell(), (None, target_norm), "train")[0]
