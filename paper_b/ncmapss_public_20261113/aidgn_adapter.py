"""Complete pinned author AIDGN with an explicit observed-input migration.

No target, label, unit identifier or future lifetime is consumed. The caller
supplies bundle['age'] = observed cycle / training-engine q95. This differs
from the author's NC cycle*1e-5 preprocessing, not from the model mathematics.

Actual NC defaults: AATE_dim16, hidden_dim128, fc_layer_dim512,
feature_fc_layer_dim=4*feature_num (author20->80; observed18->72),
dropout0.1, nlayer1, hop1, outlayer='Linear', reshape=True. Source parser:
N_CMAPSS_train.py:82,92-94,111,113; NC overrides: hparams.py:214,252,290;
constructor rules: Experiment.py:201-213 and AIDGN.py:16. Only input feature
count and sequence length (with the corresponding author *4 width rule)
change. All graph gates, graph convolutions and both decoders remain author code.

Actual Adam LR: DS02=0.0005, DS05/06/07=0.001 (hparams.py:233,271).
Adam weight_decay is 0 because Experiment.py:242-243 passes only lr.
Author training defaults are300 epochs/B128 and nonnegative RUL/max_life
targets (is_minmax=True), not centered targets. Controller's200/B64, inner
window selection and positive125 scaling are explicit unified-protocol changes.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

AUTHOR_REPOSITORY = "TCCofWANG/AIDGN-RUL"
AUTHOR_COMMIT = "d1ac173b13659af1854cf500a9b1cba8367353fe"
AUTHOR_MODEL_SHA256 = "22101fe6bc88fa792b68c909e21d70718e85b919097bf00b177586154ff245b6"
DEFAULT_AUTHOR_FILE = Path(__file__).resolve().parents[1] / "ncmapss_response/upstream/aidgn/Model/AIDGN.py"
WINDOW_CATALOG = ((1, 50), (1, 512), (20, 16), (70, 4), (70, 512))
# N_CMAPSS_train.py parser, then hparams.N_CMAPSS AIDGN overrides:
AUTHOR_ENTRY_DEFAULTS = {
    "input_length": 50, "AATE_dim": 16, "d_model": 128, "dropout": .1,
    "nlayer": 1, "hop": 1, "loss_type": "MSE", "is_minmax": True,
    "batch_size": 128, "train_epochs": 300, "ablation_mode": "linear",
}
# Experiment.py:201-213 derives these; feature_num and sequence_len are the
# only migration dimensions. feature_fc_layer_dim follows the author's *4 rule.
AUTHOR_MODEL_CONSTANTS = {
    "AATE_dim": 16, "hidden_dim": 128, "fc_layer_dim": 512,
    "fc_dropout": .1, "nlayer": 1, "hop": 1, "outlayer": "Linear", "reshape": True,
}
AUTHOR_NATIVE_MODEL_DEFAULTS = {
    **AUTHOR_MODEL_CONSTANTS, 'feature_num': 20, 'sequence_len': 50,
    'feature_fc_layer_dim': 80, 'supports': None,
}
AUTHOR_CONSTRUCTOR_RULES = {
    'feature_num': 'input_feature; source NC20, migration observed18',
    'sequence_len': 'input_length; source50, migration history*points',
    'hidden_dim': 'd_model=128', 'fc_layer_dim': '4*hidden_dim=512',
    'feature_fc_layer_dim': '4*feature_num; source80, migration72',
    'fc_dropout': 'dropout=0.1', 'AATE_dim': 16, 'nlayer': 1, 'hop': 1,
    'outlayer': 'Linear forced by Experiment._get_model',
    'reshape': 'True forced by author AIDGN.__init__', 'supports': None,
}
AUTHOR_LEARNING_RATE = {"DS02": 5e-4, "DS05": 1e-3, "DS06": 1e-3, "DS07": 1e-3}
AUTHOR_OPTIMIZER = {
    "name": "Adam", "actual_weight_decay": 0.,
    "note": "Experiment._select_optimizer passes lr only; declared hparams weight_decay=1e-4 is unused",
}
AUTHOR_LOSS = {
    "name": "MSE", "default_target": "RUL/max_life (is_minmax=True)",
    "head": "absolute nonnegative native output; no target centering in adapter",
    "migration": "controller may use min(raw_RUL,125)/125 and invert by *125; adapter does not read/transform labels",
}
AUTHOR_TRAINING_DEFAULTS = {
    'epochs': 300, 'batch_size': 128, 'is_minmax': True, 'loss': 'MSE',
    'optimizer': 'Adam', 'actual_weight_decay': 0.,
    'early_stop_patience': 10, 'learning_rate_patience': 10, 'learning_rate_factor': .3,
    'source': 'N_CMAPSS_train.py:72,75,84,125,199-202; Experiment.py:242-243,538-548',
}
SOURCE_REFERENCES = {
    "parser": "N_CMAPSS_train.py:75,82,91-94,111,113,125",
    "nc_overrides": "configs/hparams.py:214,233,252,271,290",
    "constructor": "Experiment/Experiment.py:201-213",
    "optimizer": "Experiment/Experiment.py:238-243",
    "loss": "Experiment/Experiment.py:538-548",
}
_MODULES = {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _window(history, points):
    _require(type(history) is int and type(points) is int
             and (history, points) in WINDOW_CATALOG, "window is outside the registered five-view catalog")
    return history * points


def _verify_bytes(source, manifest):
    _require(manifest.get("repository") == AUTHOR_REPOSITORY
             and manifest.get("full_commit_sha") == AUTHOR_COMMIT, "wrong pinned author identity")
    entries = [entry for entry in manifest.get("included", []) if entry["path"] == "Model/AIDGN.py"]
    _require(len(entries) == 1, "author model absent/duplicated in manifest")
    entry = entries[0]
    actual = hashlib.sha256(source).hexdigest()
    _require(entry.get("sha256") == AUTHOR_MODEL_SHA256 == actual, "author model SHA-256 mismatch")
    _require(entry.get("bytes") == len(source), "author model byte-count mismatch")
    git_sha = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    _require(entry.get("git_blob_sha1") == git_sha, "author model Git blob mismatch")
    return {"repository": AUTHOR_REPOSITORY, "commit": AUTHOR_COMMIT, "sha256": actual,
            "git_blob_sha1": git_sha, "source_url": entry["source_url"],
            "license_status": manifest.get("license", {}).get("status"), "source_modified": False}


def _verified_companion(root, manifest, relative):
    entry = next((x for x in manifest["included"] if x["path"] == relative), None)
    _require(entry is not None, f"missing author metadata source: {relative}")
    source = (root / relative).read_bytes()
    _require(hashlib.sha256(source).hexdigest() == entry["sha256"], f"author metadata hash mismatch: {relative}")
    return ast.parse(source.decode("utf-8"))


def verify_author_defaults(author_file, manifest):
    """Check parser/NC overrides by AST, never execute author entrypoints."""
    root = Path(author_file).resolve().parent.parent
    tree = _verified_companion(root, manifest, "N_CMAPSS_train.py")
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                option = node.args[0].value
                default = next((k.value for k in node.keywords if k.arg == "default"), None)
                if default is not None:
                    try:
                        defaults[option.removeprefix("--")] = ast.literal_eval(default)
                    except (ValueError, TypeError):
                        pass
    for key, value in AUTHOR_ENTRY_DEFAULTS.items():
        _require(defaults.get(key) == value, f"author parser default changed: {key}")
    hp = _verified_companion(root, manifest, "configs/hparams.py")
    nc = next(node for node in hp.body if isinstance(node, ast.ClassDef) and node.name == "N_CMAPSS")
    model_dicts = []
    for node in ast.walk(nc):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "alg_hparams":
                    model_dicts.append(ast.literal_eval(node.value)["AIDGN"])
    _require(len(model_dicts) == 3 and all(d == {"input_length": 50, "AATE_dim": 16, "d_model": 128}
                                         for d in model_dicts), "author NC model override changed")
    # Verify the actual two independent if statements and attached else:
    # DS01's first assignment is overwritten by the second if's else.
    init = next(n for n in nc.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    actual_lr = {}
    for subset in AUTHOR_LEARNING_RATE:
        selected = {}
        for node in init.body:
            if isinstance(node, ast.If):
                test = node.test
                _require(isinstance(test, ast.Compare) and len(test.comparators) == 1
                         and isinstance(test.comparators[0], ast.Constant), "unsupported author hparams control flow")
                branch = node.body if subset == test.comparators[0].value else node.orelse
                for statement in branch:
                    if isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Attribute) and target.attr == "train_params":
                                selected = ast.literal_eval(statement.value)
        actual_lr[subset] = selected["AIDGN"]["learning_rate"]
    _require(actual_lr == AUTHOR_LEARNING_RATE, "actual NC author learning rates changed")
    _verified_companion(root, manifest, "Experiment/Experiment.py")
    return {"parser_defaults": dict(AUTHOR_ENTRY_DEFAULTS), "model_constants": dict(AUTHOR_MODEL_CONSTANTS),
            "actual_learning_rate": actual_lr, "references": dict(SOURCE_REFERENCES)}


class _VerifiedLoader(importlib.machinery.SourceFileLoader):
    """Execute exactly verified bytes, without reading or writing __pycache__."""
    def __init__(self, name, path, source):
        super().__init__(name, path)
        self._verified_source = source

    def get_code(self, fullname):
        return compile(self._verified_source, self.path, "exec", dont_inherit=True)


def load_author_class(author_file=DEFAULT_AUTHOR_FILE, manifest_path=None):
    author_file = Path(author_file).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path is not None else author_file.parent.parent / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = author_file.read_bytes()
    provenance = _verify_bytes(source, manifest)
    provenance["default_verification"] = verify_author_defaults(author_file, manifest)
    provenance['author_native_defaults'] = dict(AUTHOR_NATIVE_MODEL_DEFAULTS)
    provenance['constructor_rules'] = dict(AUTHOR_CONSTRUCTOR_RULES)
    provenance['author_training_defaults'] = dict(AUTHOR_TRAINING_DEFAULTS)
    provenance['author_optimizer'] = dict(AUTHOR_OPTIMIZER)
    provenance["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance["author_file"] = str(author_file)
    key = (str(author_file), provenance["sha256"])
    if key not in _MODULES:
        name = "_ncmapss_pinned_author_aidgn_" + provenance["sha256"][:16]
        loader = _VerifiedLoader(name, str(author_file), source)
        spec = importlib.util.spec_from_file_location(name, author_file, loader=loader)
        _require(spec is not None, "cannot construct author module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _MODULES[key] = module
    cls = _MODULES[key].AIDGN
    _require(cls.__module__ == _MODULES[key].__name__, "unexpected author class")
    return cls, provenance


def author_arguments(history, points):
    length = _window(history, points)
    return SimpleNamespace(**AUTHOR_MODEL_CONSTANTS, feature_num=18, sequence_len=length,
                           feature_fc_layer_dim=18 * 4)


class AIDGNAdapter(nn.Module):
    """Exact author graph/head. Output is the native nonnegative head, [B]."""
    def __init__(self, author_file=DEFAULT_AUTHOR_FILE, history=1, points=50, manifest_path=None):
        super().__init__()
        self.history, self.points = history, points
        self.sequence_len = _window(history, points)
        self.author_class, self.provenance = load_author_class(author_file, manifest_path)
        self.author_args = author_arguments(history, points)
        self.author_model = self.author_class(self.author_args)
        self.provenance["migration"] = {
            "feature_num": 18, "sequence_len": self.sequence_len, "history": history, "points": points,
            "aging": "observed cycle / training-only q95, supplied by bundle['age']; not labels or lifetime",
            "additional_temporal_head": False, "model_math_modified": False,
            "device_compatibility_only": "author_model.device assigned to current input.device",
            "target_semantics": dict(AUTHOR_LOSS),
        }

    def forward(self, raw_window, age_window):
        _require(isinstance(raw_window, torch.Tensor) and isinstance(age_window, torch.Tensor), "tensor inputs required")
        _require(raw_window.ndim == 3 and raw_window.shape[1:] == (self.sequence_len, 18)
                 and raw_window.shape[0] > 0, "expected raw [B,T,18] for registered T")
        _require(age_window.shape == (raw_window.shape[0], self.sequence_len, 1), "expected age [B,T,1]")
        _require(raw_window.dtype == age_window.dtype == torch.float32, "registered interface requires FP32")
        _require(raw_window.device == age_window.device, "raw/age device mismatch")
        _require(next(self.author_model.parameters()).device == raw_window.device, "move adapter to input device before forward")
        _require(bool(torch.isfinite(raw_window).all()) and bool(torch.isfinite(age_window).all()), "nonfinite input")
        # This plain device attribute is not updated by nn.Module.to().
        self.author_model.device = raw_window.device
        output = self.author_model(raw_window, age_window)
        _require(isinstance(output, tuple) and len(output) == 2, "author forward API changed")
        prediction = output[1]
        _require(prediction.numel() == raw_window.shape[0], "author head does not return one RUL per query")
        return prediction.reshape(raw_window.shape[0])


def pack_window(data, query, history, points):
    """Read ONLY raw/age/windows; trust caller's engine/split boundary checks.

    Index-causality and left padding are checked here. No unit IDs are available
    at this interface, so cross-engine identity is the frozen bundle's duty.
    """
    length = _window(history, points)
    raw, age, windows = data["raw"], data["age"], data["windows"]
    _require(raw.ndim == 3 and raw.shape[1:] == (512, 18) and raw.dtype == torch.float32, "expected raw [F,512,18] FP32")
    _require(age.shape == (len(raw),) and age.dtype == torch.float32, "expected age [F] FP32")
    _require(windows.shape == (len(raw), 70) and windows.dtype in (torch.int32, torch.int64), "expected integer windows [F,70]")
    _require(raw.device == age.device == windows.device, "bundle device mismatch")
    query = torch.as_tensor(query, device=raw.device)
    _require(query.ndim == 1 and query.numel() > 0 and query.dtype in (torch.int32, torch.int64), "integer nonempty query vector required")
    query = query.to(torch.long)
    _require(bool(((query >= 0) & (query < len(raw))).all()), "query out of range")
    chosen = windows[query, -history:].to(torch.long)
    valid = chosen >= 0
    _require(bool(((chosen >= -1) & (chosen <= query[:, None])).all()), "future or invalid flight in window")
    _require(bool((chosen[:, -1] == query).all()), "window must end at its query flight")
    _require(not bool((valid[:, :-1] & ~valid[:, 1:]).any()), "padding must be a left prefix")
    _require(not bool(((chosen[:, 1:] <= chosen[:, :-1]) & valid[:, 1:] & valid[:, :-1]).any()), "history indices must increase strictly")
    positions = torch.linspace(0, 511, points, dtype=torch.float64, device=raw.device).round().to(torch.long)
    safe = chosen.clamp_min(0)
    selected = raw[safe[:, :, None], positions[None, None, :], :]
    selected_age = age[safe, None, None].expand(-1, -1, points, 1)
    mask = valid[:, :, None, None]
    packed = torch.where(mask, selected, torch.zeros_like(selected)).reshape(len(query), length, 18)
    packed_age = torch.where(mask, selected_age, torch.zeros_like(selected_age)).reshape(len(query), length, 1)
    _require(bool(torch.isfinite(packed).all()) and bool(torch.isfinite(packed_age).all()), "nonfinite selected observations")
    return packed, packed_age

