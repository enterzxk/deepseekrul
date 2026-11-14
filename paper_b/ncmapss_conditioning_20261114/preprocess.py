"""Fixed train-four quadratic conditioning; NumPy fit, optional Torch application.

This is a classical fitted preprocessing reference, not a learned neural module.
All fitting uses exactly the frozen bundle's FP32 global-standardized coordinates.
The residual standard deviation is empirical scale, never a noise estimate.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def quadratic_numpy(w):
    require(w.shape[-1] == 4, "quadratic W must have four columns")
    return np.stack([np.ones(w.shape[:-1], np.float64)] + [w[..., k] for k in range(4)] +
                    [w[..., i] * w[..., j] for i in range(4) for j in range(i, 4)], axis=-1)


def global_standardize(raw, scalers):
    """Exact coordinate/evaluation order used by frozen bundle before torch.as_tensor."""
    raw = np.asarray(raw, dtype=np.float32)
    mu, sd = (np.asarray(scalers[k], dtype=np.float32) for k in ("raw_mu", "raw_sd"))
    require(mu.shape == sd.shape == (18,) and bool(np.isfinite(mu).all()) and
            bool(np.isfinite(sd).all()) and bool((sd > 0).all()), "invalid global coordinates")
    return (raw - mu) / sd


def fit_processor(data, train_units, scalers, cache_identity=None):
    """Read only data['raw']/['unit']; consume no random values or target arrays."""
    units = list(map(int, train_units))
    require(len(units) == len(set(units)) == 4 and units == sorted(units), "four sorted unique training engines required")
    require(scalers["fitted_units"] == units, "global scalers have different fitting engines")
    raw, unit = data["raw"], data["unit"]
    require(raw.ndim == 3 and raw.shape[1:] == (512, 18) and unit.shape == (len(raw),), "wrong physical data shape")
    selected = raw[np.isin(unit, units)]
    require(len(selected) > 0 and set(map(int, np.unique(unit[np.isin(unit, units)]))) == set(units), "missing training engines")
    # Independently reject supplied coordinates fitted on other engines.
    physical = selected.astype(np.float64)
    require(np.array_equal(np.asarray(scalers["raw_mu"]), physical.mean((0, 1))) and
            np.array_equal(np.asarray(scalers["raw_sd"]), np.maximum(physical.std((0, 1)), 1e-6)), "global scaler is not train-four mean/std")
    standardized = global_standardize(selected, scalers).astype(np.float64)
    require(bool(np.isfinite(standardized).all()), "nonfinite training coordinates")
    phi = quadratic_numpy(standardized[..., :4]).reshape(-1, 15)
    target = standardized[..., 4:].reshape(-1, 14)
    gram = phi.T @ phi / len(phi)
    penalty = np.eye(15) * .001; penalty[0, 0] = 0.
    coef = np.linalg.solve(gram + penalty, phi.T @ target / len(phi))
    residual = target - phi @ coef
    scale_unfloored = residual.std(axis=0)
    scale = np.maximum(scale_unfloored, 1e-6)
    require(bool(np.isfinite(coef).all() and np.isfinite(scale).all()), "nonfinite fixed processor")
    result = {"kind": "fixed_classical_quadratic_empirical_residual_conditioning", "fitted_units": units,
              "global_scalers_sha256": identity(scalers), "cache_identity": cache_identity,
              "fit_coordinate": "frozen_bundle_FP32_global_standardize_then_FP64",
              "coefficients": coef.tolist(), "empirical_residual_scale": scale.tolist(),
              "empirical_residual_scale_unfloored": scale_unfloored.tolist(),
              "floor": 1e-6, "ridge": .001, "intercept_penalized": False,
              "training_flights": len(selected), "training_rows": len(phi),
              "gram_condition_number": float(np.linalg.cond(gram + penalty)),
              "fixed_estimated_coefficients": 210, "fixed_empirical_scale_values": 14,
              "additional_neural_trainable_parameters": 0, "label_arrays_accessed": False}
    result["processor_sha256"] = identity(result)
    return result


def verify_processor(processor):
    require(identity({k: v for k, v in processor.items() if k != "processor_sha256"}) == processor["processor_sha256"], "processor digest mismatch")
    coef = np.asarray(processor["coefficients"], np.float64)
    scale = np.asarray(processor["empirical_residual_scale"], np.float64)
    require(coef.shape == (15, 14) and scale.shape == (14,) and bool(np.isfinite(coef).all()) and
            bool(np.isfinite(scale).all()) and bool((scale >= 1e-6).all()), "invalid processor tensors")
    return coef, scale


def apply_numpy(raw_standardized, processor, conditioning="resid"):
    require(conditioning in ("raw", "resid"), "unknown conditioning")
    require(raw_standardized.dtype == np.float32 and raw_standardized.shape[-1] == 18, "expected already-standardized FP32 physical18")
    coef, scale = verify_processor(processor)
    result = raw_standardized.copy()
    if conditioning == "resid":
        double = raw_standardized.astype(np.float64)
        result[..., 4:] = ((double[..., 4:] - quadratic_numpy(double[..., :4]) @ coef) / scale).astype(np.float32)
    require(bool(np.isfinite(result).all()), "nonfinite conditioned input")
    return result


def restore_numpy(conditioned, processor):
    coef, scale = verify_processor(processor)
    double = conditioned.astype(np.float64)
    result = double.copy()
    result[..., 4:] = double[..., 4:] * scale + quadratic_numpy(double[..., :4]) @ coef
    return result


def apply_bundle(bundle, processor, conditioning="resid"):
    """Preserve every non-raw entry by identity, including labels and causal metadata."""
    import torch
    require(conditioning in ("raw", "resid"), "unknown conditioning")
    coef, scale = verify_processor(processor)
    original = bundle["raw"]
    require(original.dtype == torch.float32 and original.shape[-1] == 18, "expected frozen bundle FP32 physical18")
    result = dict(bundle)
    if conditioning == "resid":
        double = original.double(); w = double[..., :4]
        phi = torch.stack([torch.ones_like(w[..., 0])] + [w[..., k] for k in range(4)] +
                          [w[..., i] * w[..., j] for i in range(4) for j in range(i, 4)], dim=-1)
        c = torch.as_tensor(coef, dtype=torch.float64, device=original.device)
        s = torch.as_tensor(scale, dtype=torch.float64, device=original.device)
        residual = ((double[..., 4:] - phi @ c) / s).float()
        result["raw"] = torch.cat((original[..., :4], residual), dim=-1)
    require(bool(torch.isfinite(result["raw"]).all()), "nonfinite conditioned bundle")
    require(torch.equal(original[..., :4], result["raw"][..., :4]), "W changed during conditioning")
    require(all(result[k] is value for k, value in bundle.items() if k != "raw"), "nonphysical bundle metadata changed")
    return result
