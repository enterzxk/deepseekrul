"""PU-Net: prior-uncertainty-scaled conformal intervals for RUL.

Zero-intrusion package: nothing under src/ outside this directory is modified.

Submodules are imported lazily so the numpy-only modules (conformal, metrics,
diagnostics) stay usable without torch installed.
"""

from __future__ import annotations

_LAZY = {
    "PriorFeatures": ("features", "PriorFeatures"),
    "PriorFeatureBuilder": ("features", "PriorFeatureBuilder"),
    "ScaleHead": ("heads", "ScaleHead"),
    "BiasHead": ("heads", "BiasHead"),
    "PUNetModel": ("model", "PUNetModel"),
    "PUNetOutput": ("model", "PUNetOutput"),
    "BlockConformalCalibrator": ("conformal", "BlockConformalCalibrator"),
    "BlockSplit": ("conformal", "BlockSplit"),
    "block_split": ("conformal", "block_split"),
    "picp": ("metrics", "picp"),
    "pinaw": ("metrics", "pinaw"),
    "winkler": ("metrics", "winkler"),
    "engine_coverage": ("metrics", "engine_coverage"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    from importlib import import_module

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value
    return value
