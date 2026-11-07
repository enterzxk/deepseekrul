"""Zero-intrusion adapter that exposes STG_MambaNet features for PU-Net.

STG_MambaNet.forward returns only (B, 1); the representation that feeds its
regression head is cat([fused[:, -1, :], fused.mean(dim=1)], dim=-1). This
adapter reproduces exactly that tensor by hooking the fusion module at run time,
so no file under src/ is modified.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class STGFeatureExtractor:
    """Callable feature_fn for PUNetModel.

    Args:
        fusion_attr: attribute name of the fusion module on the backbone.
        hook: if True, capture fused activations with a temporary hook; if
            False, call backbone.fusion directly after a forward pass.
    """

    def __init__(self, fusion_attr: str = "fusion", hook: bool = True) -> None:
        self.fusion_attr = fusion_attr
        self.hook = bool(hook)
        self._captured: Optional[torch.Tensor] = None
        self._handle = None

    def __call__(self, backbone: nn.Module, x: torch.Tensor, adjacency=None):
        module = getattr(backbone, self.fusion_attr, None)
        if module is None:
            raise AttributeError(f"backbone has no attribute {self.fusion_attr!r}")

        self._captured = None
        self._handle = module.register_forward_hook(self._grab)
        try:
            with torch.no_grad():
                pred = backbone(x)
        finally:
            if self._handle is not None:
                self._handle.remove()
                self._handle = None

        fused = self._captured
        if fused is None:
            raise RuntimeError("fusion hook captured nothing; check fusion_attr")
        if isinstance(fused, (tuple, list)):
            fused = fused[0]
        if fused.ndim != 3:
            raise ValueError(f"expected fused (B, T, D), got {tuple(fused.shape)}")

        last = fused[:, -1, :]
        mean = fused.mean(dim=1)
        h = torch.cat([last, mean], dim=-1)  # (B, 2D), same as the regression head input
        y_base = pred.reshape(-1) if pred.ndim <= 2 else pred.reshape(pred.shape[0], -1).mean(dim=-1)
        return h, y_base

    def _grab(self, module, inputs, output) -> None:
        self._captured = output[0] if isinstance(output, (tuple, list)) else output


def d_model_for(backbone: nn.Module) -> int:
    """Width of the tensor produced by STGFeatureExtractor (2 * backbone d_model)."""
    base = getattr(backbone, "d_model", None)
    if base is None:
        raise AttributeError("backbone has no d_model attribute")
    return int(base) * 2
