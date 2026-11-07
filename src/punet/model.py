"""PU-Net wrapper: frozen backbone plus two small heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from .features import PriorFeatures
from .heads import BiasHead, ScaleHead


@dataclass
class PUNetOutput:
    y_hat: torch.Tensor
    y_base: torch.Tensor
    w: torch.Tensor
    h: torch.Tensor


class PUNetModel(nn.Module):
    """Frozen backbone + scale head + bias head.

    Args:
        backbone: the IDA-PE model. It is frozen in train mode.
        d_model: hidden width of the representation h.
        feature_fn: optional callable (backbone, x, adjacency) -> (h, y_base).
            When omitted the wrapper looks for backbone.encode(x, adjacency),
            then backbone.encode(x), and finally falls back to a forward hook on
            the submodule named by hook_name.
        hook_name: submodule name used by the hook fallback.
        use_scale_head: ablation A1 (False -> w == 1).
        use_bias_head: ablation A6 (False -> b == 0).
        use_sigma: ablation A2 (False -> sigma replaced by 1.0).
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        *,
        feature_fn: Optional[Callable] = None,
        hook_name: Optional[str] = None,
        use_scale_head: bool = True,
        use_bias_head: bool = True,
        use_sigma: bool = True,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = int(d_model)
        self.feature_fn = feature_fn
        self.hook_name = hook_name
        self.use_scale_head = bool(use_scale_head)
        self.use_bias_head = bool(use_bias_head)
        self.use_sigma = bool(use_sigma)

        self.scale_head = ScaleHead(self.d_model, hidden=hidden) if self.use_scale_head else None
        self.bias_head = BiasHead(self.d_model, hidden=hidden) if self.use_bias_head else None

        self._captured: Optional[torch.Tensor] = None
        self._hook = None
        if self.feature_fn is None and not hasattr(self.backbone, "encode") and hook_name is not None:
            module = dict(self.backbone.named_modules()).get(hook_name)
            if module is None:
                raise ValueError(f"hook_name {hook_name!r} not found in backbone")
            self._hook = module.register_forward_hook(self._capture_hook)

        self._freeze_backbone()

    # ------------------------------------------------------------------ utils
    def _capture_hook(self, module, inputs, output) -> None:
        self._captured = output[0] if isinstance(output, (tuple, list)) else output

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "PUNetModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _run_backbone(self, x: torch.Tensor, adjacency):
        with torch.no_grad():
            if self.feature_fn is not None:
                h, y_base = self.feature_fn(self.backbone, x, adjacency)
            elif hasattr(self.backbone, "encode"):
                try:
                    h, y_base = self.backbone.encode(x, adjacency)
                except TypeError:
                    h, y_base = self.backbone.encode(x)
            else:
                out = self.backbone(x) if adjacency is None else self.backbone(x, adjacency)
                y_base = out[0] if isinstance(out, (tuple, list)) else out
                h = self._captured
                if h is None:
                    raise RuntimeError(
                        "no representation captured; pass feature_fn or a valid hook_name"
                    )
                if h.ndim > 2:
                    h = h.flatten(1)
                if h.shape[-1] != self.d_model:
                    raise ValueError(f"captured h has width {h.shape[-1]}, expected {self.d_model}")
        return h.detach(), y_base.detach()

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, adjacency=None, feats: Optional[PriorFeatures] = None) -> PUNetOutput:
        if feats is None:
            raise ValueError("feats is required")
        h, y_base = self._run_backbone(x, adjacency)
        y_base = y_base.reshape(-1)

        sigma = feats.sigma.to(h.device)
        tau = feats.tau.to(h.device)
        if not self.use_sigma:
            sigma = torch.ones_like(sigma)

        w = torch.ones_like(y_base) if self.scale_head is None else self.scale_head(h, sigma, tau)
        b = torch.zeros_like(y_base) if self.bias_head is None else self.bias_head(h, sigma)
        return PUNetOutput(y_hat=y_base + b, y_base=y_base, w=w, h=h)
