"""STG-MambaNet: spatial-temporal gated network for C-MAPSS RUL prediction.

The model is implemented entirely with native PyTorch operations and contains:

1. An adaptive spatial GCN branch with a learned 14x14 sensor graph.
2. A multi-scale selective state-space temporal branch using depthwise
   Conv1d, linear selection projections, and SiLU gating.
3. Cross-attention plus sample-adaptive and channel-adaptive gated fusion.
4. A regression head that returns one RUL value per sample.

No custom CUDA extension, mamba package, or compiled third-party kernel is
required.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    """One residual graph-convolution layer over sensor nodes."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D), adjacency: (N, N)
        aggregated = torch.einsum("ij,btjf->btif", adjacency, x)
        update = self.proj(aggregated)
        update = self.dropout(F.gelu(update))
        update = self.norm(update)
        return x + update


class AdaptiveSpatialBranch(nn.Module):
    """Adaptive spatial branch with an optional identity prior.

    For graph_mode="learnable" the row-stochastic graph is

        A = (1 - alpha) I + alpha * Softmax(ReLU(E1 @ E2.T), dim=-1)

    where alpha is a learnable scalar initialised to 0.1.  The identity term
    keeps the branch close to independent sensor channels early in training
    and lets the model discover only the cross-sensor coupling that is useful
    for the current engine population.  graph_mode="identity" fixes A=I;
    graph_mode="random_fixed" uses a fixed row-stochastic random matrix.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        d_model: int = 64,
        graph_embed_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        graph_mode: Literal["learnable", "identity", "random_fixed", "hollow"] = "learnable",
        identity_prior_alpha: float = 0.1,
        hollow_alpha: float = 0.35,
    ) -> None:
        super().__init__()
        if graph_mode not in {"learnable", "identity", "random_fixed", "hollow"}:
            raise ValueError(
                "graph_mode must be 'learnable', 'identity', 'random_fixed', or 'hollow'."
            )
        if not 0.0 <= identity_prior_alpha <= 1.0:
            raise ValueError("identity_prior_alpha must lie in [0, 1].")
        if not 0.0 <= hollow_alpha <= 1.0:
            raise ValueError("hollow_alpha must lie in [0, 1].")
        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        self.graph_mode = graph_mode
        self.identity_prior_alpha = float(identity_prior_alpha)
        self.hollow_alpha = float(hollow_alpha)
        self.node_proj = nn.Linear(1, d_model)

        if graph_mode in {"learnable", "hollow"}:
            self.E1 = nn.Parameter(torch.empty(num_sensors, graph_embed_dim))
            self.E2 = nn.Parameter(torch.empty(num_sensors, graph_embed_dim))
            nn.init.xavier_uniform_(self.E1)
            nn.init.xavier_uniform_(self.E2)
            if graph_mode == "learnable":
                alpha = min(max(self.identity_prior_alpha, 1.0e-4), 1.0 - 1.0e-4)
                self.alpha_logit = nn.Parameter(
                    torch.tensor(math.log(alpha / (1.0 - alpha)))
                )
        elif graph_mode == "random_fixed":
            random_logits = torch.randn(num_sensors, num_sensors)
            self.register_buffer(
                "fixed_adjacency",
                torch.softmax(random_logits, dim=-1),
            )

        self.layers = nn.ModuleList(
            [AdaptiveGraphConv(d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.node_score = nn.Linear(d_model, 1)
        self.out_norm = nn.LayerNorm(d_model)

    @property
    def graph_alpha(self) -> torch.Tensor | None:
        """Return the mixing coefficient for the identity prior."""
        if self.graph_mode == "learnable":
            return torch.sigmoid(self.alpha_logit)
        if self.graph_mode == "hollow":
            return self.node_proj.weight.new_tensor(self.hollow_alpha)
        return None

    def _base_adjacency(self) -> torch.Tensor:
        if self.graph_mode in {"learnable", "hollow"}:
            logits = torch.relu(torch.matmul(self.E1, self.E2.transpose(0, 1)))
            if self.graph_mode == "hollow":
                logits = logits / math.sqrt(self.E1.shape[1])
                mask = torch.eye(
                    self.num_sensors, device=logits.device, dtype=torch.bool
                )
                logits = logits.masked_fill(mask, -1.0e9)
            return torch.softmax(logits, dim=-1)
        if self.graph_mode == "random_fixed":
            return self.fixed_adjacency
        device = self.node_proj.weight.device
        dtype = self.node_proj.weight.dtype
        return torch.eye(self.num_sensors, device=device, dtype=dtype)

    def get_cross_adjacency(self) -> torch.Tensor:
        """Return the pure off-diagonal component (zero diagonal)."""
        base = self._base_adjacency()
        if self.graph_mode == "hollow":
            return base
        if self.graph_mode == "learnable":
            eye = torch.eye(
                self.num_sensors, device=base.device, dtype=torch.bool
            )
            return base.masked_fill(eye, 0.0)
        return torch.zeros_like(base)

    def get_adjacency(self) -> torch.Tensor:
        """Return the current row-stochastic sensor adjacency matrix."""
        adjacency = self._base_adjacency()
        if self.graph_mode not in {"learnable", "hollow"}:
            return adjacency
        identity = torch.eye(
            self.num_sensors,
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
        alpha = (
            torch.sigmoid(self.alpha_logit)
            if self.graph_mode == "learnable"
            else self.node_proj.weight.new_tensor(self.hollow_alpha)
        )
        return (1.0 - alpha) * identity + alpha * adjacency

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, N)
        if x.ndim != 3 or x.shape[-1] != self.num_sensors:
            raise ValueError(
                f"Expected (B, T, {self.num_sensors}) input, got {tuple(x.shape)}."
            )

        adjacency = self.get_adjacency()
        h = self.node_proj(x.unsqueeze(-1))  # (B, T, N, D)
        for layer in self.layers:
            h = layer(h, adjacency)

        # Learn how much each sensor contributes to the joint spatial state.
        node_logits = self.node_score(h).squeeze(-1)  # (B, T, N)
        node_weights = torch.softmax(node_logits, dim=-1).unsqueeze(-1)
        spatial = (node_weights * h).sum(dim=2)  # (B, T, D)
        return self.out_norm(spatial), adjacency


class IndependentSpatialBranch(nn.Module):
    """Control branch without inter-sensor graph propagation.

    Each sensor is encoded independently, then pooled by learned node scores.
    It preserves the spatial branch interface while removing A-GCN coupling.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.node_proj = nn.Linear(1, d_model)
        self.node_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.node_norm = nn.LayerNorm(d_model)
        self.node_score = nn.Linear(d_model, 1)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_sensors:
            raise ValueError(
                f"Expected (B, T, {self.num_sensors}) input, got {tuple(x.shape)}."
            )
        h = self.node_proj(x.unsqueeze(-1))
        h = self.node_norm(h + self.node_mlp(h))
        node_logits = self.node_score(h).squeeze(-1)
        node_weights = torch.softmax(node_logits, dim=-1).unsqueeze(-1)
        spatial = (node_weights * h).sum(dim=2)
        return self.out_norm(spatial)


class SelectiveStateSpaceBranch(nn.Module):
    """Single-scale selective gated state-space branch in pure PyTorch.

    A depthwise temporal convolution generates a local multi-cycle context.
    Linear projections produce input-dependent step sizes and read/write
    selections. A diagonal negative state transition is then scanned across
    the 30 observed cycles.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        state_dim: int = 16,
        dropout: float = 0.1,
        max_delta: float = 5.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than 1.")
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.max_delta = float(max_delta)
        self.bidirectional = bool(bidirectional)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.delta_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # A = -exp(a_log) is strictly negative for stable selective decay.
        a_init = torch.log(torch.linspace(1.0, 8.0, self.state_dim))
        self.a_log = nn.Parameter(a_init.unsqueeze(0).repeat(d_model, 1))

        self.merge = (
            nn.Linear(2 * d_model, d_model) if self.bidirectional else None
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _direction(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.dwconv(value.transpose(1, 2)).transpose(1, 2)
        value = F.silu(value)
        delta = F.softplus(self.delta_proj(value)).clamp(max=self.max_delta)
        b_sel = self.b_proj(value)
        c_sel = self.c_proj(value)
        selected = self._selective_scan(value, delta, b_sel, c_sel)
        return selected * F.silu(gate)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b_sel: torch.Tensor,
        c_sel: torch.Tensor,
    ) -> torch.Tensor:
        """Run a diagonal selective state-space recurrence over time."""
        batch, time_steps, channels = u.shape
        state = u.new_zeros(batch, channels, self.state_dim)
        a = -torch.exp(self.a_log).to(dtype=u.dtype, device=u.device)  # (D, S)
        outputs: list[torch.Tensor] = []

        for t in range(time_steps):
            step = delta[:, t, :].unsqueeze(-1)  # (B, D, 1)
            decay = torch.exp(step * a.unsqueeze(0))  # (B, D, S)
            input_value = u[:, t, :].unsqueeze(-1)  # (B, D, 1)
            write = b_sel[:, t, :].unsqueeze(-1)  # (B, D, 1)
            read = c_sel[:, t, :].unsqueeze(-1)  # (B, D, 1)
            state = decay * state + write * input_value
            outputs.append((state * read).sum(dim=-1))  # (B, D)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        forward = self._direction(x)
        if self.bidirectional:
            reverse_input = torch.flip(x, dims=[1])
            reverse = torch.flip(self._direction(reverse_input), dims=[1])
            assert self.merge is not None
            selected = self.merge(torch.cat([forward, reverse], dim=-1))
        else:
            selected = forward
        return self.norm(x + self.dropout(self.out_proj(selected)))


class MultiScaleStateSpaceTemporalBlock(nn.Module):
    """Parallel multi-scale selective SSM branches with learned scale gates."""

    def __init__(
        self,
        d_model: int = 64,
        kernels: Sequence[int] = (3, 5, 7),
        state_dim: int = 16,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if len(kernels) == 0:
            raise ValueError("At least one temporal kernel is required.")
        self.branches = nn.ModuleList(
            [
                SelectiveStateSpaceBranch(
                    d_model=d_model,
                    kernel_size=int(kernel_size),
                    state_dim=state_dim,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
                for kernel_size in kernels
            ]
        )
        self.scale_gate = nn.Linear(d_model, len(self.branches))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        branch_outputs = torch.stack(
            [branch(x) for branch in self.branches], dim=1
        )  # (B, K, T, D)
        scale_weights = torch.softmax(
            self.scale_gate(x.mean(dim=1)), dim=-1
        )  # (B, K)
        fused = (branch_outputs * scale_weights[:, :, None, None]).sum(dim=1)
        fused = self.dropout(self.out_proj(fused))
        return self.norm(x + fused), scale_weights


class LSTMTemporalBranch(nn.Module):
    """LSTM control branch used to ablate the selective SSM temporal block."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.hidden_size = int(d_model if hidden_size is None else hidden_size)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_proj = (
            nn.Identity()
            if self.hidden_size == d_model
            else nn.Linear(self.hidden_size, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        output, _ = self.lstm(x)
        output = self.output_proj(output)
        return self.norm(x + self.dropout(output)), None


class CrossAttentionGatedFusion(nn.Module):
    """Cross-attend temporal states to spatial states, then gate the mixture."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.branch_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # temporal queries attend to spatial keys/values.
        cross, _ = self.cross_attn(
            query=temporal,
            key=spatial,
            value=spatial,
            need_weights=False,
        )
        cross = self.cross_norm(temporal + cross)
        joint = torch.cat([spatial, temporal, cross], dim=-1)

        branch_weights = torch.softmax(
            self.branch_gate(joint.mean(dim=1)), dim=-1
        )  # (B, 3)
        mixed = (
            branch_weights[:, 0, None, None] * spatial
            + branch_weights[:, 1, None, None] * temporal
            + branch_weights[:, 2, None, None] * cross
        )
        channel_gate = self.channel_gate(joint)
        fallback = 0.5 * (spatial + temporal)
        fused = channel_gate * mixed + (1.0 - channel_gate) * fallback
        return self.out_norm(fused), branch_weights, channel_gate


class BiLSTMAttentionTemporalBranch(nn.Module):
    """Bidirectional LSTM with temporal attention for the main P7 backbone."""

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        hidden_size: int = 40,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            d_model,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        width = 2 * hidden_size
        self.attention = nn.Linear(width, 1)
        self.proj = nn.Linear(width, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        h, _ = self.lstm(x)
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1, keepdim=True)
        h = h + pooled
        return self.norm(self.proj(h)), None


class STG_MambaNet(nn.Module):
    """Spatial-temporal gated RUL regressor.

    Parameters
    ----------
    num_sensors:
        Number of input sensor channels (14 for standard C-MAPSS preprocessing).
    window_size:
        Expected temporal length. The model validates the supplied length.
    d_model:
        Hidden feature width.
    num_gcn_layers:
        Number of adaptive graph-convolution layers.
    state_dim:
        Diagonal state size in each selective SSM branch.
    kernels:
        Temporal depthwise-convolution kernel sizes for the multi-scale SSM.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        window_size: int = 30,
        d_model: int = 64,
        graph_embed_dim: int = 16,
        num_gcn_layers: int = 2,
        state_dim: int = 16,
        kernels: Sequence[int] = (3, 5, 7),
        num_heads: int = 4,
        dropout: float = 0.1,
        spatial_mode: Literal["graph", "independent"] = "graph",
        temporal_mode: Literal["ssm", "lstm", "bilstm_attn"] = "ssm",
        lstm_num_layers: int = 2,
        lstm_hidden_size: Optional[int] = None,
        graph_mode: Literal["learnable", "identity", "random_fixed", "hollow"] = "learnable",
        identity_prior_alpha: float = 0.1,
        hollow_alpha: float = 0.35,
        monotonic_head: bool = False,
        use_delta_features: bool = False,
        condition_film: bool = False,
    ) -> None:
        super().__init__()
        if spatial_mode not in {"graph", "independent"}:
            raise ValueError("spatial_mode must be 'graph' or 'independent'.")
        if temporal_mode not in {"ssm", "lstm", "bilstm_attn"}:
            raise ValueError("temporal_mode must be 'ssm', 'lstm', or 'bilstm_attn'.")
        if graph_mode not in {"learnable", "identity", "random_fixed", "hollow"}:
            raise ValueError(
                "graph_mode must be 'learnable', 'identity', 'random_fixed', or 'hollow'."
            )
        self.num_sensors = int(num_sensors)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.spatial_mode = spatial_mode
        self.temporal_mode = temporal_mode
        self.graph_mode = graph_mode
        self.identity_prior_alpha = float(identity_prior_alpha)
        self.hollow_alpha = float(hollow_alpha)
        self.monotonic_head = bool(monotonic_head)
        self.use_delta_features = bool(use_delta_features)
        self.condition_film = bool(condition_film)
        self.lstm_hidden_size = (
            int(d_model) if lstm_hidden_size is None else int(lstm_hidden_size)
        )

        if spatial_mode == "graph":
            self.spatial_branch = AdaptiveSpatialBranch(
                num_sensors=num_sensors,
                d_model=d_model,
                graph_embed_dim=graph_embed_dim,
                num_layers=num_gcn_layers,
                dropout=dropout,
                graph_mode=graph_mode,
                identity_prior_alpha=identity_prior_alpha,
                hollow_alpha=hollow_alpha,
            )
        else:
            self.spatial_branch = IndependentSpatialBranch(
                num_sensors=num_sensors,
                d_model=d_model,
                dropout=dropout,
            )

        self.temporal_input = nn.Linear(
            num_sensors * (2 if use_delta_features else 1), d_model
        )
        if self.condition_film:
            self.film = nn.Linear(3, 2 * d_model)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if temporal_mode == "bilstm_attn":
            self.temporal_branch = BiLSTMAttentionTemporalBranch(
                d_model=d_model,
                num_layers=lstm_num_layers,
                dropout=dropout,
                hidden_size=self.lstm_hidden_size,
            )
        elif temporal_mode == "ssm":
            self.temporal_branch = MultiScaleStateSpaceTemporalBlock(
                d_model=d_model,
                kernels=kernels,
                state_dim=state_dim,
                dropout=dropout,
            )
        else:
            self.temporal_branch = LSTMTemporalBranch(
                d_model=d_model,
                num_layers=lstm_num_layers,
                dropout=dropout,
                hidden_size=self.lstm_hidden_size,
            )
        self.fusion = CrossAttentionGatedFusion(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        if self.monotonic_head:
            self.mono_rate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        self._reset_parameters()
        if self.monotonic_head:
            nn.init.constant_(self.mono_rate[-1].bias, -1.0)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        return_trajectory: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Predict RUL, optionally with a dense per-cycle trajectory.

        Parameters
        ----------
        x:
            Tensor with shape (B, window_size, num_sensors). A single
            unbatched sample (window_size, num_sensors) is also accepted.
        return_diagnostics:
            If True, return graph adjacency, graph alpha, scale weights, and
            fusion gates for mechanism inspection.
        return_trajectory:
            If True, return (prediction, trajectory). The trajectory has
            shape (B, window_size) and its last column is exactly the
            terminal prediction. It is used by PhysicsGuidedLoss for the
            monotonicity term. When return_diagnostics is also True, the
            trajectory is detached and stored in diagnostics["trajectory"].

        Backward compatibility
        ----------------------
        Calling model(x) still returns only (B, 1). Calling
        model(x, return_diagnostics=True) still returns (prediction,
        diagnostics) with the original diagnostic keys.
        """
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected a 2-D or 3-D input tensor, got {tuple(x.shape)}.")
        if x.shape[1] != self.window_size or x.shape[2] != self.num_sensors:
            raise ValueError(
                f"Expected input (B, {self.window_size}, {self.num_sensors}), "
                f"got {tuple(x.shape)}."
            )

        if self.spatial_mode == "graph":
            spatial, adjacency = self.spatial_branch(x)
        else:
            spatial = self.spatial_branch(x)
            adjacency = None

        temporal_source = x
        if self.use_delta_features:
            delta = torch.zeros_like(x)
            if x.shape[1] > 1:
                delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            temporal_source = torch.cat([x, delta], dim=-1)
        temporal_input = self.temporal_input(temporal_source)
        if self.condition_film and cond is not None:
            if cond.ndim == 2:
                cond = cond.unsqueeze(1)
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            temporal_input = temporal_input * (1.0 + gamma) + beta
        temporal, scale_weights = self.temporal_branch(temporal_input)
        fused, branch_weights, channel_gate = self.fusion(spatial, temporal)

        last_state = fused[:, -1, :]
        mean_state = fused.mean(dim=1)
        prediction = self.regression_head(
            torch.cat([last_state, mean_state], dim=-1)
        )  # (B, 1)

        trajectory: Optional[torch.Tensor] = None
        if self.monotonic_head:
            # Terminal-anchored structural monotonic head.  The unconstrained
            # terminal RUL prediction is reused, while non-negative future
            # increments reconstruct a non-increasing trajectory ending at it.
            rate = F.softplus(self.mono_rate(fused)).squeeze(-1)  # (B, T)
            suffix = torch.cumsum(rate.flip(1), dim=1).flip(1)
            earlier = prediction + suffix[:, 1:]
            trajectory = torch.cat([earlier, prediction], dim=1)
            trajectory = torch.clamp(trajectory, min=0.0, max=125.0)
            prediction = trajectory[:, -1:]
        elif return_trajectory:
            steps = torch.arange(
                1, fused.shape[1] + 1, device=fused.device, dtype=fused.dtype
            )
            running_mean = fused.cumsum(dim=1) / steps.view(1, -1, 1)
            prefix_input = torch.cat([fused, running_mean], dim=-1)
            trajectory = self.regression_head(prefix_input).squeeze(-1)
            trajectory = torch.cat(
                [trajectory[:, :-1], prediction], dim=1
            )

        if unbatched:
            prediction = prediction.squeeze(0)
            if trajectory is not None:
                trajectory = trajectory.squeeze(0)

        if return_diagnostics:
            diagnostics: Dict[str, torch.Tensor] = {
                "fusion_branch_weights": branch_weights.detach(),
                "fusion_channel_gate": channel_gate.detach(),
            }
            if adjacency is not None:
                diagnostics["adjacency"] = adjacency.detach()
            if scale_weights is not None:
                diagnostics["temporal_scale_weights"] = scale_weights.detach()
            if self.spatial_mode == "graph":
                alpha = self.spatial_branch.graph_alpha
                if alpha is not None:
                    diagnostics["graph_alpha"] = alpha.detach()
                if hasattr(self.spatial_branch, "get_cross_adjacency"):
                    diagnostics["cross_adjacency"] = (
                        self.spatial_branch.get_cross_adjacency().detach()
                    )
            if trajectory is not None:
                diagnostics["trajectory"] = trajectory.detach()
            return prediction, diagnostics

        if return_trajectory:
            assert trajectory is not None
            return prediction, trajectory
        return prediction

    def get_adjacency(self) -> torch.Tensor:
        """Return the current (N, N) spatial adjacency, if graph mode is on."""
        if self.spatial_mode != "graph":
            raise RuntimeError("Independent spatial branch has no adjacency.")
        return self.spatial_branch.get_adjacency()

    def get_adjacency_matrix(self, detach: bool = False) -> torch.Tensor:
        """Alias used by plotting code for the sensor coupling heat-map."""
        adjacency = self.get_adjacency()
        return adjacency.detach() if detach else adjacency

    def get_cross_adjacency(self) -> torch.Tensor:
        """Return the pure off-diagonal coupling matrix, if available."""
        if self.spatial_mode != "graph":
            raise RuntimeError("Independent spatial branch has no adjacency.")
        return self.spatial_branch.get_cross_adjacency()

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())




class LiteSpatialBranch(nn.Module):
    """A-GCN branch using raw sensor values and first-order delta features."""

    def __init__(
        self,
        num_sensors: int = 14,
        d_model: int = 64,
        graph_embed_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_graph: bool = True,
        use_delta: bool = True,
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.use_graph = bool(use_graph)
        self.use_delta = bool(use_delta)
        self.node_proj = nn.Linear(2 if self.use_delta else 1, d_model)
        if self.use_graph:
            self.E1 = nn.Parameter(torch.empty(num_sensors, graph_embed_dim))
            self.E2 = nn.Parameter(torch.empty(num_sensors, graph_embed_dim))
            self.layers = nn.ModuleList(
                [AdaptiveGraphConv(d_model, dropout=dropout) for _ in range(num_layers)]
            )
            nn.init.xavier_uniform_(self.E1)
            nn.init.xavier_uniform_(self.E2)
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            self.E1 = None
            self.E2 = None
            self.layers = nn.ModuleList()
        self.node_score = nn.Linear(d_model, 1)
        self.out_norm = nn.LayerNorm(d_model)

    def get_adjacency(self) -> torch.Tensor:
        logits = torch.relu(torch.matmul(self.E1, self.E2.transpose(0, 1)))
        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.shape[-1] != self.num_sensors:
            raise ValueError(
                f"Expected (B, T, {self.num_sensors}) input, got {tuple(x.shape)}."
            )
        if self.use_delta:
            delta = torch.zeros_like(x)
            if x.shape[1] > 1:
                delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            node_input = torch.stack([x, delta], dim=-1)
        else:
            node_input = x.unsqueeze(-1)
        h = self.node_proj(node_input)
        if self.use_graph:
            adjacency = self.get_adjacency()
            for layer in self.layers:
                h = layer(h, adjacency)
        else:
            adjacency = None
            h = h + self.node_mlp(h)
        node_logits = self.node_score(h).squeeze(-1)
        node_weights = torch.softmax(node_logits, dim=-1).unsqueeze(-1)
        spatial = (node_weights * h).sum(dim=2)
        return self.out_norm(spatial), adjacency


class LiteGatedFusion(nn.Module):
    """Channel-wise two-branch gate without cross-attention."""

    def __init__(self, d_model: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial: torch.Tensor,
        temporal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.gate(torch.cat([spatial, temporal], dim=-1))
        fused = alpha * spatial + (1.0 - alpha) * temporal
        return self.norm(self.dropout(fused)), alpha


class TemporalAttentionPooling(nn.Module):
    """Attention pooling over the observed window."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return pooled, weights


class Lite_STG_Mamba(nn.Module):
    """Debloated STG-Mamba with delta features and a normalized RUL output.

    The output is in [0, 1]. Multiply by target_scale (125 by default) to
    recover cycles. The model intentionally removes MultiheadAttention and
    double softmax/sigmoid branch gating.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        window_size: int = 30,
        d_model: int = 64,
        graph_embed_dim: int = 16,
        num_gcn_layers: int = 2,
        state_dim: int = 16,
        kernels: Sequence[int] = (3, 5, 7),
        dropout: float = 0.3,
        target_scale: float = 125.0,
        output_activation: Literal["linear", "sigmoid"] = "linear",
        bidirectional_ssm: bool = True,
        use_graph: bool = True,
        use_delta: bool = True,
        temporal_mode: Literal["ssm", "lstm"] = "ssm",
    ) -> None:
        super().__init__()
        if output_activation not in {"linear", "sigmoid"}:
            raise ValueError("output_activation must be 'linear' or 'sigmoid'.")
        self.num_sensors = int(num_sensors)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.target_scale = float(target_scale)
        self.output_activation = output_activation
        self.use_delta = bool(use_delta)
        self.temporal_mode = temporal_mode
        self.spatial_branch = LiteSpatialBranch(
            num_sensors=num_sensors,
            d_model=d_model,
            graph_embed_dim=graph_embed_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
            use_graph=use_graph,
            use_delta=use_delta,
        )
        self.temporal_input = nn.Linear(
            num_sensors * (2 if use_delta else 1), d_model
        )
        if temporal_mode == "ssm":
            self.temporal_branch = MultiScaleStateSpaceTemporalBlock(
                d_model=d_model,
                kernels=kernels,
                state_dim=state_dim,
                dropout=dropout,
                bidirectional=bidirectional_ssm,
            )
        elif temporal_mode == "lstm":
            self.temporal_branch = LSTMTemporalBranch(
                d_model=d_model,
                num_layers=2,
                dropout=dropout,
            )
        else:
            raise ValueError("temporal_mode must be 'ssm' or 'lstm'.")
        self.fusion = LiteGatedFusion(d_model=d_model, dropout=dropout)
        self.pooling = TemporalAttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected a 2-D or 3-D tensor, got {tuple(x.shape)}.")
        if x.shape[1] != self.window_size or x.shape[2] != self.num_sensors:
            raise ValueError(
                f"Expected (B, {self.window_size}, {self.num_sensors}), got {tuple(x.shape)}."
            )

        if self.use_delta:
            delta = torch.zeros_like(x)
            if x.shape[1] > 1:
                delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            temporal_features = torch.cat([x, delta], dim=-1)
        else:
            temporal_features = x
        temporal_input = self.temporal_input(temporal_features)

        spatial, adjacency = self.spatial_branch(x)
        temporal, _ = self.temporal_branch(temporal_input)
        fused, channel_gate = self.fusion(spatial, temporal)
        pooled, temporal_weights = self.pooling(fused)
        last_state = fused[:, -1, :]
        raw = self.head(torch.cat([last_state, pooled], dim=-1))
        prediction = torch.sigmoid(raw) if self.output_activation == "sigmoid" else raw
        if unbatched:
            prediction = prediction.squeeze(0)

        if not return_diagnostics:
            return prediction
        diagnostics: dict[str, torch.Tensor] = {
            "channel_gate": channel_gate.detach(),
            "temporal_pooling_weights": temporal_weights.detach(),
        }
        if adjacency is not None:
            diagnostics["adjacency"] = adjacency.detach()
        return prediction, diagnostics

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

__all__ = [
    "AdaptiveGraphConv",
    "AdaptiveSpatialBranch",
    "IndependentSpatialBranch",
    "SelectiveStateSpaceBranch",
    "LSTMTemporalBranch",
    "MultiScaleStateSpaceTemporalBlock",
    "CrossAttentionGatedFusion",
    "STG_MambaNet",
    "LiteSpatialBranch",
    "LiteGatedFusion",
    "TemporalAttentionPooling",
    "Lite_STG_Mamba",
]


if __name__ == "__main__":
    try:
        from .loss import PhysicsGuidedLoss
    except ImportError:
        from loss import PhysicsGuidedLoss

    torch.manual_seed(42)
    batch_size = 8
    window_size = 30
    num_sensors = 14

    model = STG_MambaNet(
        num_sensors=num_sensors,
        window_size=window_size,
        d_model=64,
    )
    model.train()
    inputs = torch.randn(batch_size, window_size, num_sensors)
    targets = torch.rand(batch_size, 1) * 125.0

    predictions, diagnostics = model(inputs, return_diagnostics=True)
    criterion = PhysicsGuidedLoss()
    loss, components = criterion(predictions, targets, return_components=True)
    loss.backward()

    adjacency = diagnostics["adjacency"]
    branch_weights = diagnostics["fusion_branch_weights"]
    assert predictions.shape == (batch_size, 1)
    assert adjacency.shape == (num_sensors, num_sensors)
    assert torch.allclose(adjacency.sum(dim=-1), torch.ones(num_sensors), atol=1e-6)
    assert torch.allclose(
        branch_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-6
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    print("STG_MambaNet smoke test passed")
    print(f"input:               {tuple(inputs.shape)}")
    print(f"prediction:          {tuple(predictions.shape)}")
    print(f"adaptive adjacency:  {tuple(adjacency.shape)}")
    print(f"fusion weights:      {tuple(branch_weights.shape)}")
    print(f"parameters:          {model.num_parameters:,}")
    print(f"total loss:          {loss.item():.6f}")
    print(f"loss components:     {components}")
