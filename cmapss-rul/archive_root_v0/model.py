"""STG-MambaNet: a physics-guided spatial-temporal graph model for C-MAPSS RUL.

The implementation is deliberately restricted to native torch.nn operators.
No mamba_ssm, causal_conv1d, torch_geometric, or custom CUDA extension is
required.

Architecture
------------
1. AdaptiveSpatialGCN learns a directed, row-stochastic sensor graph without
   any physical prior:

       A = softmax(ReLU(E1 @ E2.T), dim=-1),   E1, E2 in R^(N x d_e)

   Each graph layer performs the residual update

       H <- LayerNorm(H + sigma(A H W_g))

   where H has shape (B, L, N, D). The learned A is exposed by
   AdaptiveSpatialGCN.get_adjacency_matrix() for the paper's sensor coupling
   heat-map.

2. TemporalSSMBlock is a pure-PyTorch, Mamba-style dual-path block. The main
   path uses a depthwise Conv1d, two parallel dilated depthwise Conv1d
   branches (receptive fields 3 and 5), and a selective diagonal state-space
   recurrence. The gate path is SiLU(Linear(x)). The two paths are combined by
   a Hadamard product, then a residual connection and LayerNorm.

3. Gated fusion plus temporal attention pooling combines the spatial and
   temporal views with a sample/channel-adaptive gate. A learned query pools
   the fused sequence with attention. The pooled vector produces a residual
   correction to the terminal RUL, while a shared per-cycle head produces the
   dense RUL trajectory used by the monotonic physics prior.

The public model returns (B, 1) by default. Calling
model(x, return_trajectory=True) additionally returns the per-cycle RUL
trajectory (B, L) for loss.PhysicsGuidedRULLoss.

Ablation switches
-----------------
* use_gcn=False: the sensor graph is replaced by a plain dense
  nn.Linear(num_sensors, num_sensors) sensor mixer.
* use_ssm=False: each temporal block is replaced by one nn.LSTM layer.

Both switches keep the same tensor contract, so the four ablation variants can
share one training/evaluation script.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial branch
# ---------------------------------------------------------------------------


class AdaptiveGraphConv(nn.Module):
    """One residual adaptive graph-convolution layer.

    The layer implements the paper's update

        H_out = LayerNorm(H + sigma(A H W_g))

    with sigma = SiLU. The residual term is important: stacking several graph
    layers otherwise drives all sensor representations toward the same vector
    (over-smoothing).
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        # x: (B, L, N, D), adjacency: (N, N)
        aggregated = torch.einsum("nm,blmd->blnd", adjacency, x)
        message = self.proj(aggregated)
        return self.norm(x + self.dropout(F.silu(message)))


class AdaptiveSpatialGCN(nn.Module):
    """Learn the sensor graph and aggregate information over its nodes.

    Parameters
    ----------
    num_sensors:
        Number of sensor nodes N (14 for standard C-MAPSS).
    d_model:
        Per-node hidden width D.
    graph_embed_dim:
        Dimension d_e of the two learnable node-embedding matrices.
    num_layers:
        Number of residual graph-convolution layers.
    use_graph:
        If False, the graph branch degrades to a dense sensor-mixing linear
        layer. This is the w/o A-GCN ablation.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        d_model: int = 64,
        graph_embed_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        if num_sensors <= 0:
            raise ValueError("num_sensors must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if graph_embed_dim <= 0:
            raise ValueError("graph_embed_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.num_sensors = int(num_sensors)
        self.d_model = int(d_model)
        self.graph_embed_dim = int(graph_embed_dim)
        self.num_layers = int(num_layers)
        self.use_graph = bool(use_graph)

        # A scalar sensor reading is lifted to a D-dimensional node state.
        self.node_proj = nn.Linear(1, d_model)

        if self.use_graph:
            # E1 and E2 are the two learnable factors of the soft adjacency.
            self.E1 = nn.Parameter(torch.empty(num_sensors, graph_embed_dim))
            self.E2 = nn.Parameter(torch.empty(num_sensors, graph_embed_dim))
            self.graph_layers = nn.ModuleList(
                [
                    AdaptiveGraphConv(d_model=d_model, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )
            # A moderate scale keeps the initial graph close to uniform while
            # leaving enough positive ReLU activations for E1/E2 to receive
            # non-zero gradients from the first optimisation step.
            nn.init.normal_(self.E1, mean=0.0, std=0.10)
            nn.init.normal_(self.E2, mean=0.0, std=0.10)
        else:
            self.dense_mix = nn.Linear(num_sensors, num_sensors, bias=False)
            self.dense_proj = nn.Linear(d_model, d_model)
            self.dense_norm = nn.LayerNorm(d_model)

        self.out_norm = nn.LayerNorm(d_model)

    # -- graph construction -------------------------------------------------
    def _adjacency(self) -> Tensor:
        """Return A = softmax(ReLU(E1 E2^T), dim=-1)."""
        if not self.use_graph:
            raise RuntimeError(
                "This AdaptiveSpatialGCN was built with use_graph=False; "
                "no learned adjacency matrix exists."
            )
        logits = F.relu(torch.matmul(self.E1, self.E2.transpose(0, 1)))
        return F.softmax(logits, dim=-1)

    def get_adjacency_matrix(self, detach: bool = False) -> Tensor:
        """Export the current (N, N) adaptive sensor graph.

        Parameters
        ----------
        detach:
            If True, return a tensor detached from the autograd graph. This
            is the normal choice for plotting Fig. 3 after training.
        """
        adjacency = self._adjacency()
        return adjacency.detach() if detach else adjacency

    def get_adj_matrix(self, detach: bool = False) -> Tensor:
        """Alias kept for compatibility with the paper's notation."""
        return self.get_adjacency_matrix(detach=detach)

    # -- forward ------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Map (B, L, N) sensor readings to (B, L, N, D) states."""
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D (B, L, N) tensor, got {tuple(x.shape)}.")
        if x.shape[-1] != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} sensors, got {x.shape[-1]}."
            )

        # Lift each scalar reading independently: (B, L, N) -> (B, L, N, D).
        h = self.node_proj(x.unsqueeze(-1))

        if self.use_graph:
            adjacency = self._adjacency()
            for layer in self.graph_layers:
                h = layer(h, adjacency)
        else:
            # Dense sensor mixing is a deliberately weak spatial prior: it
            # can learn a fixed linear relation but cannot adapt its edges.
            mixed = self.dense_mix(x)  # (B, L, N)
            mixed_state = self.node_proj(mixed.unsqueeze(-1))
            h = h + self.dense_proj(mixed_state)
            h = self.dense_norm(h)

        return self.out_norm(h)


class SensorAttentionPool(nn.Module):
    """Adaptively collapse N sensor states into one temporal token.

    The attention weights answer the question "which sensor is informative
    for this engine at this cycle?" and are learned jointly with the graph.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, node_states: Tensor) -> Tensor:
        # node_states: (B, L, N, D)
        logits = self.score(node_states).squeeze(-1)  # (B, L, N)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (weights * node_states).sum(dim=2)  # (B, L, D)
        return self.norm(self.out_proj(pooled))


# ---------------------------------------------------------------------------
# Temporal branch
# ---------------------------------------------------------------------------


class SelectiveStateSpace(nn.Module):
    """Selective diagonal state-space recurrence in pure PyTorch.

    For each channel d and each of S diagonal states, the recurrence is

        h_t = exp(dt_t * A_d) * h_{t-1} + dt_t * B_t * u_t
        y_t = sum_s C_{t,s} * h_{t,s} + D_d * u_t

    A_d = -exp(A_log_d) < 0 guarantees stable decay. dt_t, B_t and C_t are
    produced by linear projections of the current input, which is the
    selective mechanism that distinguishes this block from a fixed SSM. The
    loop is over L (30 for C-MAPSS), so its cost is linear in sequence length
    and negligible compared with a Transformer's O(L^2) attention.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_max: float = 5.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive.")
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.dt_max = float(dt_max)

        self.dt_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(d_model))
        self.b_proj = nn.Linear(d_model, d_state, bias=False)
        self.c_proj = nn.Linear(d_model, d_state, bias=False)

        # A = -exp(A_log), with initial eigenvalues spread from -1 to -S.
        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(a_init.unsqueeze(0).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u: Tensor) -> Tensor:
        """Scan (B, L, D) inputs and return (B, L, D) outputs."""
        if u.ndim != 3 or u.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected (B, L, {self.d_model}) input, got {tuple(u.shape)}."
            )

        batch, steps, _ = u.shape
        dt = F.softplus(self.dt_proj(u) + self.dt_bias).clamp(max=self.dt_max)
        b_sel = self.b_proj(u)  # (B, L, S)
        c_sel = self.c_proj(u)  # (B, L, S)

        # A is (D, S). Keeping it out of the loop avoids re-exp() every step.
        a = -torch.exp(self.A_log)  # (D, S)
        state = u.new_zeros(batch, self.d_model, self.d_state)
        outputs = []

        for t in range(steps):
            step = dt[:, t, :].unsqueeze(-1)  # (B, D, 1)
            decay = torch.exp(step * a.unsqueeze(0))  # (B, D, S)
            write = b_sel[:, t, :].unsqueeze(1)  # (B, 1, S)
            value = u[:, t, :].unsqueeze(-1)  # (B, D, 1)
            state = decay * state + (write * value) * step
            read = c_sel[:, t, :].unsqueeze(1)  # (B, 1, S)
            y_t = (state * read).sum(dim=-1) + self.D * u[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class TemporalSSMBlock(nn.Module):
    """Multi-scale gated state-space block (the proposed temporal encoder).

    Data flow:

        main, gate = Linear(x).chunk(2)
        main = SiLU(DWConv_k3(main))
        multi = SiLU(DWConv_k3_d1(main) + DWConv_k3_d2(main))
        selected = SelectiveStateSpace(main)
        out = LayerNorm(x + Dropout(Linear(multi + selected) * SiLU(gate)))

    The two depthwise branches have effective receptive fields 3 and 5. They
    capture short mechanical transients and slower wear trends simultaneously.
    The Hadamard gate plays the role of Mamba's input-dependent information
    selection without requiring compiled kernels.
    """

    def __init__(
        self,
        d_model: int = 64,
        expand: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or expand <= 0:
            raise ValueError("d_model and expand must be positive.")
        inner = int(d_model * expand)
        self.d_model = int(d_model)
        self.inner = inner

        self.in_proj = nn.Linear(d_model, 2 * inner)
        self.dwconv = nn.Conv1d(
            inner, inner, kernel_size=3, padding=1, groups=inner
        )
        self.branch_short = nn.Conv1d(
            inner, inner, kernel_size=3, padding=1, groups=inner
        )
        self.branch_long = nn.Conv1d(
            inner,
            inner,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=inner,
        )
        self.ssm = SelectiveStateSpace(inner, d_state=d_state)
        self.out_proj = nn.Linear(inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected (B, L, {self.d_model}) input, got {tuple(x.shape)}."
            )

        main, gate = self.in_proj(x).chunk(2, dim=-1)  # (B, L, inner) each
        main = self.dwconv(main.transpose(1, 2)).transpose(1, 2)
        main = F.silu(main)

        short = self.branch_short(main.transpose(1, 2))
        long = self.branch_long(main.transpose(1, 2))
        multi_scale = F.silu(short + long).transpose(1, 2)

        selected = self.ssm(main)
        # The gate lives in the expanded inner space, so apply it before the
        # projection back to d_model.
        fused = self.out_proj((multi_scale + selected) * F.silu(gate))
        return self.norm(x + self.dropout(fused))


class TemporalLSTMBlock(nn.Module):
    """Single-layer LSTM fallback used by the w/o SSM ablation."""

    def __init__(self, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return self.norm(x + self.dropout(out))


# ---------------------------------------------------------------------------
# Fusion and prediction head
# ---------------------------------------------------------------------------


class GatedSpatiotemporalFusion(nn.Module):
    """Sample- and channel-adaptive gate between spatial and temporal views."""

    def __init__(self, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial: Tensor,
        temporal: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate = self.gate(torch.cat([spatial, temporal], dim=-1))
        fused = gate * temporal + (1.0 - gate) * spatial
        fused = self.out_proj(fused)
        return self.norm(fused + self.dropout(fused)), gate


class TemporalAttentionPool(nn.Module):
    """Learned-query attention pooling over the cycle axis."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        # h: (B, L, D); mask: (B, L) boolean, True for valid positions.
        scores = (self.proj(h) * self.query).sum(dim=-1) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        return pooled, weights


class STG_MambaNet(nn.Module):
    """Spatio-temporal graph Mamba network for RUL regression.

    Parameters
    ----------
    num_sensors:
        Number of sensor channels (14 for the standard C-MAPSS protocol).
    seq_len:
        Expected window length. With strict_seq_len=True a mismatch is a hard
        error, which catches accidental windowing bugs early.
    d_model:
        Hidden width used by both branches.
    use_gcn:
        Enable the adaptive graph branch. If False, it is replaced by a dense
        linear sensor mixer (the w/o A-GCN ablation).
    use_ssm:
        Enable the multi-scale selective SSM. If False, each temporal block
        becomes one LSTM layer (the w/o SSM-Block ablation).
    graph_embed_dim:
        Dimension of E1 and E2.
    num_gcn_layers:
        Number of residual graph convolutions.
    num_temporal_layers:
        Number of stacked temporal blocks.
    d_state:
        Number of diagonal states per SSM channel.
    expand:
        Expansion ratio of the SSM block's inner width.
    dropout:
        Dropout probability in both branches and the regression head.
    strict_seq_len:
        If True (default), reject inputs whose length differs from seq_len.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        seq_len: int = 30,
        d_model: int = 64,
        use_gcn: bool = True,
        use_ssm: bool = True,
        *,
        graph_embed_dim: int = 16,
        num_gcn_layers: int = 2,
        num_temporal_layers: int = 2,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        strict_seq_len: bool = True,
    ) -> None:
        super().__init__()
        if num_sensors <= 0 or seq_len <= 0 or d_model <= 0:
            raise ValueError("num_sensors, seq_len and d_model must be positive.")
        if num_temporal_layers <= 0:
            raise ValueError("num_temporal_layers must be positive.")

        self.num_sensors = int(num_sensors)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.use_gcn = bool(use_gcn)
        self.use_ssm = bool(use_ssm)
        self.strict_seq_len = bool(strict_seq_len)

        # Spatial view: learned graph -> per-node states -> sensor attention.
        self.spatial_gcn = AdaptiveSpatialGCN(
            num_sensors=num_sensors,
            d_model=d_model,
            graph_embed_dim=graph_embed_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
            use_graph=use_gcn,
        )
        self.sensor_pool = SensorAttentionPool(d_model=d_model)

        # Temporal view: raw sensor sequence projected to d_model, then a
        # stack of selective SSM blocks (or LSTM blocks in the ablation).
        self.temporal_stem = nn.Linear(num_sensors, d_model)
        if use_ssm:
            temporal_blocks = [
                TemporalSSMBlock(
                    d_model=d_model,
                    expand=expand,
                    d_state=d_state,
                    dropout=dropout,
                )
                for _ in range(num_temporal_layers)
            ]
        else:
            temporal_blocks = [
                TemporalLSTMBlock(d_model=d_model, dropout=dropout)
                for _ in range(num_temporal_layers)
            ]
        self.temporal_blocks = nn.ModuleList(temporal_blocks)

        self.fusion = GatedSpatiotemporalFusion(d_model=d_model, dropout=dropout)
        self.temporal_pool = TemporalAttentionPool(d_model=d_model)

        # Shared per-cycle head. Its terminal output is anchored by the
        # scalar MSE/asymmetric loss; the monotonic prior shapes the rest.
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Attention-pooled residual correction for the terminal prediction.
        # Starting at zero keeps the initial prediction exactly equal to the
        # terminal trajectory value, which stabilises early training.
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # The residual correction must not move the initial prediction away
        # from the trajectory head's terminal value.
        last_context = self.context_head[-1]
        assert isinstance(last_context, nn.Linear)
        nn.init.zeros_(last_context.weight)
        nn.init.zeros_(last_context.bias)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def get_adjacency_matrix(self, detach: bool = False) -> Tensor:
        """Export the learned (N, N) sensor graph.

        Raises
        ------
        RuntimeError
            If the model was built with use_gcn=False.
        """
        return self.spatial_gcn.get_adjacency_matrix(detach=detach)

    def get_adj_matrix(self, detach: bool = False) -> Tensor:
        """Alias for get_adjacency_matrix()."""
        return self.get_adjacency_matrix(detach=detach)

    def forward(
        self,
        x: Tensor,
        return_trajectory: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Predict RUL.

        Parameters
        ----------
        x:
            (B, L, N) sensor windows. An unbatched (L, N) tensor is also
            accepted.
        return_trajectory:
            If True, return (prediction, trajectory) where prediction has
            shape (B, 1) and trajectory has shape (B, L). The trajectory is
            required to activate the monotonic term in
            loss.PhysicsGuidedRULLoss.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            (B, 1) by default; ((B, 1), (B, L)) when requested.
        """
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 2-D or 3-D sensor tensor, got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} sensors, got {x.shape[-1]}."
            )
        if self.strict_seq_len and x.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected sequence length {self.seq_len}, got {x.shape[1]}."
            )

        # Two views are computed in parallel from the same raw window.
        spatial_nodes = self.spatial_gcn(x)  # (B, L, N, D)
        spatial = self.sensor_pool(spatial_nodes)  # (B, L, D)

        temporal = self.temporal_stem(x)  # (B, L, D)
        for block in self.temporal_blocks:
            temporal = block(temporal)

        fused, _ = self.fusion(spatial, temporal)  # (B, L, D)
        pooled, _ = self.temporal_pool(fused)  # (B, D)

        # Dense per-cycle RUL estimates and the attention-pooled correction.
        trajectory = self.trajectory_head(fused)  # (B, L, 1)
        correction = self.context_head(pooled)  # (B, 1)
        prediction = trajectory[:, -1, :] + correction  # (B, 1)

        trajectory = trajectory.squeeze(-1)  # (B, L)
        if unbatched:
            prediction = prediction.squeeze(0)
            trajectory = trajectory.squeeze(0)
        if return_trajectory:
            return prediction, trajectory
        return prediction

    def predict_trajectory(self, x: Tensor) -> Tensor:
        """Return only the dense (B, L) RUL trajectory."""
        _, trajectory = self.forward(x, return_trajectory=True)
        return trajectory

    def extra_repr(self) -> str:
        return (
            f"num_sensors={self.num_sensors}, seq_len={self.seq_len}, "
            f"d_model={self.d_model}, use_gcn={self.use_gcn}, "
            f"use_ssm={self.use_ssm}"
        )


__all__ = [
    "AdaptiveGraphConv",
    "AdaptiveSpatialGCN",
    "SensorAttentionPool",
    "SelectiveStateSpace",
    "TemporalSSMBlock",
    "TemporalLSTMBlock",
    "GatedSpatiotemporalFusion",
    "TemporalAttentionPool",
    "STG_MambaNet",
]
