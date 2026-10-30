"""Standard C-MAPSS baseline regressors in native PyTorch.

All models consume (B, T, 14), append first-order delta features internally,
and return a normalized RUL in an unconstrained linear output. Training code
multiplies by 125 to recover cycles.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def append_delta(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("Expected (B, T, C) input.")
    delta = torch.zeros_like(x)
    if x.shape[1] > 1:
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return torch.cat([x, delta], dim=-1)


class CNN1D(nn.Module):
    """Three-layer 1D convolutional RUL regressor."""

    def __init__(self, num_sensors: int = 14, d_model: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2 * num_sensors, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = append_delta(x).transpose(1, 2)
        h = self.net(h)
        last = h[:, :, -1]
        mean = h.mean(dim=-1)
        return self.head(torch.cat([last, mean], dim=-1))


class BiLSTM(nn.Module):
    """Two-layer bidirectional LSTM baseline."""

    def __init__(self, num_sensors: int = 14, d_model: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2 * num_sensors, d_model)
        self.lstm = nn.LSTM(
            d_model,
            d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.input_proj(append_delta(x)))
        last = h[:, -1, :]
        mean = h.mean(dim=1)
        return self.head(torch.cat([last, mean], dim=-1))


class GRUAttention(nn.Module):
    """Two-layer GRU with temporal attention pooling."""

    def __init__(self, num_sensors: int = 14, d_model: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2 * num_sensors, d_model)
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.input_proj(append_delta(x)))
        weights = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        return self.head(torch.cat([h[:, -1, :], pooled], dim=-1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class VanillaTransformer(nn.Module):
    """Two-layer Transformer encoder with attention pooling."""

    def __init__(self, num_sensors: int = 14, d_model: int = 64, nhead: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2 * num_sensors, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.position(self.input_proj(append_delta(x))))
        weights = torch.softmax(self.score(h).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        return self.head(torch.cat([h[:, -1, :], pooled], dim=-1))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, 0))
        return conv(padded)[:, :, : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self._causal_conv(self.conv1, x)
        out = self.activation(self.norm1(out))
        out = self.dropout(out)
        out = self._causal_conv(self.conv2, out)
        out = self.activation(self.norm2(out))
        out = self.dropout(out)
        return self.activation(out + residual)


class TCN(nn.Module):
    """Temporal convolutional network baseline with dilations 1, 2, 4."""

    def __init__(self, num_sensors: int = 14, d_model: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(2 * num_sensors, d_model, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalBlock(d_model, 3, 1, dropout),
            TemporalBlock(d_model, 3, 2, dropout),
            TemporalBlock(d_model, 3, 4, dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.input_proj(append_delta(x).transpose(1, 2)))
        return self.head(torch.cat([h[:, :, -1], h.mean(dim=-1)], dim=-1))


BASELINE_REGISTRY = {
    "cnn1d": CNN1D,
    "bilstm": BiLSTM,
    "gru_attention": GRUAttention,
    "transformer": VanillaTransformer,
    "tcn": TCN,
}


def build_baseline(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {name!r}; choose {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[key](**kwargs)


__all__ = [
    "CNN1D",
    "BiLSTM",
    "GRUAttention",
    "VanillaTransformer",
    "TCN",
    "BASELINE_REGISTRY",
    "build_baseline",
]
