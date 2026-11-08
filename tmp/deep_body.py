"""Deep body: multi-scale temporal convolution + extra transformer layer +
attention pooling on top of the channel encoder. Same interface as CandidateModel
(encode + reg_head) so PriorWrapper and the frozen training recipe work unchanged."""
import torch, torch.nn as nn

class DeepBody(nn.Module):
    def __init__(self, base, d_model, dropout=0.15, heads=4, extra_layers=1):
        super().__init__()
        self.base = base
        self.d_model = int(d_model)
        self.num_sensors = base.num_sensors
        self.ms = nn.ModuleList([nn.Conv1d(d_model, d_model, k, padding=k // 2) for k in (3, 5, 7)])
        self.fuse = nn.Linear(3 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.extra = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(int(extra_layers))])
        self.pool_score = nn.Linear(d_model, 1)
        self.reg_head = base.reg_head

    def encode(self, x, cond=None, extras=None):
        h = self.base.encode(x, cond=cond, extras=extras)      # (B, L, D)
        z = h.transpose(1, 2)                                   # (B, D, L)
        z = torch.cat([m(z) for m in self.ms], dim=1)           # (B, 3D, L)
        h = self.norm1(h + self.drop(self.fuse(z.transpose(1, 2))))
        for layer in self.extra:
            h = layer(h)
        w = torch.softmax(self.pool_score(h).squeeze(-1), dim=1)  # (B, L)
        return (w.unsqueeze(-1) * h).sum(dim=1, keepdim=True)     # (B, 1, D)
