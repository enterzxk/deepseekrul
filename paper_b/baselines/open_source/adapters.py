"""Thin input/output adapters around byte-exact upstream model classes."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch import nn
from .common import ROOT, seed_everything


def module(key, relative, name):
    root = ROOT / "upstream" / key
    sys.path.insert(0, str(root))
    sys.path.insert(0, str((root / relative).parent))
    spec = importlib.util.spec_from_file_location(name, root / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class UpstreamAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        key, subset, length = cfg["model"], cfg["subset"], cfg["window"]
        if key == "aidgn":
            mod = module(key, "Model/AIDGN.py", "upstream_aidgn")
            seed_everything(cfg["seed"])
            args = SimpleNamespace(feature_num=14, sequence_len=length, hidden_dim=64,
                                   fc_layer_dim=256, fc_dropout=0.1, nlayer=1, AATE_dim=10,
                                   hop=1, outlayer="Linear", feature_fc_layer_dim=56)
            self.net = mod.AIDGN(args)
            self.output_scale = 125.0
        elif key == "two_phase":
            self.pre_module = module(key, "Transformer_pre_model.py", "upstream_two_phase_pre")
            mod = module(key, "Transformer_ft_serialatten_model.py", "upstream_two_phase_ft")
            seed_everything(cfg["seed"])
            layers = 1 if subset in ("FD001", "FD003") else 2
            dropout = 0.1 if layers == 1 else 0.0
            self.net = mod.Transformer_ft(17, 16, length, 32, 32, layers, 2, dropout)
            self.output_scale = 1.0
        elif key == "dual_mixer":
            self.contrastive = module(key, "models/RULPrediction/ContrastiveModules.py", "ContrastiveModules")
            # The author's import is absolute; retain its exact module object.
            sys.modules["ContrastiveModules"] = self.contrastive
            mod = module(key, "models/RULPrediction/MLPMixer.py", "upstream_dual_mixer")
            seed_everything(cfg["seed"])
            self.net = mod.DualMLPMixer(window_size=length, in_features=14, hidden_dim=32,
                                       num_layers=6, dropout=0, device="cuda:0", label_norm=True)
            self.output_scale = 125.0
        elif key == "ffve":
            self.variational = module(key, "models/model_FFVE.py", "upstream_ffve")
            seed_everything(cfg["seed"])
            args = SimpleNamespace(e_layers=cfg["ffve_blocks"], norm=False, fac_C=True,
                                   short_res=True, long_res=cfg["ffve_long_residual"])
            self.encoder = self.variational.Encoder(args, sequence_length=length, input_dim=14, latent_dim=3)
            self.decoder = self.variational.Decoder(latent_dim=3)
            self.output_scale = 1.0
        elif key == "fcstgnn":
            mod = module(key, "FC_STGNN/Model.py", "upstream_fcstgnn")
            seed_everything(cfg["seed"])
            # Tuple order: patch, conv_out, lstmout, hidden, time_length, windows.
            p, conv, lstmout, hidden, time_length, windows = {
                "FD001": (5, 7, 32, 8, 6, 8), "FD002": (5, 7, 12, 8, 10, 14),
                "FD003": (2, 4, 6, 24, 25, 36), "FD004": (5, 7, 6, 8, 10, 14),
            }[subset]
            self.patch, self.time_length = p, time_length
            self.net = mod.FC_STGNN_RUL(p, conv, 8, lstmout, 2, hidden, time_length, 14,
                                       windows, [2, 2], [1, 2], 0.7, "mean", 1)
            self.output_scale = 125.0
        else:
            raise ValueError(key)

    def forward(self, x, age=None):
        key = self.cfg["model"]
        if key == "aidgn":
            return self.net(x, age)[1].reshape(-1) * self.output_scale
        if key == "ffve":
            mu, logvar = self.encoder(x)
            z = self.variational.reparameterize(mu, logvar)
            return self.decoder(z).reshape(-1)
        if key == "fcstgnn":
            x = x.reshape(len(x), self.time_length, self.patch, 14).transpose(2, 3).contiguous()
        return self.net(x).reshape(-1) * self.output_scale

    def loss(self, x, y, age=None):
        key = self.cfg["model"]
        if key == "ffve":
            mu, logvar = self.encoder(x)
            out = self.decoder(self.variational.reparameterize(mu, logvar)).reshape(-1)
            return self.variational.total_loss(out, y.reshape(-1), mu, logvar)[0]
        if key == "dual_mixer":
            # x is [batch, sampled-negatives, time, sensors]. Retain the exact
            # upstream sampler/augmentation/weighted InfoNCE and regression.
            criterion = self.contrastive.MSEContrastiveLoss("InfoNCE")
            return self.net.compute_loss(x, y, criterion)[0]
        prediction = self.forward(x, age) / self.output_scale
        loss = torch.mean((prediction - y.reshape(-1) / self.output_scale) ** 2)
        return torch.sqrt(loss) if key == "two_phase" else loss
