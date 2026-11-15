"""Training-time sensor mask and residual-scale noise (plan sections 4-5).

Both augmentors use their own torch.Generator so that the model
initialisation, the batch permutation and the model-internal dropout streams
are not displaced by extra sampling. With mask_prob=0 and no noise scale the
object is never built and no random number is consumed (T1).

Pairing across groups (plan section 5.7): the mask uniform stream is seeded by
the model seed only, so p=0.10 and p=0.20 draw the same uniforms and the
p=0.20 mask is a superset of the p=0.10 mask. The noise normal stream is also
seeded by the model seed only, so alpha=0.25 and alpha=0.50 share epsilon.
"""
from __future__ import annotations

import torch

MASK_SEED_BASE = 700000
NOISE_SEED_BASE = 800000


class SensorAugmentor:
    def __init__(self, mask_prob=0.0, noise_scale=None, n_sensors=14,
                 seed=101, device='cuda:0'):
        self.mask_prob = float(mask_prob)
        self.n_sensors = int(n_sensors)
        self.device = torch.device(device)
        self.mask_gen = None
        self.noise_gen = None
        self.noise_scale = None
        if self.mask_prob > 0:
            self.mask_gen = torch.Generator(device=self.device)
            self.mask_gen.manual_seed(MASK_SEED_BASE + int(seed))
        if noise_scale is not None:
            self.noise_scale = torch.as_tensor(noise_scale, dtype=torch.float32,
                                               device=self.device)
            self.noise_gen = torch.Generator(device=self.device)
            self.noise_gen.manual_seed(NOISE_SEED_BASE + int(seed))
        self.noise_scale_rms = (float(self.noise_scale.double().pow(2).mean().sqrt())
                                if self.noise_scale is not None else 0.0)
        self.reset_stats()

    # ------------------------------------------------------------------ stats
    def reset_stats(self):
        self.hidden = 0
        self.slots = 0
        self.resampled = 0
        self.noise_sq = 0.0
        self.noise_n = 0

    def stats(self):
        return dict(
            mask_prob=self.mask_prob,
            nominal_hide_rate=self.mask_prob,
            actual_hide_rate=(self.hidden / self.slots) if self.slots else 0.0,
            mask_slots=int(self.slots),
            resampled_samples=int(self.resampled),
            noise_scale_rms=float(getattr(self, 'noise_scale_rms', 0.0)),
            injected_noise_rms=float((self.noise_sq / self.noise_n) ** 0.5) if self.noise_n else 0.0,
            injected_noise_values=int(self.noise_n),
        )

    # ------------------------------------------------------------------- mask
    def sample_mask(self, batch, device=None):
        """(B, N) bool, True = available. Shared over time by construction."""
        if self.mask_gen is None:
            raise RuntimeError('sample_mask called without mask_prob > 0')
        device = device or self.device
        u = torch.rand(batch, self.n_sensors, generator=self.mask_gen, device=device)
        keep = u >= self.mask_prob
        # A sample with every sensor hidden is resampled until at least one
        # survives. At p <= 0.20 and N = 14 this is essentially unreachable
        # (2e-10 per sample); the count is recorded anyway.
        for _ in range(64):
            bad = ~keep.any(dim=1)
            n_bad = int(bad.sum())
            if n_bad == 0:
                break
            self.resampled += n_bad
            u2 = torch.rand(n_bad, self.n_sensors, generator=self.mask_gen, device=device)
            keep[bad] = u2 >= self.mask_prob
        if not bool(keep.any(dim=1).all()):
            keep[~keep.any(dim=1), 0] = True
        self.hidden += int((~keep).sum())
        self.slots += int(keep.numel())
        return keep

    # ------------------------------------------------------------------ noise
    def add_noise(self, x, time_mask, cond_seq):
        """x: (B, L, N); time_mask: (B, L); cond_seq: (B, L) int.

        The scale is looked up per time step from the frozen training-side
        matrix, so a window that crosses operating conditions uses each step's
        own scale. Padded time steps receive no noise. The tensor is never
        modified in place.
        """
        if self.noise_gen is None:
            raise RuntimeError('add_noise called without a noise scale')
        eps = torch.randn(x.shape, generator=self.noise_gen, device=x.device, dtype=x.dtype)
        scale = self.noise_scale[cond_seq.long()]                 # (B, L, N)
        n = scale * eps * time_mask.unsqueeze(-1).to(x.dtype)
        self.noise_sq += float((n.double() ** 2).sum())
        self.noise_n += int(time_mask.sum()) * x.shape[2]
        return x + n

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return dict(
            mask_state=self.mask_gen.get_state() if self.mask_gen is not None else None,
            noise_state=self.noise_gen.get_state() if self.noise_gen is not None else None,
            hidden=int(self.hidden), slots=int(self.slots), resampled=int(self.resampled),
            noise_sq=float(self.noise_sq), noise_n=int(self.noise_n),
        )

    def load_state_dict(self, st):
        if self.mask_gen is not None and st.get('mask_state') is not None:
            self.mask_gen.set_state(st['mask_state'])
        if self.noise_gen is not None and st.get('noise_state') is not None:
            self.noise_gen.set_state(st['noise_state'])
        self.hidden = int(st.get('hidden', 0))
        self.slots = int(st.get('slots', 0))
        self.resampled = int(st.get('resampled', 0))
        self.noise_sq = float(st.get('noise_sq', 0.0))
        self.noise_n = int(st.get('noise_n', 0))
