"""Learned vector response codes; none of these coordinates is a physical HI.

Spectral rank/range choices are detached hard decisions. Linear solves inside
the selected ranges retain gradients. Gram matrices describe excitation, not
calibrated uncertainty. No statistics are fitted across flights in this module.
"""
from __future__ import annotations

import math
import torch
from torch import nn

MODES = ("raw", "fe", "delta", "supported", "common")
LATENT = 4
RIDGE = 1e-2
MIN_SUPPORT = 32
RANK_RTOL = 1e-4
RANK_ATOL = 1e-6
RANGE_TOL = 1e-5
BASIS_PENALTY = 1e-4


def symmetric(q):
    return (q + q.transpose(-1, -2)) * .5


def identified_inverse(q, rtol=RANK_RTOL, atol=RANK_ATOL):
    """Truncated PSD inverse, projector, cleaned Gram and detached rank.

    Avoid differentiating eigenvectors at repeated/zero eigenvalues: select an
    orthonormal range without gradients, then differentiate an ordinary solve
    restricted to that range. This is an explicit fixed-range gradient policy.
    """
    q = symmetric(q)
    if q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("expected batched square Gram matrices")
    with torch.no_grad():
        values, vectors = torch.linalg.eigh(q.detach())
        threshold = torch.maximum(values[:, -1:].clamp_min(0) * rtol,
                                  values.new_full((len(q), 1), atol))
        rank = (values > threshold).sum(-1)
    inverse = torch.zeros_like(q)
    projector = torch.zeros_like(q)
    cleaned = torch.zeros_like(q)
    for r in range(1, q.shape[-1] + 1):
        ids = torch.nonzero(rank == r, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        u = vectors[ids, :, -r:].detach()
        small = symmetric(u.transpose(-1, -2) @ q[ids] @ u)
        inv = u @ torch.linalg.solve(small, u.transpose(-1, -2))
        inverse = inverse.index_copy(0, ids, symmetric(inv))
        projector = projector.index_copy(0, ids, u @ u.transpose(-1, -2))
        cleaned = cleaned.index_copy(0, ids, symmetric(u @ small @ u.transpose(-1, -2)))
    return inverse, projector.detach(), cleaned, rank.detach()


def psd_parallel_sum(current, reference):
    """PSD parallel sum with an explicit common-range numerical convention.

    In particular this is NOT pinv(pinv(current) + pinv(reference)). Directions
    outside the intersection of the two identified ranges are exactly zero.
    """
    _, pc, qc, rc = identified_inverse(current)
    _, pr, qr, rr = identified_inverse(reference)
    inverse, _, _, _ = identified_inverse(qc + qr)
    h = symmetric(qc - qc @ inverse @ qc)
    # The intersection has eigenvalue two in the sum of orthogonal projectors.
    with torch.no_grad():
        ev, uv = torch.linalg.eigh(symmetric(pc + pr))
        keep = ev > (2. - RANGE_TOL)
        common = (uv * keep.unsqueeze(-2)) @ uv.transpose(-1, -2)
    h = symmetric(common @ h @ common)
    _, ph, h, rank = identified_inverse(h)
    return h, ph, {"rank_current": rc, "rank_reference": rr, "common_rank": rank}


def response_moments(a, residual, mask):
    """Mean-over-observation Gram, with sensors summed, not independent samples."""
    weight = mask.to(a.dtype)
    count = weight.sum(-1)
    denominator = count.clamp_min(1)
    gram = torch.einsum("ftsk,ftsl,ft->fkl", a, a, weight) / denominator[:, None, None]
    rhs = torch.einsum("ftsk,fts,ft->fk", a, residual, weight) / denominator[:, None]
    return symmetric(gram), rhs, count


def ridge_code(a, residual, mask):
    q, rhs, count = response_moments(a, residual, mask)
    eye = torch.eye(LATENT, device=q.device, dtype=q.dtype)
    h = torch.linalg.solve(q + RIDGE * eye, rhs.unsqueeze(-1)).squeeze(-1)
    return h, q, count


def common_contrast(a, residual, prior_index, support_current, support_ref):
    """Supported OLS contrast; H retains local solve gradients for weighting."""
    qc, vc, nc = response_moments(a, residual, support_current)
    qr, vr, nr = response_moments(a[prior_index], residual[prior_index], support_ref)
    ic, _, _, _ = identified_inverse(qc)
    ir, _, _, _ = identified_inverse(qr)
    hc = (ic @ vc.unsqueeze(-1)).squeeze(-1)
    hr = (ir @ vr.unsqueeze(-1)).squeeze(-1)
    hmetric, projection, info = psd_parallel_sum(qc, qr)
    usable = ((nc >= MIN_SUPPORT) & (nr >= MIN_SUPPORT) &
              (info["common_rank"] > 0) &
              (prior_index != torch.arange(len(a), device=a.device)))
    raw_difference = hc - hr
    difference = (projection @ raw_difference.unsqueeze(-1)).squeeze(-1)
    projection_removed = torch.linalg.vector_norm(difference - raw_difference, dim=-1)
    difference = torch.where(usable[:, None], difference, torch.zeros_like(difference))
    info.update(support_current_count=nc.detach(), support_ref_count=nr.detach(),
                common_supported=usable.detach(), parallel_sum=hmetric,
                common_projector=projection, current_ols=hc.detach(), reference_ols=hr.detach(),
                projection_removed_norm=projection_removed.detach(),
                rank_below_latent_fraction=(info["common_rank"] < LATENT).to(a.dtype).mean().detach(),
                supported_fraction=usable.to(a.dtype).mean().detach())
    return difference, info


def direction_weighted_contrast(projected, hmetric, rank, usable=None):
    """H/(trace(H)/r) on the common range; rank zero returns exact zero.

    This normalization discards overall excitation magnitude. It measures only
    relative excitation directions, not global reliability/noise confidence.
    """
    mean_eigenvalue = (hmetric.diagonal(dim1=-2, dim2=-1).sum(-1) /
                       rank.clamp_min(1).to(hmetric.dtype)).clamp_min(1e-6)
    weighted = (hmetric @ projected.unsqueeze(-1)).squeeze(-1) / mean_eigenvalue[:, None]
    active = rank > 0
    if usable is not None:
        active = active & usable
    return torch.where(active[:, None], weighted, torch.zeros_like(weighted))


class ResponseEncoder(nn.Module):
    """Encode independent completed flights, plus a registered past-flight code.

    raw: [F,T,18], normalized W4 then Xs14; production T=512, four normalized
    position blocks. Indices must refer to the same unit in the caller's cache;
    this API has no unit IDs, so it enforces only index-causality. Self indices
    for a unit's first flight are allowed and have zero common innovation.
    """
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        # Construct identically in every arm, including the raw auxiliary head.
        self.raw_encoder = nn.Sequential(nn.Linear(18, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.response_body = nn.Sequential(nn.Linear(4, 32), nn.Tanh())
        self.background = nn.Linear(32, 14)
        self.basis = nn.Linear(32, 14 * LATENT)
        self.aux_context_head = nn.Linear(64, LATENT)
        self.embedding = nn.Linear(64 + LATENT + LATENT + 1, 64)

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        if self.mode != "raw":
            inactive += sum(p.numel() for p in self.aux_context_head.parameters())
        zero_columns = 8 if self.mode == "raw" else (4 if self.mode == "fe" else 0)
        inactive += 64 * zero_columns
        return {"total_parameters": total,
                "structurally_effective_parameters": total - inactive,
                "inactive_by_design": inactive,
                "scope": "includes auxiliary loss; sample-dependent common fallback not subtracted"}

    def forward(self, raw, prior_index, support_current, support_ref, block=None,
                *, audit_full_common=False):
        if raw.ndim != 3 or raw.shape[-1] != 18 or raw.shape[1] < 4 or raw.shape[1] % 4:
            raise ValueError("expected [flights,T,18], T divisible by four (production T=512)")
        if raw.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(raw).all()):
            raise ValueError("raw must contain finite FP32/FP64 observations")
        f, t, _ = raw.shape
        if f == 0 or prior_index.shape != (f,) or prior_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("one integer past-flight index required per flight")
        prior_index = prior_index.to(device=raw.device, dtype=torch.long)
        if bool(((prior_index < 0) | (prior_index > torch.arange(f, device=raw.device))).any()):
            raise ValueError("future or out-of-range prior index")
        masks = []
        for mask in (support_current, support_ref):
            if mask.shape != (f, t) or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("support masks must be binary [flights,T]")
            masks.append(mask.to(device=raw.device, dtype=torch.bool))
        sc, sr = masks
        tokens = self.raw_encoder(raw)
        raw_embedding = tokens.mean(1)
        hidden = self.response_body(raw[..., :4])
        background = self.background(hidden)
        a = torch.tanh(self.basis(hidden)).reshape(f, t, 14, LATENT) / math.sqrt(14.)
        residual = raw[..., 4:] - background
        all_points = torch.ones((f, t), device=raw.device, dtype=torch.bool)
        full_h, _, _ = ridge_code(a, residual, all_points)
        coverage = torch.minimum(sc.sum(-1), sr.sum(-1)).to(raw.dtype) / t
        # These spectral quantities have no path to embedding/auxiliary loss in
        # raw/fe/delta. The audit switch reproduces the original diagnostic path.
        compute_common = self.mode in ("supported", "common") or audit_full_common
        if compute_common:
            supported, support_info = common_contrast(a, residual, prior_index, sc, sr)
            weighted = direction_weighted_contrast(
                supported, support_info["parallel_sum"], support_info["common_rank"],
                support_info["common_supported"])
        else:
            support_info = {}
        level = full_h if self.mode != "raw" else torch.zeros_like(full_h)
        if self.mode == "delta":
            innovation = full_h - full_h[prior_index]
        elif self.mode == "supported":
            innovation = supported
        elif self.mode == "common":
            innovation = weighted
        else:
            innovation = torch.zeros_like(full_h)
        encoded = self.embedding(torch.cat([raw_embedding, level, innovation, coverage[:, None]], -1))
        # The temporal caller must choose this view whenever a member's prior
        # precedes the current query's history boundary. Remove coverage too:
        # even a W-only support statistic otherwise reveals that earlier flight.
        without_reference = self.embedding(torch.cat([
            raw_embedding, level, torch.zeros_like(innovation),
            torch.zeros_like(coverage[:, None])], -1))

        aux_h = torch.zeros_like(full_h)
        per_flight = raw.new_zeros(f)
        if block is not None:
            if not isinstance(block, int) or not 0 <= block < 4:
                raise ValueError("block must be None or an integer in [0,3]")
            positions = torch.arange(t, device=raw.device)
            query = (positions // (t // 4)) == block
            context = (~query).unsqueeze(0).expand(f, -1)
            if self.mode == "raw":
                # Neither query Xs nor full-flight raw embedding enters this code.
                pooled_context = (tokens * context.unsqueeze(-1)).sum(1) / context.sum(1, keepdim=True)
                aux_h = self.aux_context_head(pooled_context)
            else:
                aux_h, _, _ = ridge_code(a, residual, context)
            reconstructed = background[:, query] + torch.einsum("ftsk,fk->fts", a[:, query], aux_h)
            per_flight = (reconstructed - raw[:, query, 4:]).square().mean((1, 2))
            per_flight = per_flight + BASIS_PENALTY * a.square().mean((1, 2, 3))
        if compute_common:
            weighting_change = torch.linalg.vector_norm(weighted - supported, dim=-1)
            support_info.update(
                supported_innovation=supported.detach(), weighted_innovation=weighted.detach(),
                weighting_change_norm=weighting_change.detach(),
                weighting_relative_change=(weighting_change /
                    torch.linalg.vector_norm(supported, dim=-1).clamp_min(1e-12)).detach())
        diagnostics = {k: v.detach() if isinstance(v, torch.Tensor) else v
                       for k, v in support_info.items()}
        diagnostics.update({"common_diagnostics_status": "computed" if compute_common else "not_computed",
                       "coverage": coverage.detach(),
                       "level_code": full_h.detach(), "innovation": innovation.detach(),
                       "aux_context_latent": aux_h.detach(), "basis_penalty": BASIS_PENALTY,
                       "rank_rtol": RANK_RTOL, "rank_atol": RANK_ATOL,
                       "minimum_support_points": MIN_SUPPORT,
                       "spectral_geometry_detached": True,
                       "weighting_scope": "trace-normalized common-excitation directions; not confidence"})
        return {"embedding": encoded, "embedding_without_reference": without_reference,
                "reconstruction_loss": per_flight.mean(),
                "reconstruction_per_flight": per_flight, "diagnostics": diagnostics}
