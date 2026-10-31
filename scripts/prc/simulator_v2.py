"""PRC-R2 simulator: latent-prediction construction with protocol transforms.

Design commitments (R2 brief, part 1):
  * A model outputs a *prediction*; a protocol is a transform of (prediction, truth)
    into an error. Sign reversal therefore comes from the interaction between the
    transform (cap, coverage) and a shared latent prediction structure — not from
    flipping the sign of an error mean.
  * All protocols are derived from the SAME two prediction vectors, so protocol
    dependence is built in and an independence assumption would be false.
  * Fold structure: each engine belongs to a fold; the fold's contribution to the
    prediction is a fold-specific constant. `fold_overlap` controls how much of the
    prediction error is shared across folds (5-fold CV shares 4/5 of training engines).

Targets:
  CONDITIONAL  fold constants and seed constants are fixed at their realised values;
               only engines vary. Target = population delta of the given models.
  UNCONDITIONAL fold and seed components are zero-mean random; their variance enters
               the target. Target = population delta of the training procedure.
"""
from __future__ import annotations

import numpy as np

PROTOCOLS = ("cap_all", "uncap_all", "short", "mid")


def transform(yhat, life, protocol, *, cap=125.0, short_threshold=80.0,
              mid_lo=100.0):
    pred = np.clip(yhat, 0.0, cap)
    if protocol == "cap_all":
        return pred - np.minimum(life, cap), np.ones(life.shape, bool)
    if protocol == "uncap_all":
        return pred - life, np.ones(life.shape, bool)
    if protocol == "short":
        return pred - np.minimum(life, cap), life < short_threshold
    if protocol == "mid":
        # mid segment: above the short threshold but below the cap, so the cap does
        # not erase the model difference (above the cap every clipped prediction is
        # 125 and the errors coincide).
        return pred - np.minimum(life, cap), (life >= mid_lo) & (life < cap)
    raise ValueError(protocol)


def _population_delta(spec, pop_rng, pop_reps, fold_const, seed_const):
    """delta_p on a large population draw with the given (possibly zero) constants."""
    life = pop_rng.uniform(spec["life_lo"], spec["life_hi"], pop_reps)
    diff = pop_rng.normal(0.0, spec["sigma_u"], pop_reps)
    fold = np.zeros(pop_reps)
    if fold_const is not None:
        ids = pop_rng.integers(0, spec["n_fold"], pop_reps)
        fold = fold_const[ids]
    yhat = {}
    life_mid = 0.5 * (spec["life_lo"] + spec["life_hi"])
    for m in ("A", "B"):
        s = 0.0 if seed_const is None else seed_const[m]
        mdl = spec["models"][m]
        sig = mdl["sigma"] * np.exp(mdl.get("sigma_growth", 0.0)
                                    * (life - life_mid) / 100.0)
        noise = pop_rng.normal(0.0, 1.0, pop_reps) * sig
        bias = np.where(life < spec.get("bias_split", 80.0),
                        mdl.get("bias_short", mdl["bias"]),
                        mdl.get("bias_long", mdl["bias"]))
        yhat[m] = np.clip(life + diff + bias + fold + s + noise, 0.0, spec["cap"])
    out = []
    for p in spec["protocols"]:
        r = {}
        for m in ("A", "B"):
            e, keep = transform(yhat[m], life, p, cap=spec["cap"])
            r[m] = float(np.sqrt(np.mean(e[keep] ** 2)))
        out.append(r["B"] - r["A"])
    return np.array(out)


def make_scenario(rng, spec, n_engine=200, inference="conditional",
                  pop_reps=400_000):
    k = len(spec["protocols"])
    fold_ids = rng.integers(0, spec["n_fold"], n_engine)

    # realised fold constants: one draw per fold id, shared by every engine in it
    fold_vec = rng.normal(0.0, spec["sigma_fold"], spec["n_fold"])
    fold_const = fold_vec.copy()
    seed_const = {m: float(rng.normal(0.0, spec["sigma_seed"])) for m in ("A", "B")}

    if inference == "conditional":
        pop_fold_const, pop_seed_const = fold_const, seed_const
    else:
        # expectation over training randomness: constants are zero, their variance is
        # *not* silently dropped — it enters through the fold/seed draws of each
        # replicate, and the target uses the zero-expectation version
        pop_fold_const, pop_seed_const = None, None

    pop_rng = np.random.default_rng(int(rng.integers(1 << 63)))
    delta_true = _population_delta(spec, pop_rng, pop_reps, pop_fold_const,
                                   pop_seed_const)

    life = rng.uniform(spec["life_lo"], spec["life_hi"], n_engine)
    diff = rng.normal(0.0, spec["sigma_u"], n_engine)
    fold_eff = fold_const[fold_ids]
    yhat = {}
    life_mid = 0.5 * (spec["life_lo"] + spec["life_hi"])
    for m in ("A", "B"):
        mdl = spec["models"][m]
        sig = mdl["sigma"] * np.exp(mdl.get("sigma_growth", 0.0)
                                    * (life - life_mid) / 100.0)
        noise = rng.normal(0.0, 1.0, n_engine) * sig
        bias = np.where(life < spec.get("bias_split", 80.0),
                        mdl.get("bias_short", mdl["bias"]),
                        mdl.get("bias_long", mdl["bias"]))
        yhat[m] = np.clip(life + diff + bias + fold_eff
                          + seed_const[m] + noise, 0.0, spec["cap"])

    err_a = np.full((n_engine, k), np.nan)
    err_b = np.full((n_engine, k), np.nan)
    for j, p in enumerate(spec["protocols"]):
        ea, keep = transform(yhat["A"], life, p, cap=spec["cap"])
        eb, _ = transform(yhat["B"], life, p, cap=spec["cap"])
        err_a[keep, j] = ea[keep]
        err_b[keep, j] = eb[keep]

    return dict(err_a=err_a, err_b=err_b, delta_true=delta_true,
                theta_true=float(delta_true.min()), max_true=float(delta_true.max()),
                life=life, fold_ids=fold_ids)
