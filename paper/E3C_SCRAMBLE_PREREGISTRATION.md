# E3c — Pre-registration: scrambled-time control for the PE arm

Frozen 2026-10-30 08:40 UTC+8, before any PS run.

## Why

PE (bounded sin/cos encoding) beat A by 0.90–0.93 RMSE on both subsets, though the
pre-registered margin was never met. Two explanations remain, and the paper cannot
distinguish them:

* **H-time**: the gain comes from encoding the sample's *own* age in a bounded form.
* **H-capacity**: the gain comes from the extra 448 parameters / extra input channels
  acting as regularisation, independent of alignment.

A parameter-matched zero-pad control cannot separate these (zero inputs receive no
gradient). The cheapest informative control is to keep the encoder, the parameter
count and the marginal distribution of time values, and destroy only the
sample-to-age alignment.

## Design

| | |
|---|---|
| arm **PS** | identical to PE (`pe_proj`, same periods, same +448), but the batch's cycle vector is permuted by a fixed permutation before the encoding |
| subsets | FD002 and FD001, 3 seeds each (2027–2029), 80 epochs, patience 20 |
| reference | the existing A / BE / PE runs (unchanged, same seeds and splits) |

Implementation detail: the permutation is drawn once per forward from a generator
seeded to 0, so it is deterministic and data-independent; every sample receives
another sample's age, and the multiset of ages in a batch is unchanged.

## Decision rule (fixed now)

| outcome | reading |
|---|---|
| `PS − PE > 0`, CI excludes 0 | alignment carries the gain → the paper may say the bounded encoding *uses* time, and the PE–A difference is not a capacity artefact |
| `PS − PE ≈ 0`, CI includes 0 | the gain is capacity/regularisation, not alignment → the "bounded encoding" sentence is dropped; §4.3 keeps only the BE result |
| `PS − A < 0`, CI excludes 0 | scrambled time is *better* than the linear coordinate → strongest form: what matters is boundedness, not alignment |

No margin is moved and no arm is re-run to change a verdict.

## Cost

6 runs ≈ 0.5 GPU-hour, ~35 min wall at 3-way concurrency.
Artefacts: `experiments/e3c/*`, `results/dsh/e3/SCRAMBLE.json`.
