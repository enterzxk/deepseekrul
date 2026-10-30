# E3c — Result: the bounded encoder's advantage is alignment, not capacity

Run 2026-10-30 08:42–09:20 UTC+8, 6 runs (PS × 3 seeds × 2 subsets), prereg
`paper/E3C_SCRAMBLE_PREREGISTRATION.md`. Statistics: E6 recipe (engine-clustered,
100k resamples, RNG 7000–7002).

| arm (FD002) | RMSE matched | val |
|---|---:|---:|
| A (linear `c/250`) | 12.676 | 12.533 |
| PE (bounded, aligned) | **11.768** | 12.205 |
| **PS (bounded, scrambled)** | **15.265** | 13.738 |
| BE (no time) | 15.329 | 13.867 |

| contrast | FD002 | FD001 |
|---|---:|---:|
| PS − PE | **+3.512** [+2.214, +4.857] | **+2.144** [+0.366, +4.093] |
| PS − A | +2.615 [+1.113, +4.194] | +1.216 [−0.453, +2.879] |
| **PS − BE** | **−0.078** [−1.193, +0.995] | −0.47 (point only, 20 engines) |

## Reading (pre-registered rule, first row of the table)

**H-time is confirmed; H-capacity is rejected.** The scrambled arm has the same
encoder, the same +448 parameters, the same periods and the same multiset of ages in
every batch. It differs only in that a window no longer receives *its own* age. It is
3.5 RMSE worse than PE on FD002 and 2.1 worse on FD001. The PE advantage therefore
comes from alignment with the sample's own age and cannot be attributed to the extra
parameters.

The second result is the coherence check that makes the first credible: **PS is
statistically indistinguishable from BE** (−0.078 [−1.193, +0.995]). A bounded encoder
fed unaligned time behaves exactly like no time at all, which is what a valid control
should do — it confirms that the scramble destroyed the information and nothing else.

## What this changes in §4.3

* The sentence "the gain might be capacity" is removed from the threats list: it was
  tested and rejected.
* §4.3.2 can now state the mechanism precisely: *what matters is the presence of
  aligned time, not the shape of its encoding.* PE beats A by 0.90 at the point
  estimate, but the pre-registered margin is still not met — so "bounded is better"
  remains unsupported, while "bounded is usable" is now supported by a mechanism.
* One more number for the shortcut argument: destroying alignment costs 3.5 RMSE,
  which is larger than every method-identity effect measured in this project.

## Boundary

* FD001's PS runs are unstable across seeds (matched 12.36 / 15.11 / 12.67); with 20
  engines the contrast PS − A is not resolvable, and only PS − PE is.
* The permutation is drawn from a generator seeded to 0 on every forward pass, so it is
  deterministic but fixed across steps; a sample therefore receives a *consistent*
  wrong age rather than a fresh one each epoch. The control destroys alignment, not the
  learnability of a fixed offset.
* Three seeds per cell; no claim of a regime difference.

Artefacts: `experiments/e3c/*`, `results/dsh/e6/E6_stats.json` (contrast keys
`E3c ...`), code `src/candidates/model.py` sha `e2237626…`,
`scripts/run_candidate.py` sha `65eb5aca…`. `use_pe=False` remains bit-identical
(forward and 3-step loss diffs 0.0) after the change.
