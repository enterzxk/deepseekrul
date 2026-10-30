# E6 — Statistical audit

Date 2026-10-30 00:05 UTC+8. CPU only. Script: `scripts/paper_e6_stats.py`
(fixed here as the single CI recipe for the paper). Artefact:
`results/dsh/e6/E6_stats.json`.

## 1. Conventions (binding from here on)

| item | rule |
|---|---|
| effect sign | **Δ = RMSE(first) − RMSE(second)**; negative ⇒ first is better |
| interval | engine-clustered bootstrap: engines resampled **within** seed, seeds averaged per replicate |
| resamples | **100,000**; RNG 7000/7001/7002; the SD of the bound across the three is reported as **Monte-Carlo error** |
| NASA | the **mean over engines** of the per-engine asymmetric score; the "sum" convention is ≈ n× larger and must never be mixed with it |
| MDE | **unpaired, across engines** (the 1.60 on 260 engines) is *not* the same quantity as a **paired** bound on a re-run of the same engines (the ±0.3 seen for PE−A) |
| multiplicity | Holm, applied within a family; an unadjusted interval that crosses zero cannot survive Holm |

Earlier R3.1 write-ups used the opposite sign convention ("positive = first better").
The magnitudes are unchanged; every number below is in the new convention and the
old tables must be read with the sign flipped, not re-measured.

## 2. Recomputed contrasts (all with the recipe above)

| contrast | ΔRMSE | 95% CI | MC error | n engines |
|---|---:|---|---:|---:|
| R3.1 est − cycle | −0.046 | [−2.251, +2.129] | ±0.010 | 52 |
| R3.1 est_rate − cycle | −1.176 | [−3.650, +1.331] | ±0.008 | 52 |
| R3.1 est − pooled | +1.057 | [−1.156, +3.244] | ±0.022 | 52 |
| R3.1 oracle − cycle | **−8.191** | [−11.070, −5.390] | ±0.017 | 52 |
| E3 PE − A | −0.898 | [−2.081, **+0.317**] | ±0.007 | 52 |
| E3 BE − A | **+2.693** | [+1.352, +4.096] | ±0.010 | 52 |
| E3 BEfix − A | +2.693 | [+1.352, +4.096] | ±0.010 | 52 |
| R3-E cycle − nocycle (pooled) | **−6.894** | [−8.173, −5.624] | ±0.011 | 360 |
| R3-E cycle − nocycle (FD002) | **−8.859** | [−10.367, −7.359] | ±0.017 | 260 |
| R3-E PE − nocycle (pooled) | **−6.957** | [−8.223, −5.697] | ±0.011 | 360 |

The PE−A upper bound (**+0.317**) is the converged value from T0; the 4,000-resample
+0.291 is confirmed as a low draw, now under a single convention.

## 3. Claims that cannot be recomputed (must be downgraded or rebuilt)

| claim | state of the evidence | action |
|---|---|---|
| **C2 MDE 1.60 / 1.14 / 0.97** | only in `FINAL_DELIVERABLE.md` prose; the paired-difference SD (238.9) and the resampling check (1.58) are not in any artefact | rebuild from a per-engine residual table, or restate as "estimated" |
| **C3 21 paired comparisons, 0 after Bonferroni** | **no per-engine predictions for the 7 methods exist.** The nearest artefact, `clustered_bootstrap_vs_ref.csv`, is a different design: 18 comparisons *against one reference*, on 20-engine (FD001) and 52-engine (FD002) subsets, and **4 of the 18 already exclude zero unadjusted** | **withdraw the "21 pairs / 0 significant" sentence**; a replacement needs the 7 cells re-run with stored predictions |
| **C10 cap sweep 24.8 / 26.9 / 12.06** | `cutoff_sensitivity.csv` holds aggregates, no per-engine rows | report as a point summary without CI, or rebuild |
| **C1 variance decomposition 2.6 / 31.5 / 65.9** | point estimates in `c_interaction.json`; no interval | report as point estimates and say so |

## 4. Consequences for the manuscript

1. §4.2 can no longer open with "21 comparisons, none survives correction". It must
   either (a) rebuild the 7-cell tournament with stored per-engine predictions, or
   (b) rest on what is reproducible: the recomputed E3/R3.1 contrasts, whose CIs are
   ±2–4 RMSE wide on 52 engines, plus the arithmetic MDE.
2. The resolution argument gets **stronger**, not weaker: the one comparison that
   looked like it passed (PE−A at 4k resamples) fails at 100k, and the "21 pairs"
   claim turns out to have no artefact behind it. Both are the paper's thesis.
3. §4.3 must quote PE−A as **−0.898 [−2.081, +0.317]** — never as a win.

## 5. Reproduce

```bash
ssh lab-server
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/paper_e6_stats.py
```

Runtime ≈ 3 min (10 contrasts × 3 RNG × 100k resamples).
