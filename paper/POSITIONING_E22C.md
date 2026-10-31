# E22c — Positioning: from "a model that predicts RUL" to "a measurement study of RUL evaluation"

Status: internal positioning note, 2026-11-01. Feeds SECTION_1 (contributions) and
SECTION_5 (guidelines); does not replace the pre-registrations.

## The shift

Previous framing: "we analyse the resolution of C-MAPSS and report what it can and
cannot measure, using our own model as the instrument." Weakness: a reviewer can ask
what the *method* contribution is, beyond a set of well-executed experiments.

New framing, adopted after E21/E22a: **the paper's subject is the measurement process
itself.** The questions are about how evaluation choices change conclusions, not about
which model wins. The model becomes an instrument that is itself characterised — with
known resolution, known failure modes, and a demonstrated ability to reproduce a strong
published baseline's performance under a common protocol (E22b).

Three research questions, each answered by evidence that already exists or is being
frozen in E22b:

**RQ1 — How does the definition of the evaluation label change model rankings?**
Evidence:
* Capped vs uncapped truth: FD002 23.39 → 11.96, FD004 22.40 → 12.27 on *identical*
  predictions; FD001/FD003 move by ~1. The ranking among our three arms does not
  change under capping, but the *gap to published numbers* changes by 11 RMSE — the
  single largest protocol effect measured in this project (§4.3.6, E21 Amendment 1).
* Full-set vs retained-set coverage (E22a): including the 54 short-trajectory engines
  flips the FD002 arm ranking — A degrades 11.96 → 16.66 while PE moves 11.46 → 11.51,
  so PE becomes the best arm on FD002; on FD001/FD003/FD004 the ranking is stable.
  Label/coverage definitions are therefore not cosmetic: they decide which method
  "wins" a subset.
* NASA score under the two conventions differs by up to 3× on the same predictions
  (FD002 mean 9.45 capped vs 53.41 uncapped) — the metric's late-penalty is defined
  against the uncapped truth, so reporting it with a capped truth silently changes
  what it measures.

**RQ2 — How do evaluation protocols change the size of a performance difference?**
Evidence: the four-protocol ladder for one contrast (+2.69 single split → +1.13
five-fold → +0.98 four-subset pooled → +0.77 official test), the per-fold spread
(2.62 RMSE across five equally valid splits), the cutoff dependence (+2.69 matched vs
≈0 fixed-absolute), and the retention law (1.5 RMSE → 1% retained at test; 5.9 → 99%).
This is the paper's most developed axis: each rung is a pre-registered experiment, and
the ladder is what lets §5's guidelines be prescriptive rather than advisory.

**RQ3 — What does reliable statistical inference require on this benchmark?**
Evidence: engine-clustered intervals (97.6% of contrast variance is engine sampling),
MDE closed form and its validation by paired simulation (T1 sample sizes 320–644,
T2 627–1262), TOST equivalence instead of "not significant", the fold/seed decomposition
that kills seed replication as a variance remedy, and the four public withdrawals that
show what happens when inference is done wrongly. Deliverable: G1–G10 with runnable
tools.

## Is there an independent methodological contribution?

An honest inventory, separating what is new from what is borrowed:

| component | status | basis |
|---|---|---|
| Resolution-first protocol (MDE before comparison, engine-clustered inference, equivalence testing, pre-registration with public withdrawals) | **new as an integrated protocol for RUL**; each statistical ingredient is standard | statistical ingredients are textbook; the integration, and the demonstration that it changes published-style verdicts, are not |
| Retention law (effect size → fraction surviving protocol transfer) | **new empirical law** (5 effect sizes, 2 regimes, monotone) | needs a sixth and seventh point to be called a law; E22b adds one (Dual-Mixer vs A) |
| Label-definition sensitivity analysis (capped/uncapped, coverage, cap value) | **new as a systematic treatment**; individual traps are folklore | E21 Amendment 1 and E22a are the calibration experiments |
| Audit apparatus (claims ledger, freeze hashes, unseal chain, 4 withdrawals + 3 amendments) | **new for this literature**; standard in clinical trials | borrows CONSORT/registered-report practice |
| G1–G10 checklist + tools | engineering contribution | value is in being runnable and calibrated, not in novelty of each item |

Conclusion: the contribution is a **measurement methodology** in the TIM sense —
instrument characterisation (resolution, bias sources, protocol sensitivity) plus a
calibrated protocol — not a new estimator. That is a legitimate TIM regular-paper
category. The risk to manage is a reviewer who wants a new model; §1.3 already states
the paper does not claim one, and E22b's job is to remove the "weak instrument" rebuttal
by showing the protocol reproduces a strong published model's level.

## What E22b must deliver for this positioning to hold

1. Dual-Mixer (or the paired A-zscore arm) reaches dev RMSE within ~1.5 of arm A on at
   least two subsets — the resolution claim needs the instrument to be competitive, not
   merely present.
2. The paired design (same folds, same input convention, same training budget) makes the
   comparison cell-by-cell honest; the pre-registration forbids post-hoc tuning.
3. Whatever the outcome, it is reported: a Dual-Mixer that loses by >1.5 RMSE is itself
   a retention-law data point and a statement about protocol transfer.

## Structural mapping

| section | RQ | content |
|---|---|---|
| §1 | all | problem statement: comparisons are made below the resolution of the benchmark |
| §3 | RQ3 | design: units, clustering, pre-registration, provenance |
| §4.1–4.2 | RQ3 | variance decomposition, MDE, equivalence |
| §4.3 | RQ2 | the four-protocol ladder, mechanism control, cross-subset and test transfer |
| §4.4 | — | lifespan ceiling and the estimator that fails (scope limit) |
| §4.5 | RQ1 | label and metric definitions: cap, cutoffs, severity, coverage |
| §4.6 | RQ3 | uncertainty budget and sample-size targets |
| §5 | all | G1–G10 |
| §6 | all | limitations, including what the protocol cannot fix |
