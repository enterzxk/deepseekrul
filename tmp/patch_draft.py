from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
# fill section II with the related-work summary
old_ii = "## II. MEASUREMENT PROBLEM AND RELATED WORK\n\n<TBD — N1–N4 from NOVELTY_MATRIX.md. N2's phase-aware normalisation formula must be\nverified against the full text before submission.>"
new_ii = """## II. MEASUREMENT PROBLEM AND RELATED WORK

Four works are the nearest neighbours of this study (full comparison in
\`NOVELTY_MATRIX.md\`).

Qu et al. [N1] build a condition-aware spatio-temporal network in which operating settings
and sensor measurements interact through a synergistic feature-interaction module and a
hierarchical graph attention network. The condition handling is learned end to end inside a
large model. Our study keeps a 42.3k-parameter backbone fixed and asks instead what part of
the condition handling can be made explicit and what part of the individual lifetime
information is actually observable.

Kapoor et al. [N2] introduce a phase-aware normalisation together with delta features to
stabilise feature distributions across C-MAPSS regimes. We do not claim phase normalisation
as a contribution; on the contrary, our full-budget experiment shows that replacing the
cycle coordinate by the estimated lifetime phase c/mu_T degrades the pooled RMSE by 7.5%,
and we report that boundary rather than a positive result. The exact definition of the
phase in [N2] must be checked against the full text before submission.

Piao et al. [N3] convert a point predictor into a conformal interval with coverage
guarantees. Our intervals are engine-clustered bootstrap percentiles and carry no coverage
guarantee; the two answer different questions and are not interchangeable. We deliberately
do not describe our RMSE as an instrument uncertainty.

Gupta [N4] audits operating-condition normalisation in C-MAPSS evaluation and shows that
per-condition normalisation lowers RMSE in every matched multi-regime cell, and that
refitting the final model and the normaliser on train+validation changes the reported RMSE
by up to 1.1 cycles. We take per-condition normalisation as an established premise, freeze
the fit scope in the protocol, and do not present it as a contribution.

The gap this paper addresses is therefore not architectural: it is the absence of a
quantitative answer to how observable the individual lifetime scale is, which of its uses
survive a fair budget, and whether an adaptive correction can be identified from
observables."""
assert old_ii in s
s = s.replace(old_ii, new_ii)
old_vii = "## VII. RESULTS\n\n<TBD — Tables II–VII; Figures 1–5.>"
new_vii = """## VII. RESULTS

The complete results are in \`PAPER_RESULTS.md\` and in the generated tables and figures
(\`paper_tables/\`, \`paper_figures/\`). In summary:

1. On FD002 the sensor-history prior (FULL) is the best single arm on all four metrics
   (RMSE 13.054, MAE 9.098, NASA 3.492, bias +0.288) and is better than the plain backbone
   on 5/5 folds; the paired interval of the difference is [+0.071, +0.823] cycles.
2. On FD004 FULL wins RMSE (13.253) and MAE (8.662) but loses the NASA score to the plain
   backbone (11.251 against 8.980), the paired interval crosses zero, and FULL wins on
   only 3/5 folds.
3. Replacing the cycle coordinate by c/mu_T costs +1.034 [+0.595, +1.468] cycles on FD002
   and +1.320 [+0.783, +1.879] on FD004; multiplying a predicted life fraction by mu_T
   costs +3.909 [+2.383, +5.421] on FD002; the differential-window, engine-balanced-weight
   and sensor-attention variants are neutral.
4. The lifetime scale's correlation with the true lifetime rises from 0.26 at 50 observed
   cycles to 0.92 at 150 (FD002) and from 0.53 to 0.88 (FD004).
5. No observable predicts the individual-versus-population advantage (|rho| < 0.08), and
   every gate is worse than a fixed 0.5 fusion, although the per-window oracle is 15%
   better than that fusion.
6. Under 20% random missing samples FULL keeps its advantage (13.243 to 12.838 clean,
   15.289 to 14.862 under MC1); under a one-standard-deviation offset on three channels
   both estimators collapse (36.9 and 39.7) and FULL is now worse."""
assert old_vii in s
s = s.replace(old_vii, new_vii)
p.write_text(s); print('patched')
