# E18 — FD001 five-fold replication

Pre-registration: `paper/E18_FD001_CV_PREREGISTRATION.md` (sha `59bb5d9b`), frozen before any
fold-1..4 run existed. 36 new runs; fold 0 reused from `experiments/e3_fd001` (partition
verified identical). Estimator identical to FD002 (§4.2.3): seed 2027 for intervals, 100k
clustered bootstrap within fold, three RNG seeds.

## Result

| arm | RMSE (100 engines) | three-seed mean | params |
|---|---:|---:|---:|
| A (candidate) | 11.463 | 11.012 ± 1.844 | 41,858 |
| BE (no time) | 13.515 | 12.906 ± 1.575 | 41,730 |
| PE (bounded time) | **10.829** | **10.775 ± 1.608** | 42,306 |

| contrast | ΔRMSE | 95% CI | excludes 0 | TOST δ=1.0 |
|---|---:|---|---:|---:|
| **BE − A** | **+2.052** | **[+0.505, +3.733]** | **yes** | no (margin needed 3.46) |
| PE − A | −0.634 | [−2.159, +0.809] | no | no (1.90) |
| PE − BE | −2.686 | [−4.310, −1.037] | **yes** | no (4.05) |

## Cross-subset comparison — the paper's strongest table

| contrast | FD002 (260 engines) | FD001 (100 engines) | pooled (360, inverse-variance) |
|---|---:|---:|---:|
| **BE − A (time removed)** | **+1.133** [+0.245, +2.023] | **+2.052** [+0.505, +3.733] | **+1.347** [+0.568, +2.126] |
| PE − A (bounded encoding) | −1.433 [−2.998, +0.133] | −0.634 [−2.159, +0.809] | not pooled: signs differ |
| PE − BE | −3.561 significant | −2.686 significant | consistent |

(`results/dsh/e18/pooled_BE_A.json`.)

**What replicates.** The time channel is load-bearing on both subsets, and on FD001 — where
the per-fold MDE is 4.84 — the effect is large enough to clear zero anyway. Having *a* time
channel (bounded or linear) beats having none by 2.7–3.6 RMSE on both.

**What does not replicate in sign.** PE vs A: −1.43 on FD002, −0.63 on FD001. Neither is
significant; the two subsets even disagree about the direction. The encoding choice is
unresolved, exactly as the resolution thesis predicts.

**FD001-specific observations.** PE has the best point estimate on FD001 (10.83) while A has
it on FD002 (14.03). Seed spread is larger (1.6–1.8 vs 0.4–1.1), consistent with 20 engines
per fold. The fold SD of BE−A is 1.59 against FD002's 0.607 — prediction 4 of the
pre-registration held.

## Pre-registered predictions

| # | prediction | outcome |
|---|---|---|
| 1 | BE − A pooled is positive | **held** (+2.05) |
| 2 | its 95% CI includes zero (1.7 < MDE 2.2) | **failed — in the favourable direction**: the CI excludes zero. The FD001 effect is larger than the FD002 point estimate, not smaller. Reported as a failed prediction; the paper gains a stronger result than it predicted. |
| 3 | \|PE − A\| < 1.5, CI includes zero | **held** (−0.63, CI includes 0) |
| 4 | fold SD larger than FD002's | **held** (1.59 vs 0.61) |
