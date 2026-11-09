# 5. Discussion

<!-- draft note -->

## 5.1 What the three results say together

Read separately, each result is unremarkable: conformal intervals can be sharper than constant-width
ones; marginal coverage does not imply conditional coverage; an uncertainty scale can be unstable.
Read together, they describe a specific gap between what is reported and what is used.

The efficiency result (Section 4.1) is robust — 20/20 folds, −10.22% — and it is obtained at matched
coverage. That is a genuine and reproducible property of the pipeline. But the same 80 cells contain
engines whose intervals cover 12–49% of their windows (Section 4.2). A practitioner reading only the
marginal number would conclude that the interval is usable on a new engine; the per-engine distribution
says that for some engines it is not, and that this is not a fluke of one training run.

The scale experiment (Section 4.3) then shows that the natural remedy — condition the width on an
epistemic-uncertainty estimate — does not have a uniform effect. It is not that the scale is useless;
on the multi-operating-condition subsets it improves 19/20 cells of FD002. It is that the sign of the
effect depends on the subset, and the subset structure that separates the signs is operating-condition
heterogeneity, not fault-mode count: FD002 and FD004 are multi-condition, FD001 and FD003 single-condition,
and the fold-level split is 9/10 versus 2/10. We state this as an observed association; the experiment
does not identify the mechanism, and we do not claim it does.

The pre-registered diagnostic (Section 4.4) closes the loop in a way that is useful precisely because it
fails. The obvious way to justify a conditional scale is to check, on calibration engines, that it ranks
them by difficulty. That check is not informative here: its correlation with the outcome is −0.039, and
its own sampling standard deviation at six calibration engines is 0.41–0.49. The practical lesson is
narrow: for this statistic ($\rho_{cal}$), at the calibration sizes of this five-fold C-MAPSS protocol
(6–16 engines), the estimate is too noisy to support a go/no-go decision, even though the underlying
idea is reasonable.

## 5.2 Implications for reporting practice

Three reporting choices would have changed the picture in this paper, and each is cheap:

1. **Report the per-engine coverage distribution, not only the mean.** The marginal number is 0.895;
   the 1st percentile of per-engine coverage is 0.41–0.54 depending on subset, and the minimum is 0.12.
   <!-- C18 -->
2. **Report at least two training seeds.** With one seed, a 3.2 pp difference is inside the sampling
   noise of the scale experiment; with four seeds on a shared partition, the sign split is consistent
   across folds (9/10 versus 2/10), and two independent MC-Dropout draws change the sign in 0 of 80
   cells, so the comparison itself is reproducible even though the seed spread is large.
3. **Use a same-centre control when the scale arm also changes the centre.** Otherwise the comparison
   attributes a centre shift to the scale.

None of these requires new theory or additional data collection.

## 5.3 Relation to prior audits

Our per-engine finding is consistent with the conditional-collapse results of [@domingues2026] and the
band-failure analysis of [@yan2026audit]; we cite them as the source of the phenomenon and do not claim
to have discovered it. What we add is the controlled scale comparison and the replication structure.
Relative to [@zhou2026cpa], the object differs: CPA estimates instance-level coverage from a labeled
evaluation set, whereas we ask whether a calibration-side diagnostic is estimable at all before
deployment. Relative to [@moussa2026driftguard], the claim is weaker by design: we offer no detectability
bound, only a measured estimation variance for one named diagnostic under one frozen protocol.

# 6. Limitations

- **Dependence.** The 80 cells share one five-fold partition, and the 20 folds share training data
  through cross-validation overlap. All intervals reported here are descriptive; no independent-sample
  significance test is claimed, including the Fisher exact test in Section 4.3, which quantifies the
  separation within this design only.
- **Single backbone, four subsets.** The predictor is one IDA-PE configuration; the conclusions about
  the scale's heterogeneity are established for this configuration and these four subsets, not for
  conformal RUL in general.
- **Cross-domain evidence is limited.** The FEMTO comparison covers six run-to-full-failure bearings and
  supports only the efficiency claim, not the scale or unit-level findings.
- **Official test data are unused.** This is a protocol choice: the question concerns unseen engines
  from the training population. It also means the numbers are not comparable to papers that report
  official-test scores.
- **The diagnostic's variance is measured at three calibration sizes** (6, 15, 16). We do not extrapolate
  a sample-size requirement from them.
- **No deployment claim.** Nothing here shows that a failing engine can be identified before its labels
  arrive; Section 4.4 is evidence against one specific attempt, not a general impossibility result.

# 7. Conclusion

A frozen CQR pipeline for C-MAPSS RUL delivers a reproducible efficiency gain over a constant-width
conformal baseline — 20/20 folds, −10.22% Winkler at matched marginal coverage — and at the same time
leaves some engines covered as little as 12% of the time, a failure that is stable across training
seeds. Replacing the learned half-width with an MC-Dropout scale does not fix this uniformly: its effect
changes sign with operating-condition structure, and the natural calibration-side diagnostic for
deciding when to use it does not survive a pre-registered test, because at six calibration engines the
diagnostic is noisier than the signal it is meant to carry. The practical recommendation is
measurement-oriented: report per-engine coverage, replicate across seeds, and control the centre when
evaluating a scale.

---

---

# Acknowledgment

The authors thank the maintainers of the C-MAPSS and FEMTO/PRONOSTIA datasets.
[Funding and conflict-of-interest statements to be completed before submission.]

# Biographies

[Author biographies to be completed before submission. The IJPHM template places biographies after the references and before the appendix.]
