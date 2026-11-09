# Appendix A. Coverage-aligned comparison (post-hoc diagnostic)

> This appendix is deliberately separated from the main results. The aligned comparison interpolates
> each method's coverage–width frontier using test labels; it therefore cannot represent a deployable
> selection rule and is not used to support any claim in Sections 4–7.

For completeness we report the comparison at matched realised coverage (target 0.90, Winkler evaluated
at a fixed $\alpha=0.1$). CQR improves on the constant-width baseline in 75/80 cells and 19/20 folds,
with a cell-level mean change of −9.09% and a fold-level median of −9.69%. The per-subset means are −7.57%, −14.07%, −6.49%
and −8.22%. <!-- C14 --> <!-- C21 -->

The same diagnostic applied to the MC-Dropout arm gives 31/80 cells, +18.9% mean and +4.5% median —
the opposite sign to the nominal-level result of Section 4.3. The reversal is expected: an arm that
under-covers at the nominal level is widened to reach 0.90, and widening is penalised. We report
it so that the two protocols are not confused, and we draw no conclusion from it. <!-- C15 -->
