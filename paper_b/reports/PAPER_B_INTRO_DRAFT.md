# Paper B introduction draft (no results)

Remaining useful life (RUL) prediction for turbofan engines on C-MAPSS is a
crowded benchmark. Recent models add multi-scale temporal convolutions,
cross-sensor attention, and auxiliary health-indicator (HI) supervision, and
typically report a single-seed improvement over a base model. The implicit
claim is architectural: the module, not the protocol, produces the gain.

That claim is hard to test from the literature because three protocol choices
vary together with the architecture: (i) whether early stopping uses every
sliding window or one truncated point per engine, (ii) whether preprocessing
and HI labels are fitted on training engines only, and (iii) how often the
official test set has been observed. A module can appear to help or hurt
depending on all three.

We take a single architecture, DSPT, that combines a three-channel input
representation (raw, baseline deviation, causal EMA), a multi-scale temporal
convolution, a per-sensor temporal Transformer, a cross-sensor Transformer, a
learned sensor pooling, a dual-span gate, and an optional HI auxiliary loss.
An earlier internal ablation on this architecture reported that the full model
was *worse* than its base on FD001 (12.70 vs 10.60 RMSE) and never won on all
four subsets. We first show that this table is not fabricated: the archived
checkpoints reproduce the archived predictions with correlation 0.9935-0.9995
under a reconstructed evaluator. We then show that the archived ranking is not
caused by the evaluation bugs alone: fixing the padding, EMA initialisation and
HI masking moves the absolute numbers but leaves the ranking intact.

The remaining candidate is the training and model-selection protocol. We
therefore re-run the ablation with a pre-registered protocol: an 80/20
engine-level split, five truncated validation points per engine, three model
seeds, train-only preprocessing and HI fitting, and a frozen test set that is
opened once. Under this protocol we ask four questions. Does the full model
beat its base? Is the multi-scale convolution's gain due to scale, projection,
or capacity? Does cross-sensor attention fail because it is condition-agnostic?
Is the HI loss harmed by a noisy target or by a representation mismatch?

Our mechanism probes motivate the controls. Cross-condition sensor
correlations change 10-19x more on the multi-condition subsets (FD002/FD004)
than on the single-condition subsets, and individual sensor pairs change by up
to 0.56. The HI target is a decreasing health index with a strong global trend
(Spearman -0.79 to -0.92) but ~48% of adjacent steps move against that trend,
and it is largely redundant with RUL (R^2 0.61-0.78). The degradation signal
has a ~25-cycle correlation length, so the k=3/5/7 kernels all see nearly the
same low-frequency content.

The contribution is therefore not a new block. It is a controlled answer to
whether three popular blocks help, with the protocol, the reproduction, and
the mechanism made explicit. `[RESULTS AND CONTRIBUTION SENTENCE TO BE FILLED]`
