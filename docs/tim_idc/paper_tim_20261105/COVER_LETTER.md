# Cover letter (draft)

**To:** The Editor-in-Chief, *IEEE Transactions on Instrumentation and Measurement*

**Re:** Submission of "When Does a Causal Lifetime Prior Help Remaining Useful Life Prediction? A Budget-Controlled Study on C-MAPSS"

Dear Editor,

Please consider the enclosed manuscript for publication in *IEEE Transactions on Instrumentation and Measurement*.

**What the paper asks.** Lifetime priors estimated from sensor histories are an appealing way to inject engine-specific information into remaining-useful-life (RUL) predictors, but their reported benefit is often measured under a training budget that is not controlled across the compared arms. The manuscript asks a measurement question: when does a causal lifetime prior help RUL prediction, and when does it not?

**What we find.** We build a strictly causal, nested-cross-fitted lifetime prior and inject it into a positional-encoding Transformer on the four C-MAPSS subsets. Under a lightweight budget, the sensor prior improves over the PE baseline on three of four subsets; under a full budget, the RMSE improvement is significant only on FD003, and on the multi-condition FD002 the sensor prior is significantly worse than an age/condition-only prior. The asymmetric NASA score improves significantly on FD003 and FD004, however, while MAE does not; the prior shifts predictions earlier, trading accuracy for safety. A fixed-age measurement analysis shows that sensor change and slope features carry substantial lifetime information (8/8 engine-paired CIs), so the negative end-to-end result is not explained by uninformative sensors. The one-shot official test is reported for both budgets; no test-driven tuning is performed.

**Why TIM.** The paper treats the lifetime prior as a measurement instrument rather than as another architectural module. The fixed-age analysis separates level, change, and full sensor information; the budget-controlled protocol separates a controlled contrast from a performance table; and the one-shot official test is sealed and opened once. We believe this measurement-oriented framing, and the negative result it produces, fit the journal's interest in instrumentation, measurement methodology, and prognostics.

**Novelty and honesty.** The contribution is a causal, leakage-controlled lifetime prior, a sensor-information measurement analysis, and a budget-controlled evaluation that explains why a prior can look strong in one regime and weak in another. The manuscript does not claim state-of-the-art RUL accuracy, calibrated uncertainty, or a new architecture; the negative result is the point.

**Compliance.** The manuscript is original, is not under review elsewhere, and all authors have approved the submission. C-MAPSS is public; the code, protocols, sealed predictions, and hashes are archived. The official test labels were opened exactly once, and no model, recipe, arm, or budget was changed afterward.

We would be glad to suggest reviewers with expertise in prognostics, C-MAPSS benchmarking, and measurement uncertainty if that would help.

Sincerely,

[Corresponding author name]  
[Affiliation]  
[Email]
