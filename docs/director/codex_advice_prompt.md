You are advising a research engineer on an IEEE TIM submission about C-MAPSS RUL prediction. Be concrete and critical. Do NOT write code or files; just give a prioritized plan.

CURRENT EVIDENCE (all validation-only, official test sealed except FD001/FD002 which are open):
- Protocol: 80/20 engine split (split_seed=7), RUL cap 125, window 50, last-window evaluation.
- Strong Transformer baseline (144k params): FD001 val 12.34, FD002 val 13.97, FD002 matched-cutoff 16.99.
- Best candidate so far (ORDER_03, FD002, 1 seed): "M4 RACDR" = condition-matched shrinkage healthy reference (KMeans K=6 on op settings; x_ref = n/(n+tau)*xbar_engine_cluster + tau/(n+tau)*mu_population_cluster, tau=2) + reliability gate g=sigmoid(MLP(6 evidence)+b_sensor) in (0,1)^14 + zero-init gated sensor-ID + asymmetric late loss 1.8. FD002: val 12.54, matched 14.96, Protocol C 9.99.
- Ablation (1 seed each): M0 raw TF 13.97/16.99/11.24; M2 population-ref only 13.77/15.59/10.90; M3 cond-shrink ref without gate 13.39/16.41/10.98. So the gain appears to come from the reliability gate + gated identity, NOT from the condition reference itself; M3 is worse than M2 on matched-cutoff.
- FD001 earlier: Transformer+features (raw + baseline residual + EMA + cycle) reached val 10.94, Protocol B 10.21, Protocol C 10.53; late_fraction 0.69 (too high). Safety gate requires late_fraction <= 0.52.
- Known issues: reliability evidence e4 uses a sensor-space deviation proxy instead of the specified op-space distance to the cluster centre; conditional reference not yet validated at 3 seeds; S5 (Transformer + identity) not implemented.

QUESTIONS:
1. Given that the reliability gate (not the reference) seems to carry the gain, is the "condition-referenced reliability" story still defensible for TIM, or should we re-frame? What minimal extra evidence would make it defensible?
2. What are the 3-5 highest-value experiments to run next (each <= 60 min on one A40) to get a publishable, protocol-comparable result?
3. How should we fairly compare against published numbers given our stricter 80/20 split rather than training on 100% of engines? Should we add a 100%-train variant for the final table?
4. What failure modes / reviewer objections should we pre-empt (especially the late_fraction 0.69 problem and the fact that FD001 favours a plain Transformer with features)?
5. Concrete recommendation on what the single "final method" should be.