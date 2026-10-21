# FD001 Ablation Report (3 seeds)

Date: 2026-10-20

## Scope

Twelve runs: four variants x three seeds (42, 67, 123). Each run used an
engine-level train/validation split, training-only Min-Max fitting, validation
RMSE checkpoint selection, and one final evaluation on the official FD001 test
set.

## Observed results

| Variant | RMSE (mean +/- std) | NASA Score (mean +/- std) | MAE (mean +/- std) | Params | Seeds |
|---|---:|---:|---:|---:|---:|
| w/o A-GCN | 15.048 +/- 0.308 | 347.370 +/- 41.614 | 11.423 +/- 0.376 | 143,816 | 42, 67, 123 |
| w/o SSM-Block | 14.756 +/- 0.304 | 383.466 +/- 60.829 | 11.007 +/- 0.056 | 126,981 | 42, 67, 123 |
| w/o Physics Loss | 15.308 +/- 0.442 | 450.918 +/- 101.568 | 11.248 +/- 0.254 | 144,264 | 42, 67, 123 |
| Full Model (Ours) | 14.878 +/- 0.712 | 379.771 +/- 81.421 | 11.107 +/- 0.790 | 144,264 | 42, 67, 123 |

Full-model test RMSE range: 14.072--15.422. It did not reach the target
11.5--12.5 range.

## Paired differences relative to Full

Positive RMSE difference means the ablated variant is worse. Positive NASA
difference means the ablated variant is worse.

| Ablation | Mean delta RMSE | Mean delta NASA | Seed-level direction |
|---|---:|---:|---|
| w/o A-GCN | +0.170 | -32.401 | RMSE mixed; NASA favors no-graph on average |
| w/o SSM-Block | -0.122 | +3.691 | LSTM slightly better RMSE; Full slightly better NASA |
| w/o Physics Loss | +0.430 | +71.147 | Physics loss worse in all three seeds on NASA |

## Claim audit

### A-GCN

The claim that removing A-GCN clearly worsens RMSE is not supported. Full has a
slightly lower mean RMSE, but w/o A-GCN has a lower mean NASA score. The
direction also changes across seeds. The current evidence is mixed.

### SSM-Block

The claim that removing the SSM block worsens RMSE is not supported. The LSTM
control has a slightly lower mean RMSE and lower RMSE standard deviation.
Full has a small NASA advantage. This does not establish SSM superiority.

### Physics loss

This is the only module with a consistent direction: removing the physics loss
worsens NASA score in all three seeds, with mean RMSE +0.430 and mean NASA
+71.147. The effect is meaningful but not the claimed "large" collapse, and the
sample size is only three seeds.

### Full model best overall

This is false in the observed experiments. The LSTM control has the lowest mean
RMSE; w/o A-GCN has the lowest mean NASA score. Full is not uniformly best.

## Scientific decision

These results must not be rewritten to match the intended story. The current
model is not ready for a paper table claiming all modules are beneficial.

The next development cycle should:

1. Keep the official test set sealed for final reporting.
2. Improve the base model using only validation-engine evidence.
3. Reassess whether A-GCN and the SSM block should remain core contributions.
4. Tune the physics-loss strength only on validation data.
5. After freezing the design, run fresh seeds and evaluate the official test
   once for the final table.

## Limitations

- Only FD001 and three seeds.
- Model-selection variance is substantial.
- No statistical significance claim is made with n=3.
- The official test set has been observed in this development pilot. Future
  design decisions must not use these test scores.
