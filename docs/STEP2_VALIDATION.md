# Step 2 Validation Record (P0 updated)

Date: 2026-10-20; P0 update 2026-10-21

## Deliverables

- src/model.py: STG-MambaNet with identity-prior graph, SSM/LSTM, trajectory
- src/loss.py: PhysicsGuidedLoss with asymmetric and monotonic terms
- tests/test_model.py: model, graph, trajectory, monotonic, and loss tests
- tests/test_dataset.py: clip125 and raw-appendix tests

## Architecture

- Adaptive spatial branch:
  A = (1 - alpha) I + alpha * Softmax(ReLU(E1 E2^T)), alpha init 0.1.
- graph_mode: learnable / identity / random_fixed.
- Multi-scale selective state-space temporal branch in pure PyTorch.
- Optional parameter-matched LSTM control (hidden 70, 142,806 params vs
  Full 144,265, 1.01% difference).
- Cross-attention + branch/channel gating.
- return_trajectory=True returns a dense per-cycle trajectory; the final
  trajectory step is exactly the terminal prediction.

## Loss

~~~text
L = mse_weight * MSE
    + asymmetric_weight * mean(w * d^2)
    + lambda_mono * mean(max(0, r_t - r_{t-1}))

w = late_weight (2.0) if d > 0 else early_weight (1.0)
~~~

The monotonic term is zero when no trajectory is supplied; the component
records monotonic_available=0.

## Verification

- python -m unittest discover -s tests -v: 34 tests collected, OK;
  6 dataset tests are skipped after the official test files were sealed.
- CUDA forward/backward and real FD001 batch: passed
- Graph row sums: 1.0 (atol 1e-6)
- E1/E2 gradients: finite and non-zero for learnable graph
- Identity graph export: exactly I
- Random-fixed graph: row-stochastic
- Parameter matching: 1.01% difference
- Clip125 and raw appendix dataset paths: tested

## Artifact hashes (SHA256, first 16)

~~~text
B5EF67DF7C440FDF  src/model.py
94A3B8E293CD925E  src/loss.py
F7D84E4E10DEE333  src/dataset.py
8C61EC632BEB0F67  scripts/run_ablation.py
4BBB919F2C04A2A0  scripts/summarize_ablations.py
~~~

## Caveat

These are implementation and integration checks only. The P3 five-seed
confirmatory result is in docs/STAGE3_REPORT.md; Full is not the best model.
Do not convert this validation record into a performance claim.
