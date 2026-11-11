# Paper B method draft (protocol + model)

This is a draft of the method/evaluation section. It contains no results.

## 1. Problem

Given a truncated multivariate sensor trajectory
$X_{1:t}\in\mathbb{R}^{t\times 14}$ of an engine, predict the remaining
useful life $y_t = \min(T-t, 125)$, where $T$ is the failure time. Training
engines are complete run-to-failure trajectories; test engines are truncated at
an unknown point.

## 2. Data protocol

- **Split.** The official training engines are split 80/20 by engine id with a
  fixed permutation seed. All models and all ablations share the split. No
  window of a validation engine appears in training.
- **Preprocessing.** Operating conditions are clustered with KMeans
  ($k=6$) on the three operating settings; each sensor is z-scored within its
  cluster. The clusterer and the statistics are fitted on training engines
  only. The same transformation is applied to validation and test.
- **Windows.** A window of length $L=70$ ends at cycle $t$ and is left-padded
  with a binary mask. Training uses every cycle of every training engine.
- **Baseline reference.** For each window, the reference is a shrinkage
  estimate $\bar x = \frac{n}{n+2}\bar x_{\text{early}} +
  \frac{2}{n+2}\bar x_{\text{pop}}$, where $\bar x_{\text{early}}$ is the
  mean of up to five early-life frames ($\le 30$ cycles) in the same operating
  cluster and $\bar x_{\text{pop}}$ is the training-population mean for that
  cluster. It uses only past frames of the same engine and training statistics.
- **Validation.** Each validation engine is evaluated at five truncations,
  at 40/55/70/85/95% of its life. The target at a cut $c$ is $T-c$. Model
  selection uses the RMSE over these engine-level points, not the RMSE over all
  sliding windows.
- **Test.** The official test engine is truncated at its last observed cycle;
  the target is the value in `RUL_FDxxx.txt`, capped at 125. Test files are
  opened only by the frozen evaluator.

## 3. Models

### 3.1 DSPT

For each sensor $n$ and step $t$, the input representation is
$\phi(x_{t,n}) = [x_{t,n}, x_{t,n}-b_n, e_{t,n}]$, where $b_n$ is the baseline
reference and $e_{t,n}$ is a causal EMA with $e_{t,n} = \alpha e_{t-1,n} +
(1-\alpha)x_{t,n}$ ($\alpha=0.5$), initialised at the first valid frame.

A shared linear map embeds $\phi$ to $d=64$. A depthwise temporal convolution
(or a multi-scale bank $k\in\{3,5,7\}$ with a projection) produces a residual
update. A per-sensor Transformer encoder (2 layers, 4 heads, padding mask)
models temporal context. An optional cross-sensor Transformer layer mixes the
14 sensors at each step. A learned softmax pools sensors; a Fourier encoding of
the absolute cycle is added. Two nested suffix pools give $h_A$ (last 50 valid
steps) and $h_B$ (all valid steps), and a length-dependent gate
$g=\mathrm{clip}((m-50)/50,0,1)$ gives $h_f=g h_B+(1-g)h_A$. The RUL head is
shared: $\hat y_A=f(h_A)$, $\hat y_B=f(h_B)$, $\hat y=f(h_f)$.

The loss is
$$\mathcal L = \mathrm{MSE}(\hat y,y) +
0.5\,\mathbb E[\,w(e)\,e^2\,] +
\frac{0.25}{2}\left(\mathrm{MSE}(\hat y_A,y)+\mathrm{MSE}(\hat y_B,y)\right)
+ \lambda\,\mathrm{MSE}(\hat h,h^\star)
+ \mu\,\mathcal M,$$
where $e=\hat y-y$, $w(e)=1.8$ if $e>0$ and $1$ otherwise, $h^\star$ is the
PCA-1 health index (training engines only), and $\mathcal M$ is an optional
monotone penalty over adjacent **valid** pairs.

### 3.2 Vanilla Transformer baseline

A standard encoder: linear embedding of the 14 normalised sensors, fixed
sinusoidal positional encoding, two pre-norm Transformer layers with a padding
mask, masked mean pooling, and a two-layer RUL head. It has no convolution, no
cross-sensor attention, no learned sensor pooling, no dual-span gate, no
absolute cycle, and no HI. Two feature-matched controls add (i) the absolute
cycle encoding and (ii) the same three-channel input representation, so the
architecture and feature effects can be separated.

## 4. Ablation definitions

| name | MS | SA | HI |
|---|:--:|:--:|:--:|
| Base | off (single k=3 conv) | off (learned sensor pooling remains) | off |
| MS | k=3/5/7 + projection | off | off |
| SA | off | one cross-sensor layer | off |
| HI | off | off | $h_f$, correct monotone |
| MS+SA, MS+HI, SA+HI | pairs | | |
| FULL | on | on | on |

MS mechanism controls: `single_proj` (one kernel + projection),
`multi_noproj` (three scales, no projection), `multi3` (three k=3 kernels +
projection, parameter-matched to full).
SA mechanism controls: sensor-identity embedding; operating-condition
embedding.
HI mechanism controls: $h_B$ source; no monotone; inverted monotone sign.

## 5. Metrics

RMSE, MAE, NASA scoring function (sum), bias, fraction of late predictions,
fraction with error > 10 cycles. Means and standard deviations over three model
seeds. The test set is not used for any choice.
