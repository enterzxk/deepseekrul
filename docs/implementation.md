# PU-Net 实现指南（implementation.md）

> 生成时间：2026-11-07 | 阶段 D 产出 | 对应 idea_report.md Part 2 §3 与 Part 3 §2
> 路径：A（基于现有仓库）｜落位：新建 `src/punet/` 子包，**不改动** `src/` 下任何现有文件

---

## 0 总览

### 0.1 目标

在现有 IDA-PE 代码之上实现 PU-Net：冻结主干，新增尺度头与偏置头，训练这两个头，再用分块保形校准产出区间，最后做引擎级覆盖诊断。

### 0.2 硬约束

1. **零侵入**：不修改 `src/model.py`、`src/loss.py`、`src/dataset.py`、`src/lifetime_prior/` 等现有文件。
2. **主干冻结**：训练时 `requires_grad=False`，只更新 `src/punet/` 内的参数。
3. **因果性**：先验特征只允许使用 `StaircasePrior` 提供的、截断点 `c <= t` 的取值，禁止插值。
4. **数据隔离**：调参块、校准块、测试集三者互不重叠，切分函数必须显式断言。
5. **可归因**：偏置头最后一层零初始化，训练起点处 `b ≡ 0`。

### 0.3 目录结构

```text
src/punet/
  __init__.py
  features.py      # 先验特征对齐（包装 StaircasePrior）
  heads.py         # ScaleHead / BiasHead
  model.py         # PUNetModel：主干 + 两个头
  conformal.py     # BlockConformalCalibrator
  metrics.py       # PICP / PINAW / Winkler
  diagnostics.py   # 引擎级覆盖诊断 + 块自助
  train.py         # 只训练两个头的训练循环
  evaluate.py      # 评估入口
scripts/
  run_punet.py     # CLI：train / calibrate / evaluate / diagnose
tests/
  test_punet_heads.py
  test_punet_conformal.py
  test_punet_diagnostics.py
```

---

## 1 文件与函数设计

### 1.1 `src/punet/features.py`

```python
@dataclass(frozen=True)
class PriorFeatures:
    mu: torch.Tensor      # (B,) 先验均值 mu_T
    sigma: torch.Tensor   # (B,) 先验标准差 sigma_T
    tau: torch.Tensor     # (B,) 退化时钟 t / mu_T
    valid: torch.Tensor   # (B,) bool，False 表示该样本无可用截断（回退到中位数）

class PriorFeatureBuilder:
    def __init__(self, prior: StaircasePrior, subset: str, sigma_floor: float = 1e-3) -> None: ...
    def build(self, units: Sequence[int], cutoffs: Sequence[int]) -> PriorFeatures: ...
```

**逻辑**：对每个 `(unit, cutoff)` 调用 `StaircasePrior` 查表；缺失时用 `medians` 回退并把 `valid` 置 False。`tau = cutoff / mu`，`mu` 为 0 或 NaN 时用 `sigma_floor` 兜底。

**边界**：`cutoff > engine_length` 必须抛 `ValueError`（对应 T4 断言）。

### 1.2 `src/punet/heads.py`

```python
class ScaleHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64, eps: float = 1e-3) -> None: ...
    def forward(self, h: torch.Tensor, sigma: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """h: (B, D), sigma: (B,), tau: (B,) -> w: (B,) 且 w > 0"""

class BiasHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64) -> None: ...
    def forward(self, h: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """-> b: (B,)，最后一层零初始化，故初始 b ≡ 0"""
```

**ScaleHead 实现要点**

```python
a = F.softplus(self.a_raw)                     # > 0，保证单调
mod = torch.exp(self.mlp(torch.cat([h, tau.unsqueeze(-1)], dim=-1)))   # (B,)
w = (sigma.clamp_min(self.eps) + self.eps) ** a * mod
return w.clamp_min(self.eps)
```

- `a_raw` 初始化为 `softplus^{-1}(1.0) ≈ 0.5413`，使初始 `a ≈ 1`。
- `mlp` 输出层权重初始化为 0、偏置为 0，使初始 `mod ≡ 1`，即初始 `w = (sigma+eps)^1`。
- 该初始化让"训练前"的尺度就是纯幂律，便于与消融 A5 对照。

**BiasHead 实现要点**：两层 MLP，最后一层 `weight.zero_(); bias.zero_()`。

### 1.3 `src/punet/model.py`

```python
class PUNetModel(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, use_scale_head: bool = True,
                 use_bias_head: bool = True, use_sigma: bool = True) -> None: ...
    def train(self, mode: bool = True) -> "PUNetModel": ...   # 覆盖：backbone 恒为 eval
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None,
                feats: PriorFeatures) -> PUNetOutput: ...

@dataclass
class PUNetOutput:
    y_hat: torch.Tensor      # (B,) 修正后的点预测
    y_base: torch.Tensor     # (B,) 主干原点预测
    w: torch.Tensor          # (B,) 尺度
    h: torch.Tensor          # (B, D) 表征
```

**逻辑**：`h, y_base = backbone.encode(x)`（若主干无 `encode`，则在 `forward` 内用 hook 取倒数第二层输出，见 1.3.1）；`y_hat = y_base + b`；`w` 由 `ScaleHead` 给出。

**1.3.1 表征提取**：优先检查主干是否已有返回表征的接口；若无，用 `torch.nn.Module.register_forward_hook` 在指定层截获，hook 句柄在 `__init__` 中注册并在 `close()` 中移除。**不得**修改主干源码。

**消融开关**：`use_sigma=False` 时把 `sigma` 置为常数 1 后传入两个头（对应 A2）；`use_scale_head=False` 时 `w ≡ 1`（A1）；`use_bias_head=False` 时 `b ≡ 0`（A6）。

### 1.4 `src/punet/conformal.py`

```python
@dataclass(frozen=True)
class BlockSplit:
    fit: np.ndarray        # 样本索引
    tune: np.ndarray
    calib: np.ndarray
    test: np.ndarray

def block_split(units: np.ndarray, cutoffs: np.ndarray, *, tune_frac: float,
                calib_frac: float, seed: int) -> BlockSplit:
    """按发动机聚类 + 相邻时间块切分；返回前断言三者两两不交。"""

class BlockConformalCalibrator:
    def __init__(self, delta: float = 0.1) -> None: ...
    def fit(self, scores: np.ndarray) -> "BlockConformalCalibrator": ...
    @property
    def q_hat(self) -> float: ...
    def interval(self, y_hat: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
```

**分位点**：`q_hat = np.quantile(scores, min(1.0, (1 - delta) * (1 + 1 / n)), method="higher")`。

**切分**：先按发动机随机分组（不拆散同一台），再在组内按 cutoff 排序切成相邻时间块，保证 `fit` 的时间早于 `tune`、`tune` 早于 `calib`。

### 1.5 `src/punet/metrics.py`

```python
def picp(y, lo, hi) -> float: ...
def pinaw(y, lo, hi) -> float: ...
def winkler(y, lo, hi, alpha: float = 0.1) -> float: ...
def engine_coverage(units, y, lo, hi) -> dict[str, float]: ...
```

**定义**：`PICP = mean(lo <= y <= hi)`；`PINAW = mean(hi - lo) / (max(y) - min(y))`；`Winkler` 按标准式：宽度加越界惩罚 `2/alpha * (lo - y)_+ + 2/alpha * (y - hi)_+`。

### 1.6 `src/punet/diagnostics.py`

```python
def engine_bootstrap_ci(units, covered, n_boot: int = 5000, seed: int = 42,
                        alpha: float = 0.05) -> tuple[float, float]:
    """按发动机重采样，返回覆盖率的百分位置信区间（与 FINAL_PROTOCOL 的引擎聚类自助一致）。"""

def stratified_coverage(units, cutoffs, mu, y, lo, hi, n_bins: int = 5) -> pd.DataFrame:
    """按退化阶段（cutoff / mu 分位）与先验失配（|mu - y| 分位）分箱，返回每箱覆盖率。"""
```

### 1.7 `src/punet/train.py`

```python
def train_heads(model: PUNetModel, loader, *, epochs: int, lr: float,
                lam: float, gamma: float, device: str) -> dict[str, float]:
    """只优化 model.parameters() 中属于两个头的参数；主干 requires_grad=False。"""
```

**损失**：`L = point_loss(y_hat, y) + lam * pinball(s, q_lo, q_hi) + gamma * ||psi||^2`，其中 `s = |y - y_hat| / w`。

**断言**：训练前检查 `sum(p.requires_grad for p in backbone.parameters()) == 0`。

### 1.8 `scripts/run_punet.py`

```text
python scripts/run_punet.py --subset FD001 --stage train     --epochs 30 --lam 0.5
python scripts/run_punet.py --subset FD001 --stage calibrate --delta 0.1
python scripts/run_punet.py --subset FD001 --stage evaluate  --seeds 1 2 3 4 5
python scripts/run_punet.py --subset FD001 --stage diagnose  --n-boot 5000
python scripts/run_punet.py --subset FD001 --stage train --ablate A4   # 随机置换 sigma
```

**输出**：`results/punet/{subset}/{stage}_{timestamp}.json`，字段含 `q_hat`、`picp`、`pinaw`、`winkler`、`engine_cov_min`、`rmse`、`nasa`。

---

## 2 与现有代码的接入点

| 接入对象 | 现有接口 | 用法 |
|---|---|---|
| 数据 | `prepare_cmapss_datasets(data_dir, subset, window_size=30, stride=1, max_rul=125)` → `CMAPSSDataBundle` | 直接复用，不改 |
| 主干 | `STG_MambaNet`（`forward(x, adjacency)`，`x: (B,T,N)`） | 冻结加载；表征用 hook 截获 |
| 先验 | `StaircasePrior`（`mu_T(t)`、`sigma_T(t)`，禁止插值） | 由 `PriorFeatureBuilder` 包装 |
| 损失 | `PhysicsGuidedLoss` / `risk_metrics` | 点损失直接复用，不改 |

> 需要确认的接口细节（编码第一步先验证）：主干是否已有返回表征的公开方法；`StaircasePrior` 的查表方法名与返回字段；`CMAPSSDataset` 是否保留 `unit` 与 `cutoff` 元信息（诊断必需）。若缺元信息，在 `src/punet/features.py` 内重建索引，**不改** `dataset.py`。

---

## 3 测试计划

| 测试 | 内容 | 通过标准 |
|---|---|---|
| `test_scale_head_monotone` | 固定 `h, tau`，令 `sigma` 递增 | `w` 严格递增 |
| `test_bias_head_zero_init` | 初始状态下前向 | `b` 全为 0（`atol=1e-8`） |
| `test_backbone_frozen` | 训练一步后 | 主干参数逐位不变 |
| `test_conformal_coverage` | 合成高斯数据，`w ≡ 1` | 覆盖率 ∈ [0.88, 0.92]（`delta=0.1`） |
| `test_split_disjoint` | 随机 units/cutoffs | 四块两两不交，且时间顺序成立 |
| `test_engine_bootstrap` | 已知覆盖率 | CI 包含真实值 |

---

## 4 校验结果

**✅ 实验要求覆盖**
- E1 主实验 → `run_punet.py` 的 train/calibrate/evaluate + `metrics.py`
- E2 消融 A1/A2/A4/A5/A6 → `PUNetModel` 的 `use_scale_head`/`use_sigma`/`use_bias_head` 开关 + `--ablate`
- E3 附加 B1–B4 → `diagnostics.py` 的 `engine_bootstrap_ci` 与 `stratified_coverage`

**✅ 逻辑一致性**
- `x: (B,T,N)` → 主干 → `h: (B,D)`、`y_base: (B,)`
- `PriorFeatures` 三个标量均为 `(B,)`，与 `h` 在 `(B,)` 维度对齐
- `w: (B,)` → `s = |y - y_hat| / w` → 标量分位点 → `interval` 输出 `(B,)` 上下界

**✅ 完整性**
- `features.py`、`heads.py`、`model.py`、`conformal.py`、`metrics.py`、`diagnostics.py`、`train.py`、`evaluate.py`、`run_punet.py` 均有对应章节

---

## 5 编码前确认清单（待用户确认）

| 项 | 值 |
|---|---|
| 框架 | PyTorch（现有仓库已用） |
| 运行环境 | 复用仓库现有环境（conda/venv 名称待确认） |
| 设备 | 单卡（待确认是否有 CUDA） |
| 数据集 | 已就绪 `data/CMAPSS/` |
| 运行策略 | 快速阶段 agent 跑，全量 5 种子×4 子集由用户或后台任务跑 |
| README 位置 | `code/` 不存在，写入仓库根 `README_PUNET.md`（待确认） |
