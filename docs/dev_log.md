# PU-Net 开发日志

> 阶段 E 维护 | 起始：2026-11-07

## 环境事实（已核实，非假设）

| 项 | 结论 |
|---|---|
| 本机（Windows, DSH） | Python 3.12，**无 torch**，无法本地训练 |
| 执行机 | `ssh a40` → Linux，`/home/yndx/miniconda3/envs/a40_env/bin/python`，torch 2.5.1+cu121，CUDA 可用 |
| 远端仓库现状 | `/home/yndx/bbwork/src` 仅有 baselines/dataset/loss/model/candidates，**缺少** `lifetime_prior/`、`ecs/`、`cet/`、`eac.py` |
| 数据 | 本机 `data/CMAPSS/` 完整；远端 `/home/yndx/bbwork/data/CMAPSS` 存在 |

## 进度

| 步骤 | 状态 | 证据 |
|---|---|---|
| E-1 编写 `src/punet/`（8 个模块）+ `scripts/run_punet.py` + 3 个测试文件 | ✅ | 文件已创建 |
| E-2 冒烟：numpy 模块本地跑 | ✅ | `Ran 8 tests ... OK` |
| E-3 冒烟：torch 模块远端跑 | ✅ | `Ran 6 tests ... OK`；`--stage smoke` 输出 q_hat=7.32 |
| E-4 接入真实主干与先验表 | ⬜ 待办 | 需要远端具备 `src/lifetime_prior/` 与先验表 |
| E-5 跑 A4（σ_T 置换）/ A5（相关性） | ⬜ 待办 | 用户已定：先证伪机制假设 |
| E-6 主实验 4 子集 × 5 种子 | ⬜ 待办 | 后台任务 |

## 决策记录

1. **零侵入**：所有新增代码在 `src/punet/`，未修改 `src/` 任何现有文件。
2. **懒加载 `__init__`**：初版在包初始化时 import torch，导致 numpy-only 测试无法在本机运行；改为 `__getattr__` 懒加载后 8 个测试通过。
3. **尺度头初始化**：`a_raw = softplus^{-1}(1)`、MLP 末层零初始化 → 训练起点 `w = (sigma+eps)`，与消融 A5 的幂律形式一致。
4. **偏置头零初始化**：已验证 `y_hat == y_base`（`atol=1e-6`），保证起点等价于 Gate 0。
5. **同步方式**：PowerShell 管道 tar 会破坏二进制，改用 `tar -czf` 落地后 `scp`。

## 已知风险

- 远端缺少 `lifetime_prior/`，无法直接跑真实先验；需同步仓库或指定已有路径。
- `PUNetModel` 依赖 `feature_fn` 或 `backbone.encode`；真实主干是否提供表征接口尚未核实（下一验证点）。

## 2026-11-07 追加：仓库内已有保形基线

发现 `scripts/dsh_conformal.py`（45 行）：**已实现常数尺度 split-conformal RUL 区间**，按 engine-level 折校准绝对残差分位数并检查覆盖率。另有 `dsh_cp_*.py` 一组相关脚本。

含义：
1. 实验大纲里的「常数尺度分块 CP」基线**不需要从零实现**，可直接复用该脚本的逻辑，只需换成相邻时间块切分。
2. 这也印证了 gap 的真实性：仓库内部的既有做法正是"常数尺度 + engine 折"，PU-Net 的增量就是「用 σ_T 做尺度 + 学习两个头」。
3. 论文中必须把它写清楚：要么作为内部 baseline 引用，要么在 Related Works 中说明其局限（常数尺度，不随先验不确定性变化）。

## 2026-11-07 重大发现：σ_T 恒为 0

**证据**：`experiments/ida/prior/FD001_prior_cutoffs.csv` 共 7145 行（train 5716 / val 1429），`sigma_T` 非零行数 = 0，最大值 2.8e-14（浮点噪声）。

**根因**：`nested_prior.py` 用「5 个 seed 各训一个 HistGradientBoostingRegressor，取预测 std」估计 σ_T。HGB 是确定性学习器，同数据下不同 random_state 给出逐位相同预测 → std 精确为 0。

**影响**：
1. A4（置换 σ）与 none 结果完全一致——没有 σ 可置换
2. 尺度头学到的 a≈0.98 但 σ≡0，实际退化为仅依赖 (h, τ) 的调制
3. PU-Net 的 PINAW 一度到 1.19e12（尺度发散），修好稳定化后仍劣于常数尺度（PINAW 1.62 vs 0.41）
4. RQ2 的前提在该数据上不可检验

**修复**：新增 `scripts/build_prior_uncertainty.py`，对训练发动机做 B=20 次有放回重采样重训 HGB，取预测 std 作为 σ_T；训练窗口用内层交叉拟合得到 OOF σ_T。零侵入：不修改 `nested_prior.py`。
冒烟（B=3, FD001）：val σ 中位数 6.1–7.1，均值 9.13，非零。

**结论**：这一发现本身构成论文的贡献 0（"既有先验交付的 σ 是 0"），已写入 Part 2。

## 2026-11-07 加权保形：两次尝试均失败（如实记录）

**动机**：发动机级异质性使边际覆盖不可迁移（校准发动机 mean|err|=7.48，测试 8.92，跨度 3.67–16.94）。

| 校准方式 | FD001 PICP/Winkler | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| 分层（regime×σ） | 0.914 / 39.2 | 0.816 / 57.7 | 0.876 / 44.8 | 0.911 / 55.5 |
| 加权 CP（协变量=预测 RUL） | 0.986 / 63.3 | 0.804 / 60.9 | — | — |
| 加权 CP（协变量=发动机级 σ） | 0.975 / 51.1 | 0.816 / 58.4 | 0.879 / 46.3 | 0.908 / 55.4 |

**结论**：两种协变量都无法把覆盖率拉回 0.90，Winkler 在 3/4 子集上劣于分层校准。
根因：发动机难度不由 σ 或寿命阶段决定，现有可观测量里没有"这台发动机有多难"的代理。

**处理**：分层校准保留为主方案；加权 CP 作为消融行如实报告（用户已确认要进主表）。
