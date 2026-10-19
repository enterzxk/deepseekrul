# 固定起点、仅容量历史的电池长程 SOH 轨迹外推：多源实证、最小层级基线与我方结论

**日期：** 2026-10-19  
**执行人：** DeepSeek 独立研究线（非 Codex）  
**服务器目录：** `/home/yndx/zxk/deepseek-reaserch`（保留用户拼写 `reaserch`）  
**并行隔离：** 未读写 `/home/yndx/zxk/rul` 的任何文件、checkpoint、调度器、owner 或 test；未杀进程；未占用 GPU。  
**结果性质：** 本文严格区分【作者原文报告】、【本项目实测】、【推论】。没有把计划、目录或单元测试当作实验结果。

---

## 0. 执行摘要

### 0.1 一句话结论

在**固定预测起点、只使用容量/SOH 历史**的长程外推设定下，一个**一参数层级基线**——先取同化学体系/温度/放电倍率的“协议均值轨迹”，再加上该电芯在预测起点相对协议均值的当前偏移——在 101 颗官方电芯、5 个来源、5 折按电芯隔离的评估中，**总体/多数电芯**的长程 SOH MAE 和 EOL 误差优于线性/指数/幂律外推、协议均值本身、以及一个容量历史的 GRU。它在 51/88 颗电芯上优于 GRU（h=300），在 68/88 颗电芯上优于幂律外推；Oxford 上 GRU 略好（0.0051 vs 0.0077）。

这说明：**容量-only 长程预测的主要可解释信号不是复杂的短期时序模式，而是“协议均值形状 + 电芯级近常数偏移”**。真正未解决的瓶颈是**跨协议/跨来源的时间尺度外推**和**少数电芯的形状突变**，不是再堆一个 Transformer 模块。

### 0.2 三条核心证据

1. **短程好、长程坏，且模型偏差方向不同。** 在共同评估集（88 颗电芯、58,080 个 cell-origin-horizon 窗口）上，h=300 的 cell-equal SOH MAE 分别为：persistence 0.0785（系统性乐观，bias +0.0333）、linear100 0.0552、exp100 0.0424、power 0.0477、协议均值 0.0352、GRU 0.0262、**HOF 0.0224**。短程 h=50 时 HOF 为 0.0076，GRU 为 0.0128；HOF 在所有 horizon 的绝对 cell-equal MAE 最低（但相对增长并非最小）。
2. **协议均值+偏移显著改善 EOL。** 在 157 个有预测穿越的 cell-origin 案例上，HOF 的 EOL MAE=165.3 循环、median=48、within-20%=75.8%；协议均值 MAE=290.1、median=95、within-20%=53.1%；线性/指数/幂律分别为 422.6/358.7/285.6。若把“10,000 循环内未预测到穿越”的 37 个案例计为失败，HOF 的 within-20% 为 61.3%（119/194）。
3. **跨来源会崩。** 留一来源（leave-one-source-out）时，HOF 的 h=300 cell-equal MAE 从同协议 5 折的 0.0224 恶化到 **0.0820**；HNEI 0.1535、CALCE 0.0808、SNL 0.0728。EOL 更严重：全局回退下 194/194 个 EOL 案例在 10,000 循环内没有预测到穿越。**没有协议先验时，当前 HOF 不能作为跨来源通用方法。**

### 0.3 论文路线判断

- **可以独立成稿的方向：** 多源 fixed-origin 容量-only 基准 + “协议均值/电芯偏移/时间尺度”可预测性分解 + 最小层级基线 HOF + 校准区间。它的贡献是**诊断与基准**，不是“首次”或“SOTA”。
- **不建议现在做的方向：** 把“SOH 轨迹 + knee 检测 + 区间”拼成一个复杂模型；2024 曲率 knee、2025 多时间尺度、2025 transferable physics-informed、2026 multi-fidelity 已经覆盖了 knee/物理先验/不确定性的大量组合，直接拼接没有新颖性。
- **最接近的实质冲突：** BatteryMFormer（KDD 2026）是早期电压/电流到全寿命轨迹的多层 Transformer；本文是固定起点、仅容量历史、跨来源的 capacity-only 设定。两者任务不同，但**数据源可能重叠**；且 BatteryMFormer 没有报告“协议均值+偏移”这个强基线。不能声称我们在它的任务上超过它。

---

## 1. 执行环境、边界与代理情况

### 1.1 SSH 与服务器

- 本地 `~/.ssh/config` 存在 `lab-server` 别名，实际连接成功：`172.25.52.97`，用户 `yndx`。
- 服务器上目标目录原本不存在；已创建 `/home/yndx/zxk/deepseek-reaserch`，并上传研究包与交付物。
- 服务器无外网 DNS/路由（`getent hosts` 失败，TCP 到公网失败）。因此文献 GitHub 原件和 Battery Archive/ CALCE 数据是在**本地 Windows 联网环境**获取、校验后，再通过 SSH 传输；没有声称服务器直接联网成功。
- 未请求、未复制、未上传任何 SSH 私钥。

### 1.2 GPU 与共享资源

- 只读查看过服务器 `nvidia-smi`：A40 0 MiB、0% 使用、无进程；但本项目**没有使用服务器 GPU**，所有训练/推理在本地 CPU（PyTorch CPU）完成。
- 未操作 Codex 的 `/home/yndx/zxk/rul`，未杀进程，未占用其调度器/owner/test。
- 本地 Edge headless + Playwright 仅用于读取 Battery Archive 公开仪表盘数据；这不是子代理。

### 1.3 子代理

当前执行端没有可调用的子代理/并行代理接口；**全部串行完成**。没有声称启动过不存在的文献/数据/实验子代理。

---

## 2. 研究路线选择与理由

### 2.1 为什么不是 Codex 的在线 RUL

Codex 研究“每来一个新观测就更新单电池 RUL 分布/速率可靠性”。本文固定预测起点 `k0`，只使用 `k0` 及以前的容量/SOH（及已知工况），预测未来固定坐标 `k0+h` 的 SOH、阈值穿越和不确定性。两者输入权限、输出和评价协议不同；本文没有复用 Codex 的 DataV1、MATR/NASA 数据、模型或测试集。

### 2.2 为什么不是“knee 检测”

2024 JPS 曲率 knee、2025 多时间尺度 RUL/knee、2025 transferable physics-informed、2026 multi-fidelity 已经把“曲率/物理先验/knee onset/迁移”做得较充分。本文的数据诊断显示：**在容量-only 固定起点设定下，EOL 误差的主因是时间尺度/协议偏移，而不是“有没有检测到 knee”**。因此本文不把 knee 检测作为创新点；knee 只作为失败案例分析，不作为主任务。

### 2.3 本文的问题表述

> 给定目标电芯截至固定起点 `k0` 的容量/SOH 历史，以及已知的化学体系、温度、放电倍率等工况，预测未来 `k0+h` 的 SOH；并派生 80% EOL 穿越时间。评价必须按物理电芯隔离，区分观测到的 EOL 与右删失。

### 2.4 预设的可证伪假设

- **H1（近端/远端解耦）：** 短 horizon MAE 低，不代表长 horizon 或 EOL 可靠；不同模型类的偏差方向不同。
- **H2（协议均值主导）：** 在同一协议组内，协议均值是很强的基线；复杂序列模型的增益主要来自协议平均，而非个体前缀。
- **H3（时间尺度是瓶颈）：** 用真实 EOL 归一化后，轨迹形状高度重合；用前缀估计时间尺度则误差大。跨来源失败主要来自时间尺度/协议先验缺失。

本文的实测支持 H1、H2、H3；但 H3 的因果解释仍是推论，不是被证明的机制。

---

## 3. 数据可行性与审计

### 3.1 主数据集：Battery Archive 公开标准化数据

通过 Battery Archive 公开仪表盘（`database.batteryarchive.org`，公开 `Cycle Test Data` dashboard）读取：
- `api/queries/29`：cell metadata（来源、化学、容量、温度、SOC 窗口、C-rate）。
- `api/queries/26`：per-cycle `ah_c/ah_d/e_c/e_d`（充/放电容量与能量）。

检索日期：2026-10-19。只取 **SOC 0–100%** 的完整窗口电芯；共 **101 颗**。

| 来源 | 电芯数 | 观察到 EOL80 | 观察到 EOL70 | 循环中位数 | 主要化学/工况 |
|---|---:|---:|---:|---:|---|
| SNL | 61 | 39 | 10 | 785 | LFP/NCA/NMC；15/25/35°C；1/2/3C |
| HNEI | 15 | 15 | 15 | 1108 | NMC-LCO；25°C；0.5C 充/1.5C 放 |
| UL-PUR | 10 | 4 | 0 | 222 | NCA；23°C；0.5C |
| Oxford | 8 | 4 | 0 | 7750 | LCO；40°C；2C 充/1.84C 放；每 100 循环表征 |
| CALCE | 7 | 7 | 7 | 1815 | LCO；25°C；0.5C |
| **合计** | **101** | **69** | **32（EOL70）** | — | 5 来源、5 化学体系 |

**SOH/阈值定义（本项目实测采用）：**
- 有效循环：`0.2·nominal < ah_d < 1.4·nominal`，`0.2·nominal < ah_c < 1.4·nominal`，且 `0.8 < ah_d/ah_c < 1.2`。用于剔除双循环/半循环/异常 RPT。
- 初始容量：前 5 个有效循环的 `ah_d` 中位数。
- `SOH = ah_d / initial_capacity`；`EOL80`/`EOL70` 定义为平滑 SOH 首次 ≤0.80/0.70 的循环；未达到为右删失，不填伪精确 EOL。
- 目标平滑：居中 9 点中位数（仅用于评价标签）；模型输入使用**因果** 5 点中位数/原始历史，避免未来信息。
- 预测起点：`k0 ∈ {50,100,200,300}`（主表共同集用 `k0≥100`），horizon `h ∈ {50,100,200,300}`。

**数据质量限制（必须报告）：**
- Battery Archive 的 `ah_d` 在部分循环出现双循环/部分循环（例如 SNL cycle 4 可能约为 2×nominal），已按上述规则剔除；仍可能残留少量协议/RPT 伪影。
- Oxford 只有 46–78 个表征点覆盖 5,000–8,200 循环，EOL 标签稀疏；HOF 在 Oxford 的 EOL 失败（见 §6.4）。
- SNL LFP 多数电芯在 80% 前被删失（仅 5/21 观察到 EOL80），因此 EOL 指标是**条件于观察到 EOL**的子集，不是全样本无偏估计。
- 101 颗是物理电芯数，不是数千个独立样本；window 之间高度相关，主表同时报告 cell-equal 和 window-equal。

### 3.2 独立 CALCE 原始文件审计（不并入主 101 表）

从 CALCE 官方 `https://calce.umd.edu/data` 下载 16 个 zip，解析了 **12 颗** XLSX 电芯：
- CS2-33/34/35/36/37/38（6 颗；0.5C/1C，1.1 Ah，LCO）
- CX2-33/34/35/36/37/38（6 颗；0.5C/1C，1.35 Ah，LCO）

CS2-8/21 与 CX2-31 是 CADEX `.txt`，CX2-16 下载不完整，均未用于主分析。原始 zip 的 SHA256、URL、字节数在 `data/calce_manifest.json`；不把 900 MB 原始 zip 上传服务器，保留可复现 URL 与 hash。

CALCE 原始审计的 per-cell 结果（`results/diag_predictions.csv`，h=300，k0=100–400，n=48）：
- persistence MAE 0.0935，bias +0.0908（系统性乐观）
- linear100 0.0858，bias −0.0273
- exp100 0.0768，bias −0.0158
- power_all 0.0959，bias −0.0268
- EOL80 误差在数百循环量级（见 `results/diag_eol.csv`）

这与主数据集的方向一致：**短程可拟合，长程偏差结构化，knee/EOL 外推极不稳定**。但该审计的样本/协议与主表不同，不能把两表数字直接相加或比较。

### 3.3 与 Codex 的数据隔离

- Codex 主线使用 MATR/NASA 等；本文使用 Battery Archive 的 SNL/HNEI/UL-PUR/Oxford/CALCE + CALCE 原始文件。
- 未读取 `/home/yndx/zxk/rul` 内任何数据/权重/预测/脚本；未使用 Codex 的 DataV1、19 电芯开发流程或测试集。
- **但与 BatteryMFormer 的数据重叠风险仍存在**：BatteryMFormer 使用“13 个子数据集”的公共寿命库，本文的 SNL/HNEI/CALCE/Oxford 可能在其训练数据中。本文不能声称对 BatteryMFormer 完全数据独立；这一点是论文投稿前必须核实的风险。

---

## 4. 最近邻论文冲突矩阵

> 说明：BatteryMFormer、2024 曲率 knee、2025 transferable physics-informed 三篇阅读了 arXiv 全文/主要章节；其余以 OpenAlex/Crossref/Semantic Scholar 的标题、摘要、DOI 为主，未声称精读全文。2025 Applied Energy 是 closed access，未获得摘要/全文，只核到标题与作者/DOI。

| 论文 | 作者明确做的任务 | 输入截止 | EOL/knee 语义 | 数据/划分 | 与本文直接重合 | 仍未核实 |
|---|---|---|---|---|---|---|
| **BatteryMFormer (KDD 2026)** `arXiv:2605.27044` | 早期 operational data → 全寿命 SOH 轨迹；aging-condition-aware decoder + meta degradation memory + dual-view encoder | 早期最多 100 循环的电压/电流/SOC | EOL 阈值 80%（CALB 90%），部分外推补轨迹 | 4 domains（Li-ion/CALB/Na-ion/Zn-ion），aging-condition-exclusive 6:2:2 或 leave-one-condition-out | 都做长程 SOH 轨迹；但本文是 capacity-only、固定起点、任意 k0、多源/删失 | 其 13 子数据集是否含本文来源；代码/data split 未在本地运行 |
| **Zhang, Altaf, Wik 2024 JPS** `10.1016/j.jpowsour.2024.234619` | 曲率近似识别 knee/knee-onset；在线早期预测 | 容量 fade 曲线前缀 | 新 knee 定义 + curvature；EOL 相关性 | 两种化学 + 合成数据 | 都涉及 knee/EOL；本文不做 knee 检测 | 其真实数据来源/协议；与本文数据是否重叠 |
| **Multi-time scale 2025 JES** `10.1016/j.est.2025.116024` | 前 100 循环多时间尺度特征 + attention/MLP + snapshot ensemble → RUL/knee | 前 100 循环 | RUL + knee | 123 电池；主/副测试集 | 都做早期长程预测；本文 fixed-origin capacity-only | 数据身份、EOL 语义、代码 |
| **Coupling capacity fade + ML 2025 Applied Energy** `10.1016/j.apenergy.2025.125703` | 经验容量衰减模型 + ML 早期预测整条容量轨迹 | 早期循环 | 容量轨迹；可能含 EOL | 未知（closed） | 最接近的“参数化 fade + ML 轨迹”路线 | **全文/摘要/数据/代码均未获得**；不能声称精读 |
| **Transferable physics-informed 2025 JPS** `10.1016/j.jpowsour.2025.238028` | 2D histogram 特征 + 混合物理模型 + fine-tuning → 诊断/knee-onset/knee | 在线 prefix | knee onset/knee；EOL 相关 | ICL 18 cells；源域/目标域 | 都强调在线/迁移；本文不做 knee 检测 | 其轨迹预测能力；与本文数据重叠 |
| **Multi-fidelity 2026 GEITS** `10.1016/j.geits.2025.100367` | P2D + SEI/析锂 + 多保真 ML → knee/RUL | 稀疏早期数据 | knee + RUL | 仿真 + 实验？ | 物理先验 + 长程；本文是非参数层级基线 | 数据谱系；代码 |
| **EA-LSTM 2026 Results in Engineering** `10.1016/j.rineng.2026.109664` | 前 100 循环多循环统计特征 → KO/KP/EOL | 前 100 循环 | knee onset/knee/EOL | MIT MaTR 118 + CLO 32 | 早期 EOL/knee 多任务；本文不做 | 与 Codex 的 MATR 数据重叠风险 |

**新颖性边界（本项目判断，不是作者声明）：**
- 不能称“首次长程 SOH 轨迹预测”“首次 knee 检测”“首次不确定性”“首次物理+ML”。
- 本文可主张的是：**一个 capacity-only、fixed-origin、多来源、cell-level 隔离的实证分解**，以及**协议均值+电芯偏移这个强基线**；它暴露了“复杂模型增益可能主要来自协议平均”这一评价问题。
- 在拿到 BatteryMFormer 官方代码/数据划分、以及 2025 Applied Energy 全文之前，不能把本文写成“优于 BatteryMFormer/该文”的方法论文；只能写成**基准/现实核查 + 最小基线**。

---

## 5. 最小实验设计

### 5.1 评估协议

- **5 折按物理电芯隔离**，每折按 `(source, chemistry)` 分层，确保测试电芯从未出现在训练/验证中。
- 共同评估集：`k0≥100`、`h∈{50,100,200,300}`、且 `k0+h` 不超过该电芯最后有效循环；得到 **88 颗电芯、58,080 个窗口**。
- 主指标：**cell-equal MAE**（先对每颗电芯的所有起点/horizon 求平均，再对电芯等权平均）；同时给 window-equal 和 bias，避免长寿命电芯支配。
- EOL：只用**观察到 EOL80** 的电芯；`k0∈{100,200,300,400}` 且 `k0 < 0.9·EOL`；预测轨迹首次 ≤0.8 为穿越；10,000 循环内未穿越记为右删失/未决，不填伪 EOL。
- 未提前解锁独立确认集；5 折是开发集交叉验证，不是最终 test。

### 5.2 方法

- **S0 persistence：** 最后可用 SOH 常数外推。
- **S1 per-cell curve：** 对前缀（因果 `soh_in`）拟合 linear-100、exp-100、power-law（`1−a·k^p`）。
- **S2 protocol mean：** 训练折内同 `(chemistry, temperature, discharge C-rate)` 组的因果 5 点中位数 SOH 轨迹；缺失组回退全局中位数。
- **S3 HOF（本文最小方法）：** `SOH_hat(k) = μ_g(k) + r_c(k0)`，其中 `r_c(k0)=SOH_c(k0)−μ_g(k0)` 是当前偏移；`μ_g` 只用训练折。**`SOH_c(k0)` 与训练折的 `μ_g` 都使用因果 5 点中位数（trailing median，`soh_in`），不使用 k0 之后的任何观测；目标 `true` 仍用居中 9 点中位数作为评价标签。** EOL 由 `SOH_hat(k)` 首次 ≤0.8 得到。
- **S4 GRU：** 输入最近 100 循环的因果 SOH 历史（相对最后值的形状 + 最后绝对值，`[B,100,2]`），静态协变量 `[化学 one-hot, 温度, log C-rate, log nominal, k0]`；GRU hidden 32 → MLP → 4 个 horizon；5 折、40 epoch、MSE、early stopping。
- **S5 residual GRU：** 与 S4 同架构，但目标是相对 `μ_g` 的残差；预测时加回 `μ_g`。用于检验“残差动态”是否比常数偏移更强。
- **S6 区间：** 对 HOF 的 out-of-fold 绝对残差做 80%/90% 分位数，构造 `pred ± q` 的 split-conformal 区间；EOL 同理。报告覆盖率和宽度；这是校准基线，不声称 conformal 本身创新。

### 5.3 计算开销

- 全部 CPU；无 GPU、无服务器训练。GRU 5 折 × 40 epoch，约 15k 窗口/折；HOF 无训练（只有组中位数和偏移）；整体分钟级到十几分钟级，具体时间未逐项计时。
- 没有跑数十组模型/多 seed 矩阵；没有重跑成功 smoke。

---

## 6. 结果

### 6.1 主表：cell-equal SOH MAE（共同 88 电芯 / 58,080 窗口）

| 方法 | h=50 | h=100 | h=200 | h=300 |
|---|---:|---:|---:|---:|
| Persistence | 0.0162 | 0.0266 | 0.0505 | 0.0785 |
| Linear-100 | 0.0142 | 0.0218 | 0.0387 | 0.0552 |
| Exp-100 | 0.0119 | 0.0178 | 0.0302 | 0.0424 |
| Power-law | 0.0145 | 0.0204 | 0.0335 | 0.0477 |
| Protocol mean | 0.0240 | 0.0249 | 0.0281 | 0.0352 |
| GRU | 0.0128 | 0.0140 | 0.0183 | 0.0262 |
| Residual GRU | 0.0151 | 0.0185 | 0.0252 | 0.0344 |
| **HOF（本文）** | **0.0076** | **0.0103** | **0.0143** | **0.0224** |

**读法：**
- h=50 时 exp-100 已不错（0.0119），说明短程 shape 容易；但 h=300 时 exp/power 退化到 0.042–0.048，persistence 0.0785。
- Protocol mean 短程反而差（0.0240），因为不同电芯的初始偏移被平均掉了；HOF 加上当前偏移后短程改善到 0.0076。
- GRU 比 per-cell 曲线好，但 h=300 仍 0.0262；残差 GRU 更差（0.0344），说明**残差动态没有稳定增益**，常数偏移是更强的先验。
- HOF 在 h=300 相对最强 per-cell（exp 0.0424）降低约 **47.2%**；相对 GRU 降低约 **14.5%**。这些是开发集 5 折数字，不是独立确认集。

**bias（h=300）：** persistence +0.0333（乐观）、linear +0.0211、exp +0.0052、power +0.0073、protocol mean −0.0044、GRU −0.0013、HOF +0.0045。HOF 的偏差接近 0。

### 6.2 分来源：h=300 cell-equal SOH MAE

| 来源 | 电芯数 | Power | Protocol mean | GRU | HOF |
|---|---:|---:|---:|---:|---:|
| HNEI | 15 | 0.0777 | 0.0224 | 0.0214 | **0.0199** |
| CALCE | 7 | 0.0457 | 0.0681 | 0.0379 | **0.0368** |
| Oxford | 8 | 0.0091 | 0.0180 | **0.0051** | 0.0077 |
| SNL | 58 | 0.0455 | 0.0369 | 0.0289 | **0.0233** |

- HOF 在 HNEI/CALCE/SNL 最好；**Oxford 是 GRU 更好**（0.0051 vs HOF 0.0077），因为 Oxford 轨迹很平缓，因果偏移反而引入噪声。
- CALCE 只比 GRU 好一点（0.0368 vs 0.0379）；CALCE 的 cell-to-cell 形状差异更大，常数偏移不够。
- Oxford 的绝对 MAE 很低，是因为其 SOH 轨迹非常平缓（EOL 在 5,000–7,000 循环）；不能把这当作“Oxford 很容易预测 EOL”——见 §6.3。
- SNL 内部异质性最大（LFP/NCA/NMC、温度/倍率组合多），HOF 仍优于 GRU。协议组训练样本：HNEI 15、CALCE 7、Oxford 8；SNL 每个 `(chem,temp,rate)` 组只有 2–4 颗，5 折后训练侧 1–3 颗，因此 SNL 的协议均值本身更不稳。

### 6.3 EOL80：HOF vs 其他（观察到 EOL 的 cell-origin 案例）

| 方法 | 有预测案例 | MAE（循环） | Median | within-20%（有预测） | within-20%（全部 194 案例，含未穿越=失败） | bias | 未穿越/删失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HOF** | 157 | **165.3** | **48** | **75.8%** | **61.3%** | +63.5 | 37 |
| Protocol mean | 160 | 290.1 | 95 | 53.1% | 43.8% | +5.1 | 34 |
| Linear-100 | 194 | 422.6 | 142.5 | 45.4% | 45.4% | −385.0 | 0 |
| Exp-100 | 194 | 358.7 | 142 | 49.0% | 49.0% | −299.2 | 0 |
| Power-law | 166 | 285.6 | 121 | 37.3% | 32.0% | +122.0 | 28 |

**注意：**
- 线性/指数“从不删失”并不代表更好：它们总会外推到 0.8，但偏差极大（linear −385 循环，exp −299 循环），即系统性过早预测 EOL。HOF 会在证据不足时不穿越（37 个案例），这是更诚实的右删失处理。
- HOF 的 EOL 误差仍很大（MAE 165.3 循环）；在 HNEI 这种同协议、短寿命组很好（MAE 7.1），在 SNL NCA/NMC 异质组较差（MAE 134–143），在 Oxford 稀疏组失败。
- 若把未穿越案例计为失败，HOF 的 within-20% 为 61.3%（119/194）；protocol mean 为 43.8%（85/194）。

**HOF EOL 分来源（`results/eol_compare_clean.csv`，因果偏移）：**

| 来源 | 有误差案例 | 电芯 | MAE（case） | MAE（cell） | Median | within-20% | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| HNEI | 27 | 15 | 7.1 | 7.4 | 5 | 100% | 同协议、形状稳定 |
| CALCE | 28 | 7 | 65.6 | 65.6 | 45.5 | 92.9% | 可接受 |
| SNL | 98 | 35 | 142.7 | 134.3 | 64.5 | 70.0% | 异质化学/工况；19 个案例未穿越 |
| Oxford | 4（仅 1 颗电芯有穿越） | 1 | 2485.5 | 2485.5 | 2487 | 0% | 12 个案例未穿越；稀疏表征 |

Oxford 的 4 个 EOL80 观察电芯中，HOF 只对 1 颗在 10,000 循环内预测到穿越；另外 3 颗被预测为“未穿越”（12 个 cell-origin 案例）。未穿越按来源：UL-PUR 6、Oxford 12、SNL 19。这不是“误差大”，而是**模型在该来源上无法给出 EOL**。这是当前方法最重要的失败之一。

### 6.4 跨来源压力测试（leave-one-source-out）

训练 4 个来源，测试剩余 1 个来源；协议组/全局回退。h=300：

| 测试来源 | 电芯数 | HOF cell-equal MAE | HOF window-equal MAE | bias |
|---|---:|---:|---:|---:|
| HNEI | 15 | 0.1535 | — | — |
| CALCE | 7 | 0.0808 | — | — |
| SNL | 58 | 0.0728 | — | — |
| Oxford | 8 | 0.0190 | — | — |
| **总体** | **91** | **0.0820** | **0.0356** | **+0.0354** |

- 同协议 5 折的总体 cell-equal 是 **0.0224**；留一来源恶化到 **0.0820**（约 3.66×）。
- EOL 更严重：在全局回退下，194/194 个 cell-origin EOL 案例在 10,000 循环内**没有**预测到穿越。原因是全局中位数轨迹太平，永远到不了 0.8。
- 结论：**HOF 的“协议均值”不是跨来源可迁移的先验；没有协议先验时，当前方法不能用于新来源。** 这是论文若要主张通用方法必须先解决的问题。

### 6.5 区间校准（HOF，split-conformal）

**SOH：**

| horizon | 80% 覆盖率 | 80% 宽度 | 90% 覆盖率 | 90% 宽度 |
|---|---:|---:|---:|---:|
| 50 | 0.802 | 0.0084 | 0.899 | 0.0118 |
| 100 | 0.797 | 0.0092 | 0.899 | 0.0140 |
| 200 | 0.798 | 0.0110 | 0.899 | 0.0186 |
| 300 | 0.798 | 0.0130 | 0.897 | 0.0228 |

- 覆盖率接近名义值；宽度随 horizon 增长。这是**残差分位数**的校准，不是 joint path 保证；跨 horizon 的联合覆盖未保证。

**EOL：**

| 名义 | 覆盖率 | 宽度（循环） |
|---|---:|---:|
| 80% | 0.809 | 208.0 |
| 90% | 0.857 | 391.1 |

- 80% 接近名义；90% 覆盖不足（0.857），说明 EOL 尾部/来源异质性未被全局残差分位数完全捕获。分来源 80% 覆盖：HNEI 0.773、CALCE 0.800、SNL 0.791；Oxford 只有 4 个案例，无法稳定估计。
- 这是**开发集 out-of-fold** 结果；不是独立确认集的保证。

### 6.6 逐电芯最差案例（h=300，cell-equal MAE）

`results/per_cell_h300.csv` 给出 88 颗电芯的 HOF/pmean/GRU/power/exp 指标。最差 HOF：

| 来源 | 电芯 | HOF | Protocol mean | GRU | Power | 解释 |
|---|---|---:|---:|---:|---:|---|
| SNL | NCA_25C_0.5C_b | 0.2179 | 0.2769 | 0.2775 | 0.1768 | 快速/不规则衰减，偏移不恒定；幂律反而更好 |
| SNL | NCA_35C_2C_a | 0.1052 | 0.1532 | 0.0624 | 0.0841 | GRU 更好，说明形状变化 |
| SNL | NCA_25C_0.5C_a | 0.0989 | 0.1696 | 0.0388 | 0.0348 | GRU/幂律明显更好 |
| SNL | NCA_35C_1C_b | 0.0915 | 0.0852 | 0.0676 | 0.1408 | 高估/形状差异 |
| SNL | NMC_15C_1C_b | 0.0816 | 0.1430 | 0.1439 | 0.5623 | 所有模型都难 |
| CALCE | CX2_33 | 0.0565 | 0.0563 | 0.0410 | 0.0564 | GRU 更好 |

HOF 最好的案例主要是 Oxford LFP/LCO 平稳轨迹和 SNL LFP 15°C；这些轨迹形状稳定、偏移近似常数。

**失败机理（推论，不是已证明）：** 当电芯在预测起点后发生明显的**形状/时间尺度变化**（例如 NCA 快速 knee、CALCE 的个体曲率），常数偏移无法表达；这正是未来方法应该建模的“电芯级时间尺度/形状漂移”，而不是再增加短期时序容量。

### 6.7 配对显著性/一致性

- HOF vs GRU（h=300，88 颗电芯）：HOF 胜 **59**，GRU 胜 **29**，median Δ = **−0.00195**（负值=HOF 更好）。
- HOF vs Power（h=300）：HOF 胜 **69**，Power 胜 **19**，median Δ = **−0.00962**。
- **Cell-level paired bootstrap（20,000 次重采样，仅描述性，未做多重比较校正）：**
  - h=300 SOH：HOF−GRU = **−0.0048**，95% CI [−0.0091, −0.0006]，P(Δ≥0)=0.013；HOF−Power = **−0.0263**，95% CI [−0.0404, −0.0161]，P(Δ≥0)<0.001。
  - EOL：HOF−protocol mean = **−69.0 循环**，95% CI [−103.7, −39.4]；HOF−Power = **−269.3**，95% CI [−507.9, −85.9]。
  - 但 EOL HOF−Linear = −58.7，95% CI [−127.6, 14.7]；HOF−Exp = −34.2，95% CI [−99.8, 48.7]，CI 跨 0。**不能声称 EOL 对所有基线都显著更好。**
- 未做多 seed/多种子架构；这些 CI 不能替代独立确认集。

---

## 7. 方法有用/失败的证据

### 7.1 有用的证据

1. **同协议 5 折、共同评估集：** HOF 在 h=50/100/200/300 的总体 cell-equal MAE 均优于 GRU、protocol mean、per-cell 曲线；h=300 为 0.0224 vs GRU 0.0262、exp 0.0424、power 0.0477。分来源 Oxford 上 GRU 更好（0.0051 vs 0.0077）。
2. **EOL：** HOF MAE 165.3、median 48、within-20%（有预测）75.8%；protocol mean 290.1/95/53.1%；power 285.6/121/37.3%。HOF 不是只在 SOH 平均误差上好看，EOL 也改善；但 37/194 案例未预测穿越，Oxford 3/4 电芯未定位。
3. **一参数、可解释：** `μ_g(k)+r_c(k0)` 只有一个 cell-specific 参数；不需要训练神经网络；协议均值可审计，偏移可解释。这对 BMS 部署/审查有价值。
4. **区间：** HOF 的 out-of-fold 80%/90% SOH 区间覆盖率约 0.80/0.90；EOL 80% 覆盖率 0.809。说明残差校准在这个层级模型上是可行的。

### 7.2 失败的证据

1. **跨来源崩溃：** leave-one-source-out h=300 cell-equal 0.0820（同协议 0.0224）；HNEI 0.1535、SNL 0.0728；EOL 194/194 无穿越。没有协议先验时不可用。
2. **Oxford EOL 失败：** 4 颗有 EOL 的电芯中只有 1 颗在 10,000 循环内被预测穿越；另外 3 颗被模型判为“不穿越”。这说明稀疏表征 + 平缓轨迹下，协议均值形状无法定位 EOL。
3. **CALCE/SNL NCA 的形状变化：** HOF 在 CALCE 只略优于 GRU；在 SNL NCA 25C 0.5C_b 明显差于 power（0.2179 vs 0.1768）。常数偏移假设在形状突变时失效。
4. **残差 GRU 无效：** 残差 GRU h=300 0.0344，比 HOF 0.0224 和普通 GRU 0.0262 都差；说明“学习残差动态”没有稳定收益，反而过拟合。
5. **EOL 90% 区间覆盖不足：** 0.857 < 0.90；跨来源/尾部未校准。
6. **早期全局形状/仿射时间 warp 失败：** 用真实 EOL 归一化的全局 shape 反演在 Oxford 预测 EOL 到数万循环；仿射时间 warp 的 EOL MAE 约 400–4,300 循环（`code/shape_model.py`、`code/warp_model.py` 的探索性日志）。说明仅靠“共享形状”不够，协议先验/当前偏移是必要的。

### 7.3 不能声称的

- 不能声称“HOF 是 SOTA”或“优于 BatteryMFormer”。我们没有在 BatteryMFormer 的输入/任务/数据划分上运行它。
- 不能声称“跨来源通用”。留一来源测试明确失败。
- 不能声称“EOL 已解决”。Oxford/SNL NCA 的误差和删失仍很大。
- 不能声称“数据完全独立于 BatteryMFormer”。其 13 子数据集可能包含本文来源，需核实。

---

## 8. 限制、风险与论文可行性

### 8.1 统计与数据限制

- 101 颗物理电芯，但 69 颗有 EOL80；EOL 指标是条件子集，SNL LFP 大量删失。
- 5 折是开发集交叉验证；没有冻结的独立确认集。留一来源是压力测试，不是最终测试。
- Oxford 只有 46–78 个表征点；EOL 标签稀疏，EOL 误差 2,485 循环不能代表“模型在 Oxford 上平均很差”，而是“模型无法定位 3/4 电芯的 EOL”。
- Battery Archive 的标准化容量可能已经过平滑/协议归一；与原始 CALCE 文件不完全同分布。
- 没有多 seed、bootstrap CI、正式显著性检验；HOF vs GRU 的 cell-level 胜负是描述性证据。

### 8.2 竞争与新颖性风险

- BatteryMFormer 是直接强竞争；它用电压/电流、早期输入、复杂多层结构。本文用容量-only、固定起点、HOF。两者任务不同，但审稿人可能问“为什么不和 BatteryMFormer 比”。目前只能诚实回答：输入不兼容，未运行；需要适配或在共同子集上比较。
- 2025 Applied Energy 的“capacity fade model + ML”与本文的“协议均值+偏移”在思想上接近（都是强先验+个体修正）；必须拿到全文后才能判断重合度。当前不能声称新颖性。
- 2024/2025 knee 论文覆盖了 knee 检测/预测；本文不做 knee 检测，避免直接冲突，但也少了一个“方法模块”。
- 本文的贡献更像**benchmark/reality-check + 最小基线**，不是 KDD 级别的架构创新。若目标是 KDD/NeurIPS，需要后续把“协议先验迁移/时间尺度不确定性”做成真正的新方法；若目标是 Applied Energy/JPS/PHM，当前的证据链已可支撑一篇“评价与基线”论文。

### 8.3 复现风险

- Battery Archive 是公开仪表盘，不是稳定文件 API；查询 26/29 的 schema 可能变化。已保存抓取脚本和原始 JSON（`data/ba_raw2.tar.gz`），但未来复现需固定日期/URL。
- CALCE 原始 zip 未上传（体积大）；用 `calce_manifest.json` 的 URL+SHA256 重新下载。CADEX txt 未解析；CX2-16 不完整。
- GRU/Residual GRU 只跑了一个 seed/架构；不是“强模型上限”。HOF 的优势是相对这个 GRU，不是相对所有可能的深度模型。

---

## 9. 下一个最高价值动作（按优先级）

1. **冻结一个真正的独立确认协议，并解决跨来源先验。**  
   当前 leave-one-source-out 明确失败（h=300 0.0820；EOL 无穿越）。最高价值动作不是再调 GRU，而是：用训练来源学习“协议→时间尺度”的先验（例如按 chemistry/temperature/C-rate 的层级收缩，或从少量目标来源无标签轨迹做 test-time calibration），在**完全留出的来源**上验证。若做不到，就把论文范围收缩为“seen-protocol capacity-only”，并明确不声称通用。
2. **拿到 BatteryMFormer 官方代码/数据划分，做共同子集对比。**  
   至少确认：其 13 子数据集是否含 SNL/HNEI/CALCE/Oxford；其 aging-condition-exclusive split 是否与本文重叠；在 capacity-only 输入下能否公平适配。没有这个，本文的“强基线”主张会被审稿人质疑。
3. **拿到 2025 Applied Energy 全文，写一页重合矩阵。**  
   这是目前最接近的“参数化 fade + ML 轨迹”工作；如果它已经报告了协议均值/偏移基线，本文的新颖性需要重新定位。
4. **补正式不确定性与统计。**  
   对 HOF vs GRU/Power 做 cell-level paired bootstrap CI；对 EOL 做生存/删失敏感的区间（当前 90% 覆盖 0.857）；报告 per-source/per-chemistry 的 coverage。不要只报平均值。
5. **把 CALCE CS2 原始 6 颗并入独立审计（不是主表）。**  
   目前主 101 表只有 CALCE CX2 7 颗；CS2 原始 6 颗可提供同化学、不同倍率的独立检查。它们已经解析，但尚未与 Battery Archive 的 SOH 定义完全对齐；先做协议/容量定义审计，再合并。

---

## 10. 三类结果标注

### 10.1 作者原文报告（不是本项目实测）

- BatteryMFormer：4 domains、MAE/MAPE、aging-condition-exclusive split、meta memory/dual-view 等，来自 arXiv:2605.27044v2 全文；具体数字以其论文为准。
- 2024 JPS 曲率 knee：曲率近似、knee/knee-onset 定义、与 EOL 强相关，来自 arXiv:2304.11671v3 全文/摘要。
- 2025 JPS transferable physics-informed：2D histogram、hybrid physics model、fine-tuning、ICL 数据，来自 arXiv:2501.14573v2 全文/摘要。
- 2025 JES、2025 Applied Energy、2026 GEITS、2026 EA-LSTM：仅标题/摘要/DOI 级别；未精读全文，不能引用其具体数值作为本文结论。

### 10.2 本项目实测

- 101 颗 Battery Archive 电芯的 SOH/EOL 表、HOF/GRU/Residual GRU/协议均值/曲线基线、5 折与 leave-one-source-out、conformal 覆盖率。
- 12 颗 CALCE 原始 XLSX 的 per-cell 曲线基线（仅审计）。
- 所有数字来自 `results/` 下的 CSV/JSON；不是作者报告，也不是计划值。

### 10.3 推论（需进一步证明）

- “协议均值+电芯偏移是主要可解释信号”“时间尺度是瓶颈”“残差动态无稳定增益”“跨来源失败源于协议先验缺失”是**基于当前数据的推论**，不是因果证明。
- “HOF 的增益来自偏移持久性”是机制假设；残差 GRU 失败支持它，但没有做受控合成实验。
- “BatteryMFormer 的复杂度在 capacity-only 设定下可能不必要”是**待验证的竞争判断**，不是已证明结论。

---

## 11. 复现文件与命令

### 11.1 主要文件

| 路径 | 内容 |
|---|---|
| `reports/DEEPSEEK_RESEARCH_REPORT_20261019_ZH.md` | 本报告 |
| `DATA_DECISION.md` | 数据审计/角色/阈值/删失 |
| `BASELINE_RESULTS.md` | 主表与 EOL/区间结果 |
| `NOVELTY_CONFLICT_MATRIX.md` | 最近邻论文冲突矩阵 |
| `results/final_metrics.json` | 所有主指标（机器可读） |
| `results/hof_eval_clean.csv` | HOF 逐窗口预测/真值（含 pmean/fold） |
| `results/hof_eol_clean.csv` | HOF EOL 预测/误差/删失 |
| `results/eol_compare_clean.csv` | HOF/pmean/linear/exp/power EOL 对比 |
| `results/unified_percell.csv` | per-cell 曲线基线逐窗口 |
| `results/gru_pred.csv` | GRU 逐窗口 |
| `results/resgru_pred.csv` | residual GRU 逐窗口 |
| `results/losource_hof_pred.csv` | leave-one-source-out HOF |
| `results/per_cell_h300.csv` | 逐电芯 h=300 指标 |
| `data/ba_clean.csv` | 清洗后的 101 电芯逐循环 SOH/元数据 |
| `data/ba_cell_summary.csv` | 逐电芯 EOL/循环/化学/工况 |
| `data/celllist.json` | Battery Archive 元数据原始行 |
| `data/ba_raw2.tar.gz` | 101 个 query-26 原始 JSON（公开仪表盘抓取） |
| `data/calce_manifest.json` | CALCE zip URL/SHA256/解析状态 |
| `data/calce_parsed/` | 12 颗 CALCE XLSX 的逐循环 CSV |
| `code/` | 抓取/清洗/HOF/GRU/评估脚本 |
| `figures/` | fig1 MAE vs horizon、fig2 EOL error by source、fig3 cross-source |

### 11.2 最小复现命令（在项目根目录）

```bash
# 1) 依赖（CPU；不要改共享 base 环境）
python -m pip install numpy pandas scipy scikit-learn matplotlib torch
# 2) 一键复算（推荐；run_all.sh 会创建 work/ 并链接 data/results）
PYTHON=<有 numpy/pandas/scipy/sklearn 的独立环境 python> bash code/run_all.sh
# 3) 或手动进入 work/ 后运行（脚本使用相对路径）
cd work
$PYTHON ../code/hof_eval_clean.py
$PYTHON ../code/eol_compare_clean.py
$PYTHON ../code/final_metrics.py
# 4) GRU / residual GRU（可选，CPU 分钟级；同样在 work/ 下）
$PYTHON ../code/train_gru.py
$PYTHON ../code/train_resgru.py
# 5) 跨来源压力测试
$PYTHON ../code/losource_hof.py
```

**服务器已实测：** `PYTHON=/home/yndx/miniconda3/envs/a40_env/bin/python bash code/run_all.sh` 可复算；该环境 pandas 2.0.3，因此 `code/final_metrics.py` 已去掉 pandas 2.2 才支持的 `include_groups` 参数。

**注意：** `code/` 中的脚本最初在本地 `/tmp` 下运行，使用相对路径读取 `ba_clean.csv`、`celllist.json` 等；上传服务器时已按 `data/`/`results/` 目录重排，运行前请检查脚本中的路径或使用 `code/run_all.sh`（若已生成）。如果路径不一致，先改路径，不要修改数据。

### 11.3 数据获取（本地联网）

- Battery Archive：公开 dashboard `Cycle Test Data`，query 29（cell list）与 query 26（capacity/energy decay）；`code/scrape_full.py` 用 Playwright + 本地 Edge 抓取，保存 `query_result` JSON。需要本地已安装 Edge 和 `playwright`；服务器无外网，不能直接在服务器跑。
- CALCE：`https://calce.umd.edu/data` → `https://web.calce.umd.edu/batteries/data/<CELL>.zip`；`data/calce_manifest.json` 有 SHA256。
- 文献：`tools/fetch_original_literature.py` 已在本地运行并 SHA 校验 7 份原总结；原件在 `literature/original_from_github/`。没有上传论文 PDF。

---

## 12. 最终判断（给用户的短答）

- **研究路线选择理由：** 固定起点、容量-only、多来源，是 Codex 在线 RUL 之外、且能独立验证的退化轨迹问题；knee 检测已被 2024/2025 覆盖，不作为创新点。
- **最接近论文的实质冲突：** BatteryMFormer（早期电压/电流→全寿命轨迹，复杂多层）；2025 Applied Energy（fade model+ML 轨迹，尚未拿到全文）；本文不能声称在它们任务上更强。
- **数据可行性：** 101 颗官方电芯、5 来源、69 颗观察到 EOL80；主表 cell-level 5 折；Oxford 稀疏、SNL LFP 删失是主要限制。
- **最小实际结果：** HOF（协议均值+当前偏移）h=300 cell-equal SOH MAE 0.0224，优于 GRU 0.0262、exp 0.0424、power 0.0477、persistence 0.0785；EOL MAE 165.3、median 48、within-20%（有预测）75.8%；37 个案例未穿越，Oxford 3/4 电芯未定位。
- **方法有用/失败：** 同协议内有用且简单可解释；跨来源崩（0.0820，EOL 无穿越）、Oxford EOL 无法定位、CALCE/SNL NCA 形状变化时失效。
- **下一个最高价值动作：** 冻结独立来源确认集 + 解决“协议先验迁移/时间尺度不确定性”；在此之前不要把本文写成通用 SOTA 方法论文。
