# CRULP 与 IJPHM 2026 差异表（D009 修订清单第 ① 步）

> 生成于 2026-11-09。核实方式：CrossRef / OpenAlex / Semantic Scholar API + 出版方页面，
> 不采信二手转述。**凡是需要全文才能确认的行都标 ⚠，不得在论文里写成确定事实。**

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| CRULP 是否真实存在 | ✅ 三源一致（CrossRef / OpenAlex / Semantic Scholar） |
| IJPHM 2026 是否真实存在 | ✅ CrossRef 给出 DOI 与卷期 |
| 它们是否占据"CQR 用于 C-MAPSS RUL" | ✅ 都占；CQR-RUL 已不能承载方法贡献（支持 D016） |
| 我们能否主张"优于 CRULP" | ❌ 未在同一协议下比较，不能 |
| 我们能否主张"证伪 CRULP 的认知不确定性路线" | ❌ **不能**：CRULP 用 UACQR，我们的 MC-Dropout 不是 UACQR（见 §4） |
| 本表是否支持 D016 的推翻条件 | ❌ 不支持：没有发现冻结方法含"此前遗漏的实质原创贡献" |

## 1. 核实记录（可复现）

### 1.1 CRULP

| 字段 | 值 | 来源 |
|---|---|---|
| 标题 | CRULP: Reliable RUL Estimation Inspired by Conformal Prediction | CrossRef |
| 作者 | Shiyuan Piao, Ruyi Huang, Fugee Tsung（HKUST 广州） | CrossRef |
| 期刊 | IEEE Transactions on Instrumentation and Measurement | CrossRef |
| 卷/页 | vol. 74, pp. 1–11（IEEE 文档号 10816036，文章号 3505411） | CrossRef / IEEE Xplore |
| 年份 | 2025（在线 2024-12-25；OpenAlex publication_date 2024-12-25） | CrossRef / OpenAlex |
| DOI | `10.1109/TIM.2024.3522678` | CrossRef |
| 被引 | OpenAlex 79 / CrossRef 74 / Semantic Scholar 70（2026-11-08 快照，三源不同步属正常） | 三源 |
| 关键词 | conformal prediction, **epistemic uncertainty**, quantile regression, reliability, RUL | OpenAlex |
| 开放获取 | 否（closed access，无 OA 全文） | OpenAlex / S2 |

摘要原文要点（S2 与 OpenAlex 一致）：
> … integrating the deep learning (DL)-based RUL prediction model and **uncertainty awareness
> conformalized quantile regression (UACQR)**. The proposed CRULP method … converts the
> single-point RUL prediction model into construct uncertainty set (i.e., interval in RUL
> estimation) **based on the epistemic uncertainty** through a collaborative process of
> **ensemble conformalized quantile regression (CQR) and cross-conformal prediction**.
> Experiments conducted on the **two wildly used aero engine datasets** …

参考文献中的关键条目（CrossRef 参考文献列表，佐证其机制来源）：
- ref44 Romano et al., *Conformalized Quantile Regression*（CQR 本体）
- ref45 Rossellini et al., *Integrating uncertainty awareness into conformalized quantile regression*（UACQR 本体）
- ref50 Saxena et al., *Turbofan engine degradation simulation data set*（C-MAPSS）

⚠ **未核实（需全文）**：具体用了哪两个航发数据集（摘要只写"两个航发数据集"，参考文献含 C-MAPSS，
但未确认是 FD001–FD004 的哪几个，也未确认第二个数据集是什么）；报告的指标定义与数值；
是否报告引擎级覆盖；是否做覆盖对齐比较；代码是否公开。

### 1.2 IJPHM 2026

| 字段 | 值 | 来源 |
|---|---|---|
| 标题 | Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression | CrossRef |
| 作者 | Colby Robinson（单作者；ORCID 0009-0009-2286-3206） | CrossRef |
| 期刊 | International Journal of Prognostics and Health Management | CrossRef |
| 卷/期 | 17(1) | CrossRef |
| 出版日期 | 2026-04-27 | CrossRef |
| DOI | `10.36001/ijphm.2026.v17i1.4724` | CrossRef |
| 方法（据摘要与公开代码仓） | 梯度提升回归（GBR）做点预测 + CQR 构造区间 + 航空风险偏好 | 摘要 / GitHub 仓库说明 |
| 数据 | **NASA C-MAPSS 全部四个子集** | GitHub 仓库说明 |
| 代码 | 公开（GitHub: ColbyRobinson/RUL-Prediction-Framework-…） | GitHub |

⚠ **未核实（需全文）**：具体覆盖/宽度数值；是否报告引擎级覆盖；是否做覆盖对齐比较。

> 注：HANDOFF §2① 写的"2026 年 IJPHM … article 4724"**核实为真**，且拿到了确切 DOI；
> 原文把它记为"未独立核实"，本表完成核实。

## 2. 机制差异表（核心）

| 维度 | CRULP（IEEE TIM 2025） | Robinson（IJPHM 2026） | 我们（冻结基线） |
|---|---|---|---|
| 基础预测器 | DL RUL 预测器（⚠ 未点名具体结构） | 梯度提升回归（GBR） | IDA-PE 主干 + 双分位头（端到端联合训练） |
| 区间机制 | UACQR：集成 CQR + cross-conformal，**以认知不确定性构造不确定集** | CQR（分位回归 + 保形校准） | 端到端双分位输出 + CQR split conformal |
| 半宽的来源 | 认知不确定性（epistemic），按特征空间区域差异化调整分位 | 学到的分位宽度（aleatoric 主导） | 网络学到的条件半宽 |
| 校准方式 | cross-conformal + ensemble | split conformal | split conformal（引擎级划分，calib 30%） |
| 数据 | 两个航发数据集（⚠ 未确认是否四子集） | C-MAPSS FD001–FD004 | C-MAPSS FD001–FD004 + FEMTO 跨域（6 轴承） |
| 报告指标 | 覆盖 + 宽度（"desired coverage … as narrow as possible"） | 覆盖/宽度 + 风险偏好（⚠ 具体指标未核实） | 覆盖、Winkler（**α 固定 0.1**）、引擎级最小覆盖、跨域 |
| 覆盖对齐比较 | ⚠ 未核实 | ⚠ 未核实 | ✅ 是（D005 修正后的协议：固定 α、按各自前沿插值到 0.90） |
| 引擎级覆盖 | ⚠ 未核实 | ⚠ 未核实 | ✅ 是（每引擎最小覆盖 0.50–0.71，18/20 折 <0.80） |
| 跨域验证 | ⚠ 未核实 | 仅 C-MAPSS 四子集 | FEMTO 5/6、均值 −12.2% |
| 代码 | ⚠ 未核实 | ✅ 公开 | 仓库内可复现（a40 产物） |

## 3. 我们能主张什么 / 不能主张什么

**能主张（证据在本仓库）**
1. 在固定 α=0.1 与覆盖对齐协议下，CQR 相对常数宽度的效率收益：75/80 格、均值 −9.1%、中位 −10.8%。
2. 名义水平下的配对结果：19/20 折、中位 −8.7%。
3. **平均覆盖掩盖引擎级失效**：边际 ≈0.90，但每引擎最小覆盖 0.50–0.71；这是可复现的诊断。
4. 跨域 FEMTO 的经验迁移：5/6、−12.2%。

**不能主张**
1. ❌ 方法学新颖性（CQR-RUL 已被 CRULP / IJPHM 2026 / Sensors 2026 等占满）。
2. ❌ "优于 CRULP"——没有在同一划分、同一 α、同一指标下比较。
3. ❌ **"证伪了 CRULP 的认知不确定性尺度"**——见 §4，机制不对应。
4. ❌ "均匀改善"（FD003/FD004 在覆盖对齐口径下接近打平）。

## 4. 关键更正：MC-Dropout ≠ UACQR（不要把这条写错）

- **CRULP 的机制**是 UACQR（Rossellini, Barber & Willett, AISTATS 2023；arXiv:2306.08693）：
  把**认知不确定性**与 aleatoric 不确定性分开，用认知不确定性估计**按特征空间区域差异化调整分位回归**。
- **我们的 MC-Dropout 基线**是另一种操作化：对同一个点预测主干做 T=30 次 dropout 采样，
  取其标准差作为**每样本 conformal 尺度**（`scripts/run_mcdropout.py`）。
- 因此 MC-Dropout 的结果**不能**直接外推成"CRULP 的机制被证伪"。
  可辩护的说法只能是：

  > "以 dropout 采样标准差作为每样本条件尺度，在 C-MAPSS 上并未带来相对常数宽度的改善
  > （待 80 格结果确认）；这提示**该操作化**的认知不确定性尺度不是有效的条件尺度。"

  若要直接评价 CRULP 的机制，必须实现 UACQR 基线（是否值得做，交 L2 裁决，见 §6）。

## 5. 邻近工作登记（供新立项查新，均为本轮新发现）

| 工作 | 出处 | 与我们的关系 |
|---|---|---|
| Robinson 2026 | IJPHM 17(1), DOI `10.36001/ijphm.2026.v17i1.4724` | CQR + 四子集 + 风险偏好；占住"CQR 应用稿"位置 |
| Onilede 2026 | TechRxiv，DOI `10.36227/techrxiv.177220374.46388084/v1` | 涡扇 CQR + 分布无关 UQ；⚠ 预印本，未同行评审 |
| Diao et al. 2026 | Sensors 26(7):2249，DOI `10.3390/s26072249` | LSTM 分位回归 + 保形校准，含跨工况（FD001→FD002 等） |
| **Zenodo 审计包 2026-07-13** | `zenodo.org/records/21330745`（University of Arizona） | **与我们新立项角度最近**：C-MAPSS FD001–FD004 + N-CMAPSS DS02 的保形区间**审计**，含 split conformal、CV+、ACI、**regime-conditional ACI**、**Mondrian（按预测 RUL 带分层）**、CQR 分数；5 个基础预测器；逐引擎区间 parquet。⚠ 论文本体未核实，必须先读再定新立项的差异化 |

> 最后一条对 D016 的新立项影响最大：如果那篇审计已经覆盖"按工况/寿命带分层后覆盖仍失效"，
> 我们新立项必须落在**可识别性**（能否只用校准集信息预测哪些对象会失效），而不是再报一次分层覆盖。

## 6. 待 L2 裁决（已提交 Codex）

1. 在 D016（TIM 路线已关闭）之下，80 格 MC-Dropout 结果的正确用途是什么：
   未来稿的相关工作定位 / 新立项的动机证据 / 还是做完 Stage A 即止？
2. §4 的说法（限定到"dropout 采样尺度"）是否足以支撑一个可发表的经验结论，
   还是必须实现 UACQR 基线才能谈 CRULP 的机制？
3. Zenodo 审计包是否已经占掉新立项的角度；若是，新立项的可识别性问题该怎么切？

## 7. 审计包本体核验（2026-11-09 02:00，Codex C3 行动）

下载 `zenodo_upload_package.zip`（12.0 MB）并读了包内 `docs/methodology.md`、
`docs/phase8_summary.md`、`docs/mondrian_upgrade.md`、`ARTIFACT_MANIFEST.md`。结论如下。

**它做了什么（已核实）**
- 5 个基础预测器：LSTM、CNN-LSTM 双注意力、encoder-only Transformer、functional MLP（B 样条）、LightGBM。
- 3 类工况检测器：KMeans / HDBSCAN / Beta-VAE。
- 4 个保形层：split、jackknife+（引擎级 LOO，>100 引擎时退化为 10 折引擎分层 CV）、
  ACI、**regime-conditional ACI**（本文贡献）+ 池化收缩 λ。
- **CPA**（Conformal Prediction Assessment）：核平滑的逐实例条件覆盖 + 按 regime / RUL 带分组，
  统计量是各组覆盖与名义值的平均绝对偏差。
- **Mondrian 升级**：按预测 RUL 带分层；包内记录 (60,120] 带仍低于目标、α≤0.10 时轻微边际欠覆盖。
- MILP 维护决策层（HiGHS + CBC 交叉验证）。
- 外部验证：N-CMAPSS DS02。

**它没有做什么（已核实）**
- 没有**对象（引擎）级**失效的第一类对象化：逐引擎产物只用于 MILP，不是研究对象。
- 没有**可识别性**：不预测"哪些对象会失效"，也不给识别边界；
  其 regime-ACI 需要**在线标签**才能更新 α，因此不是"只用校准集信息"的预警。
- 包内自陈局限："Engine count: only 4 training engines + 2 calibration engines is small;
  the per-engine variance is high. Multi-seed sweeps on the engine assignment are a recommended follow-up."
  —— 他们**知道**引擎级方差大，但没有把它变成研究问题。

**对 D018 新问题的含义**

| 问题 | 审计包 | 我们（D018） |
|---|---|---|
| 覆盖失效的轴 | regime、RUL 带 | **引擎（对象）** |
| 做什么 | 测量 + 补救（ACI/regime-ACI/Mondrian） | **先判定可识别性**（能否预测、边界在哪） |
| 需要什么信息 | 在线标签（ACI 更新） | **仅校准集信息**（不含测试标签） |
| 结论形态 | 覆盖率/宽度/成本 | 识别边界（可识别 / 不可识别及其条件） |

→ **可识别性框架仍然成立**，但必须在 Related Work 明确区分上述三轴；
并且**不能**再声称"分层后覆盖仍失效"是我们的新发现（他们已报 regime 与 RUL 带两个轴）。

**一条可低成本检验的差异化假设（建议下一步，未执行）**

> H-REG：工况（operating regime）分层校准**不能**修复引擎级失效。
> 判据：在冻结 CQR 预测上做 regime-Mondrian 校准，若 FD002/FD004 的 engine_min 仍 <0.60，
> 则"失效是对象级而非工况级"成立；若 engine_min 升到 ≥0.80，则该假设被证伪。

- 数据可得：`experiments/punet_matrix/*/preds_cqr.csv` + `data/` 原始 op setting（按 unit+cutoff 连接），**无需重训**。
- 与已关闭路线的区别：已关闭的 Mondrian 是 μ/σ 分箱，**不是**工况分层；这一点必须在实验前写清。
- 需先立判定门与预算，再交 L2 立项门。



## 8. 可直接使用的 BibTeX（已核实，未写入 references.bib）

> 未写入的原因：D016 已冻结当前稿件，`references.bib` 参与构建断言（18 条/11 处引用），
> 现在插入会破坏断言且没有对应正文引用。恢复投稿时再插入。

```bibtex
@article{piao2025crulp,
  title   = {{CRULP}: Reliable {RUL} Estimation Inspired by Conformal Prediction},
  author  = {Piao, Shiyuan and Huang, Ruyi and Tsung, Fugee},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {74},
  pages   = {1--11},
  year    = {2025},
  doi     = {10.1109/TIM.2024.3522678},
  note    = {Art. no. 3505411}
}

@article{robinson2026ijphm,
  title   = {Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware
             Prediction Intervals via Conformalized Quantile Regression},
  author  = {Robinson, Colby},
  journal = {International Journal of Prognostics and Health Management},
  volume  = {17},
  number  = {1},
  year    = {2026},
  doi     = {10.36001/ijphm.2026.v17i1.4724}
}

@inproceedings{rossellini2023uacqr,
  title     = {Integrating Uncertainty Awareness into Conformalized Quantile Regression},
  author    = {Rossellini, Raphael and Barber, Rina Foygel and Willett, Rebecca M.},
  booktitle = {Proceedings of the 27th International Conference on Artificial Intelligence
               and Statistics (AISTATS)},
  year      = {2023},
  note      = {arXiv:2306.08693}
}
```
