# 复现指南（Reproducibility）

> 本文件说明如何从原始数据重新生成论文中的每一个数字、表和图。
> 自动化核对：`python scripts/verify_paper_numbers.py`（16 项断言，全部通过才算有效）。

## 1. 环境

| 项 | 值 |
|---|---|
| 实验机 | `a40`（Linux，无外网、无 git） |
| Python | `/home/yndx/miniconda3/envs/a40_env/bin/python` |
| 必须设置 | `export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH`（否则 sklearn 报 GLIBCXX 错） |
| 本地 | Windows；pandoc 3.13、tectonic 0.15.0（编译 PDF） |
| 数据 | C-MAPSS 在 `data/`；FEMTO 在远端 `~/femto`（3.1 GB） |

## 2. 一键复现顺序

```bash
# ① 训练矩阵（4 子集 × 5 折 × 4 seeds × 2 头 = 80 格）
bash ~/matrix2seeds.sh            # 或 run_more_seeds.sh 补 seed 2024/13
# ② 覆盖对齐五方法对比 → docs/tables/cells_all.csv
python scripts/table_all_methods.py
# ③ 论文表格
python scripts/make_tim_tables.py
# ④ 覆盖诊断
python scripts/aggregate_diagnostics.py
# ⑤ 跨域 FEMTO
python scripts/run_femto_cqr.py --only Bearing1_1,... --label rel
python scripts/evaluate_femto.py
# ⑥ 图
python scripts/make_figures.py
# ⑦ 数字核对（必须 16/16）
python scripts/verify_paper_numbers.py
```

本地编译：

```powershell
node scripts/build_manuscript.mjs               # markdown -> LaTeX -> PDF (includes abstract sync)
```

## 3. 论文元素 → 脚本 → 产物

| 论文元素 | 生成脚本 | 产物 |
|---|---|---|
| Table 1（名义水平 20 折） | `make_tim_tables.py` | `experiments/punet_e2e/*/e2e_*.json` |
| Table 2（覆盖对齐 80 格） | `table_all_methods.py` | `docs/tables/cells_all.csv` |
| Table 3（覆盖诊断） | `aggregate_diagnostics.py` | `experiments/punet_e2e/*/e2e_cqr_none.json` |
| Fig. 1（覆盖-宽度前沿） | `make_figures.py` | `docs/figures/fig1_frontiers.pdf` |
| Fig. 2（分阶段覆盖） | `make_figures.py` | `docs/figures/fig2_stages.pdf` |
| Fig. 3（引擎级覆盖箱线） | `make_figures.py` | `docs/figures/fig3_engine.pdf` |
| §V.E 不确定性来源表 | `table_all_methods.py` | 同上 |
| §V.F A4 置换 | `verify_paper_numbers.py` 内部重算 | `experiments/punet_e2e/*/e2e_cqr_A4.json` |
| §V.G FEMTO 跨域 | `evaluate_femto.py` | `experiments/femto/*/preds_*.csv` |

## 4. 已关闭路线（不要重复跑）

| 路线 | 结论 | 证据 |
|---|---|---|
| 非对称校准 | 最优=对称 | `docs/tables/asymmetric.csv` |
| 分层/Mondrian 校准 | 全部更差 | `docs/tables/mondrian.csv` |
| 引擎级校准 | 等价于放宽区间 | `docs/tables/engine_vs_inflation.csv` |
| seed 集成 | n=2 有效、n=4 退化 | `docs/tables/ensemble_scaling.csv` |
| 局部化 CQR | 40/80 格无收益 | `docs/tables/cells_all.csv` |
| 宽度放大（D001） | a=1 最优 | `docs/tables/width_scaling.csv` |
| 上下文中心修正（D002） | 不可跨引擎迁移 | `docs/tables/d002_centre.csv` |

## 5. 已知限制

- **MC-Dropout 只有 10 格**（不是 80），论文 Table 2 中该列为部分观测，已注明
- **FEMTO 只用了 6 个全寿命轴承**（Learning_set），其余 11 个是截断记录，无完整 RUL 标签
- **FEMTO 每轴承最多 300 个窗口**（等间隔下采样），为控制训练时间
- 表格仍有 1 处 21pt 溢出（约 0.3 英寸），排版定稿时微调

## 6. 随机性

- C-MAPSS：5 折由 `outer_folds` 固定；seed 控制模型初始化与采样
- 校准/测试划分：`engine_calib_split(frac=0.3, seed=42)`
- 所有 seed 显式记录在目录名中（`f{fold}_s{seed}`）
