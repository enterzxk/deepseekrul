# C-MAPSS RUL 项目交接文档

更新时间：2026-10-21  
项目根目录：/home/yndx/zxk/bearing-rul-pi  
本地镜像：C:\Users\20562\Desktop\deepseek-rul

## 0. 一句话状态

项目已经完成 P0–P5 的真实实验，包括 FD001 五 seed 确认性消融和 FD002 三 seed 外部验证；但**尚未完成**最初 7-Stage 任务书中的完整 5 基线矩阵、5-Fold Ensemble、不确定性、噪声鲁棒性、少样本迁移，以及基于最新数据的论文重写。

当前最重要的结论不是“Full Model 全场最佳”，而是：

- FD001 上 Full 的 RMSE/MAE 最好。
- FD002 上 Full 明显弱于 matched LSTM。
- A-GCN 是目前最稳定的正向模块。
- Mono loss 改善 RMSE，但恶化 NASA Score 和晚报率。
- 官方测试集已经在 P3/P4 中被观察，后续不得再用于模型选择。

---

## 1. 环境与路径

服务器：

~~~text
/home/yndx/zxk/bearing-rul-pi
~~~

执行环境：

- Python：/home/yndx/miniconda3/envs/a40_env/bin/python
- PyTorch：2.5.1+cu121
- GPU：NVIDIA A40
- sklearn 需要 LD_LIBRARY_PATH：
  /home/yndx/miniconda3/envs/a40_env/lib

数据：

~~~text
/home/yndx/zxk/bearing-rul-pi/data/CMAPSS
~~~

Git：

- 当前项目没有 .git。
- 远程 PAT 不包含原生 git 可执行文件。
- 交接依赖文件、冻结哈希和结果目录，不依赖 Git 历史。

测试文件当前权限：

~~~text
test_FD001~FD004.txt    000
RUL_FD001~FD004.txt     000
~~~

注意：权限冻结只防止继续误读，不能撤销 P3/P4 已经完成的测试集观察。

---

## 2. 阅读顺序与权威性

### 第一优先级：最新真实结果

1. MORNING_REPORT.md
2. results/final_FD001_5seeds/
3. results/final_FD002_3seeds/
4. results/final_tables.tex
5. results/checkpoint_status.json
6. docs/FINAL_PROTOCOL.md

### 第二优先级：历史与实现细节

1. docs/STAGE3_REPORT.md
2. results/final_FD001_prelim/
3. results/validation_p1_p2_20261020/
4. src/model.py
5. src/loss.py
6. src/dataset.py
7. scripts/run_ablation.py
8. scripts/summarize_ablations.py

### 已过期、不要直接引用

以下文件生成时间早于最新 P3/P4：

- MORNING_BRIEFING.md
- results/FINAL_PAPER_REPORT.md
- paper/main.tex
- docs/GEMINI_REVIEW_BRIEF_ZH.md
- results/overnight_progress.log

它们仍然保留作为历史记录，但其中的 14.364 门禁失败结论、缺失实验列表和论文表格已经不是最新状态。

docs/STAGE3_REPORT.md 自身也注明：文件中的 P3 表是早期 d0.2 pilot，最新数字以 MORNING_REPORT.md 为准。

---

## 3. 真实实验状态

checkpoint_status.json 记录：

- P0：complete
- P1：complete
- P2：complete
- P3：complete
- P4：complete
- P5：complete
- 模式：unattended_overnight
- 开始：2026-10-21 01:00
- 结束：2026-10-21 04:50

这里的 P5 只表示图表和最终表格生成完成，不等价于最初 7 个 Stage 全部完成。

---

## 4. 冻结配置

来源：results/locked_hparams.json

~~~json
{
  "window_size": 50,
  "dropout": 0.15,
  "weight_decay": 0.0001,
  "d_model": 64,
  "batch_size": 256,
  "lr": 0.001,
  "epochs": 80,
  "patience": 20,
  "seeds": [42, 67, 123],
  "loss": {
    "asym_weight": 1.0,
    "lambda_mono": 0.05,
    "late_weight": 2.0
  }
}
~~~

P1 验证结果：

- val RMSE：11.917 ± 0.534
- val NASA：7659.008

P2 loss 选择：

- 主模型：MSE + Asym(weight=1.0) + Mono(lambda=0.05)
- w/o Mono：MSE + Asym(weight=1.0)
- w/o Physics：MSE

---

## 5. FD001 五 seed 确认性消融（clip125）

来源：results/final_FD001_5seeds/summary.csv

| Variant | RMSE | NASA | MAE | Late | Severe>10 | MonoViol | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o A-GCN | 14.104 ± 1.200 | 369.635 ± 102.242 | 10.352 | 0.530 | 24.6 | 0.465 | 143,816 |
| Identity-Graph (A=I) | 14.689 ± 0.683 | 392.977 ± 86.252 | 10.583 | 0.504 | 21.8 | 0.463 | 143,816 |
| w/o SSM-Block (matched LSTM) | 14.405 ± 0.329 | 359.656 ± 22.626 | 10.379 | 0.468 | 18.6 | 0.313 | 142,806 |
| w/o Physics Loss | 14.123 ± 0.868 | 357.238 ± 56.818 | 10.194 | 0.510 | 21.6 | 0.451 | 144,265 |
| w/o Mono Loss | 14.133 ± 0.538 | 324.621 ± 40.663 | 10.246 | 0.450 | 17.8 | 0.446 | 144,265 |
| Full Model | 13.423 ± 0.487 | 327.119 ± 59.184 | 9.719 | 0.522 | 18.6 | 0.453 | 144,265 |

解释：

- Full 在 RMSE 和 MAE 上最好。
- Full 不是 NASA 最优；w/o Mono 略好。
- 去掉 A-GCN：RMSE +0.682，NASA +42.516。
- 固定 Identity graph：RMSE +1.267，NASA +65.859。
- A-GCN 是目前最强、最一致的正向证据。
- 去掉 SSM 换 matched LSTM：RMSE +0.982，NASA +32.538，但 LSTM 方差更低。
- 去掉 physics：RMSE +0.700，NASA +30.119。
- Mono loss 改善 RMSE，但让 late fraction 从 0.450 升到 0.522，NASA 略差。

### Raw-label appendix

| Variant | RMSE | NASA |
|---|---:|---:|
| w/o A-GCN | 15.122 ± 1.053 | 395.962 ± 99.756 |
| Identity-Graph | 15.667 ± 0.534 | 423.210 ± 83.312 |
| w/o SSM | 15.428 ± 0.355 | 389.163 ± 21.521 |
| w/o Physics | 15.086 ± 0.938 | 384.123 ± 53.159 |
| w/o Mono | 15.321 ± 0.633 | 360.064 ± 37.463 |
| Full | 14.456 ± 0.468 | 354.360 ± 55.360 |

主协议是 clip125；raw-label 只能作为附录。

---

## 6. FD002 三 seed 外部验证（clip125）

来源：results/final_FD002_3seeds/summary.csv

| Variant | RMSE | NASA | MAE | Late | Severe>10 | Params |
|---|---:|---:|---:|---:|---:|---:|
| w/o A-GCN | 16.126 ± 1.037 | 1385.647 ± 88.676 | 11.631 | 0.463 | 52.3 | 143,816 |
| w/o SSM (matched LSTM) | 14.363 ± 1.059 | 1001.130 ± 248.916 | 10.538 | 0.422 | 43.3 | 142,806 |
| w/o Physics | 15.512 ± 0.784 | 1513.909 ± 215.770 | 10.958 | 0.528 | 59.0 | 144,265 |
| Full | 14.955 ± 0.715 | 1456.569 ± 248.479 | 10.580 | 0.474 | 53.3 | 144,265 |

FD002 关键发现：

- A-GCN 仍有正向作用。
- SSM 结论反转：matched LSTM 明显优于 Full。
- Full 的 RMSE 和 NASA 都不是最优。
- raw-label 误差很高：Full 约 26.568 RMSE，不适合直接主表。

---

## 7. 当前科学结论（谨慎版）

可以支持：

- 自适应图结构对跨发动机泛化有帮助，至少在 FD001 和 FD002 上方向一致。
- SSM 在 FD001 上优于 matched LSTM，但在 FD002 上不成立，不能写成普遍优势。
- 非对称物理损失在 FD001 上有中等正向作用。
- Full 是 FD001 的 RMSE/MAE 最优模型，但不是所有指标的全面最优。

不能支持：

- Full 在所有数据集、所有指标上全面最佳。
- SSM 普遍优于 LSTM。
- Mono loss 提升安全性；数据显示它主要提升精度，恶化 NASA 和晚报率。
- FD002 raw-label 已达到可投稿水平。
- 5-Fold Ensemble 已完成；当前最新结果更像五个独立 seed，而不是明确记录的 5-Fold Ensemble。
- 外部 5 基线矩阵已经完整完成；目前未见完整 baseline 结果表。

---

## 8. 已生成资产

最新预测/指标：

~~~text
results/final_FD001_5seeds/
results/final_FD002_3seeds/
results/final_tables.tex
results/locked_hparams.json
results/checkpoint_status.json
~~~

最新图表：

~~~text
figures/final/fig1_fd001_sorted_rul.pdf/png
figures/final/fig2_engine_trajectories.pdf/png
figures/final/fig3_learned_adjacency_comparison.pdf/png
figures/final/fig4_ablation_metrics_bar.pdf/png
~~~

早期图表：

~~~text
figures/fig1_sorted_rul_prediction.*
figures/fig2_engine_lifecycle_tracking.*
figures/fig3_sensor_graph_heatmap.*
figures/fig4_ablation_comparison.*
~~~

这些早期图可能来自 pilot，引用前必须核对生成时间和数据来源。

代码冻结哈希（当前文件已核对）：

~~~text
b5ef67df7c440fdf4b7a28f1fb3e232eb4e356d08e95612a8edc58f04430bbbd  src/model.py
94a3b8e293cd925e6c56aed575aad55583ffb7f212dfa60326f22be525d096bf  src/loss.py
f7d84e4e10dee333f813635da8c7ae2338acec2f15274c8f950bab446f4da2b5  src/dataset.py
8c61ec632beb0f67bf5baf8e603514a445e3c38fdefcf04d68ff15e0191b096e  scripts/run_ablation.py
4bbb919f2c04a2a09f0df81b17d1e2938ecc250b382788cb0d1c6df2c4209c31  scripts/summarize_ablations.py
~~~

---

## 9. 代码地图

### src/dataset.py

- FD001–FD004 读取
- 14 个标准传感器
- MinMax / Standard scaler
- 6 工况 KMeans + 簇内 Z-score
- 目标缩放
- 发动机级划分
- 训练/测试窗口

### src/model.py

- Lite_STG_Mamba
- A-GCN / identity / random-fixed 图模式
- 双向多尺度 SSM
- delta 特征
- 参数匹配 LSTM control
- 轨迹输出和 MC-Dropout 支持

### src/loss.py

- PhysicsGuidedLoss
- StateAwarePhysicsLoss
- L_Mono
- 复合验证指标
- 安全/单调性诊断

### src/baselines.py

- 1D-CNN
- Bi-LSTM
- GRU-Attention
- Vanilla Transformer
- TCN

### scripts

- run_ablation.py：锁定配置消融
- summarize_ablations.py：多种子汇总和配对差值
- lock_hparams.py：冻结配置
- run_p3_locked.py：P3 确认性运行
- run_fd002.py：FD002
- generate_final_tables.py
- generate_final_figures.py
- train_dev_fold.py
- run_5fold_dev.py
- final_blind_eval.py
- statistical_tests.py

---

## 10. 已知问题与风险

1. 官方测试集已经被 P3/P4 使用过，不能再根据测试分数改模型。
2. FD002 上 SSM 结论反转，论文不能把 SSM 写成普遍有效。
3. Mono loss 与 NASA/晚报率存在冲突，不能宣称全面提升安全。
4. P3 是五个 seed；没有看到明确的 5-Fold Ensemble 产物。
5. 外部 5 基线完整矩阵尚未看到。
6. 不确定性、噪声/传感器缺失、少样本迁移实验尚未看到最终结果。
7. 论文 main.tex 是旧版，未包含最新 P3/P4 表。
8. 多处旧报告与最新结果并存，必须按本文档的权威顺序阅读。
9. 项目没有 Git，重大改动前应先人工备份 src、scripts、results、figures。
10. 共享 GPU 会造成明显吞吐波动，训练脚本需要断点续跑和日志。

---

## 11. 下一位执行者的优先级

### 第一优先：整理证据而不是继续训练

- 核对 final_FD001_5seeds 与 final_FD002_3seeds 的 config、split、seed、checkpoint。
- 确认五个 seed 的数据划分是否独立；若是固定 split，需要说明不是 5-Fold Ensemble。
- 重新生成论文表格和摘要，淘汰旧版 main.tex。

### 第二优先：补齐审稿必需的比较

- 在相同数据和划分下跑 5 个外部基线。
- 若计算允许，补真正的 5-Fold Ensemble。
- 用配对 Wilcoxon 对 per-engine 误差做检验。

### 第三优先：决定论文故事

推荐候选故事：

> A-GCN 提供跨发动机的稳定空间耦合；SSM 在单工况下有优势但在多工况下存在边界；非对称损失提升 FD001 精度但 mono 项存在安全权衡。

必须诚实报告：

- FD002 上 LSTM 更好。
- Full 不是 NASA 最优。
- Mono 的副作用。

### 第四优先：补可靠性与泛化

- MC-Dropout / 异方差不确定性
- PICP / MPIW / safe bound
- SNR 噪声和传感器缺失
- FD002 → FD001 少样本迁移

---

## 12. 快速复现命令

运行测试：

~~~bash
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python -m unittest discover -s tests -v
~~~

注意：测试文件现在 mode 000，依赖官方 test 文件的测试会 skip 或失败。开发阶段不应恢复权限。

查看最新摘要：

~~~bash
cat MORNING_REPORT.md
cat results/final_FD001_5seeds/summary.csv
cat results/final_FD002_3seeds/summary.csv
~~~

---

## 13. 完整性声明

- 本交接文档只引用磁盘上真实存在的 CSV/JSON/日志和已经核对的源码哈希。
- 没有把期望结果写成实验结论。
- 没有把缺失实验写成已完成。
- 当前测试文件仍为 mode 000。
- 任何后续论文数字都应从 results/final_* 重新生成，而不是手工抄写。
