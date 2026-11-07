# C-MAPSS 数据清单

数据来源：本地下载的 `6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`，其中包含 NASA C-MAPSS 原始 `CMAPSSData.zip`。

服务器路径：`/home/yndx/zxk/bearing-rul-pi/data/CMAPSS`

## SHA256

```text
a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca  RUL_FD001.txt
c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b  RUL_FD002.txt
df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb  RUL_FD003.txt
196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1  RUL_FD004.txt
3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851  test_FD001.txt
de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02  test_FD002.txt
299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790  test_FD003.txt
1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5  test_FD004.txt
963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8  train_FD001.txt
dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f  train_FD002.txt
2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2  train_FD003.txt
27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218  train_FD004.txt
```

## 预处理协议

- 输入仅使用标准 14 个传感器：2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21。
- Min-Max 统计量只由训练集拟合，测试集只应用训练统计量。
- 训练目标为分段 RUL，平台上限 125。
- 训练窗口长度 30，步长 1。
- 测试每台发动机使用最后 30 周期；不足 30 时在左侧复制首帧。
- 测试真值默认使用 `RUL_FDxxx.txt` 原值；如实验协议显式要求，可设置 `clip_test_rul=True`。

---

## V4 实验产物清单（2026-11-08）

代码（服务器 `/home/user/rulb`）：

| 文件 | 作用 |
|---|---|
| `src/candidates/dspt.py` | 跨度监督 Transformer（`long`/`short`/`dual`/`slot`/`repeat` 五种池化模式） |
| `scripts/run_dspt.py` | 训练入口，`--mode` 选择池化臂 |
| `scripts/eval_dspt.py` | 官方 test 评估（重建折内 scaler） |
| `eval_matrix.py` | 全矩阵评估，单模型与集成分列 |
| `d1_eval.py` | D1 筛选分析 |
| `d5_abl.py` | D5 消融（四种池化臂） |
| `d6_stats.py` | D6 引擎聚类 bootstrap + 分箱误差 |
| `mk_tab4.py` / `mk_figs4.py` | 出表出图 |
| `q_lanes.sh` / `q_matrix.sh` | GPU 绑定的双 lane 调度 |

产物：

| 路径 | 内容 |
|---|---|
| `experiments/M_FD00[1-4]_s{101,202,303}_f{0-4}/` | 60 个训练模型（`.pt` + `.csv` + `.json`） |
| `experiments/TESTCACHE/FD00x_s*_f*.csv` | 60 份官方 test 逐发动机预测 |
| `RESULTS_V4.csv` | 四子集单模型与集成指标 |
| `paper_figures_v4/` | 三张图（PDF + PNG） |
| `paper_tables_v4/main.tex` | 主结果表 |
| `tim_20261105/full_20261105/manuscript_v4.{tex,pdf}` | 论文（4 页） |
| `tim_20261105/full_20261105/DECISION_V4.md` | 决策记录（含 D1–D6 结果） |
| `tim_20261105/full_20261105/PROTOCOL_V4.md` | 冻结协议 |

训练配置：1 层 Transformer，d=64，78,978 参数，AdamW，80 epoch 上限，patience 20（按折内验证 RMSE 早停），窗口 100，RUL 上限 125。

官方 test 只用于最终评估，未参与任何模型选择。

Git 提交：`eb19ac0`（分支 `punet-interval`）。
