# SUPERSEDED — 不要引用本表中的区间

`resampled_metric.csv`（本目录）是**已被取代**的旧表，保留仅为可追溯。

## 为什么被取代

生成它的脚本 `dsh_resampled_metric.py` 有两个已确认缺陷（里程碑 A1）：

1. `load_cut` 直接合并各种子 CSV；`engine_errors` 以 `unit` 分组后只用 `abs_cut` 作键，
   会同键覆盖不同种子——通常只剩最后一个种子；
2. `draw_rmse` 的 `seeds` 参数未被使用，且用列表位置/取模假定不同方法的截断对齐。

因此表中的 `p05/p95` **不是总体 95% 置信区间**，也**不是**“P(更优)=结论正确概率”。

## 用什么代替

* 规范指标：`results/dsh/review_r1/metrics_clipped_round146.csv`（裁剪到 [0,125]）；
* 区间估计：按**发动机聚类**、保留全部 cutoffs/seeds 重算（见 REVIEW §36 与 §88）；
* 跨种子波动单列，不与区间混用。