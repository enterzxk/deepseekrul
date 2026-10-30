# rul_archive — 预测与权重存档（完整版）

满足目标第六条“保存最佳权重、预测”。Round 159 补齐为**服务器上的全部**文件。

| 子目录 | 文件数 |
|---|---:|
| `predictions/` | 57 |
| `weights/` | 38 |

完整性：与服务器 `experiments/*/*/FD002/*/` 下的数量逐一对齐（57 预测 / 38 权重）。
注意 `M4b_popgate` 只有预测、无权重（其运行未留下 best_model.pt）。

## 命名

`{root}__{cell}__{variant}_s{seed}__matched.csv` / `{root}__{cell}__{variant}_s{seed}.pt`

`root` ∈ {order03, split8, split9}。

## 已验证

| 项 | 状态 |
|---|---|
| 预测 → 指标可复现 | **已验证**（Round 158，RMSE 差 ≤0.06，late 逐位一致） |
| 权重大小 vs 参数量 | **已验证**（38/38 吻合） |
| 权重容器格式 | **已验证**（38/38 为合法 ZIP/torch 容器） |
| 权重加载并推理 | **已验证**（Round 161，split8 上 corr 1.000000、max diff 0.062） |

## 完整性校验

`sha256sum -c MANIFEST.sha256`（或 PowerShell `Get-FileHash` 对照）。

## 复算方法

预测 CSV 为 4 列：`unit_id, y_true, y_pred, signed_error`（**不是 5 列**，没有 abs_cut）。
先把 `y_pred` 裁剪到 [0,125]，再算误差；不得先对有符号误差求均值再平方。

Round 158 已用这些文件复算过全部指标：RMSE 与文档差 ≤0.06，late_fraction 逐位一致。