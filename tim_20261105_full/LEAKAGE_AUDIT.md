# 泄漏审计（2026-11-05 20:05）

目的：确认外层验证的每一折中，验证发动机的标签、未来观测与测试标签都没有进入
推理输入或任何拟合过程。审计对象是 `run_full_fold.py` / `run_full_fold2.py` /
`run_full_fold3.py` 及其调用的 `scripts/run_ida.py`、`src/dataset.py`、
`src/candidates/model.py`。

## 1. 数据划分
- `ida.outer_folds(subset)`：对发动机编号做 `KFold(5, shuffle=True,
  random_state=20261104)`，训练/验证发动机集合不相交。
- 所有逐窗口样本由发动机整段轨迹切片而来，切片不跨发动机。

## 2. 归一化与工况处理
- `ida.prep_fold` 只用 `tr_df`（该折训练发动机）拟合
  `OperatingConditionScaler`（KMeans 6 簇 + 簇内 Z-score）与 KMeans 标签，
  再用同一对象 transform 验证发动机。
- `pop_ref[m]`（各工况簇的总体参考态）只用训练发动机中 cycle ≤ 15 的行计算。
- 传感器 MinMax（FD001/FD003）同样只在训练发动机上 fit。

## 3. 个体早期参考态（extras[:, :14]）
- `build_windows_condref_meta` 对每个窗口末端 t 只取
  `cycles <= 30 且 clusters == m 且 index <= t` 的行，故只用 1..t 的观测。
- 收缩系数 tau=2 固定，不依赖标签。

## 4. 寿命先验（μ_T）
- `run_fold.prior_tables`：外层用该折训练发动机拟合，内层 4 折（按发动机）
  交叉拟合给训练窗口提供折外预测；验证发动机的 μ_T 由**只用训练发动机**拟合的
  模型给出。
- `prior_values` 的阶梯构造：窗口末端 t 只消费 `min(t//10*10, 200) ≤ t` 的
  cutoff，绝不用未来插值。
- 训练窗口的 μ_T 来自内层折外预测（不含该发动机自身），无自泄漏。

## 5. 标签
- 训练标签 `RUL = min(125, T − t)`，T 为该发动机整段轨迹长度（训练标签，允许）。
- 验证标签同样用验证发动机的 T（评价需要），但**不进入任何输入**。
- 官方测试只在 `final_test.py` / `score_*.py` 中读取，且是在预测封存之后。

## 6. 结论
在本审计范围内未发现泄漏。需要说明的两点边界：
1. 外层验证发动机的 T 在数据文件中可见（完整轨迹），但只用于计算标签；
   推理输入不含 T。
2. 官方测试集已被本项目历史使用，本轮不构成盲测。

## 7. 复算
```bash
PY=/home/user/miniconda3/envs/pmv/bin/python
cd /home/user/rulb/tim_20261105/full_20261105
$PY run_full_fold.py FD002 0 --seed 101 --arms M0   # 单折可复现
$PY tests/../  # 项目自带 tests/ 见 /home/user/rulb/tests
```

## 8. 自动化因果性测试（2026-11-05 20:10 实跑）

\`/home/user/rulb/tests/test_prior_causality.py\` 的四项断言全部通过：

\`\`\`
PASS test_t1_future_sensors_cannot_change_prior
PASS test_t2_and_t3_val_label_cannot_leak
PASS test_t4_staircase_cutoff_never_after_window_end
PASS test_unit_mse_hand_computed
ALL PRIOR CAUSALITY TESTS PASSED
\`\`\`

说明：这四项测试针对旧 \`nested_prior\` 实现；本轮 \`run_fold.prior_tables\` 复用了
同样的阶梯构造与按发动机的内层折，测试通过说明该构造的因果性约定成立。
