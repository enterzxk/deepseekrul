
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
