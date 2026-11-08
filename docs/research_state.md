# 研究状态（恢复入口）

> 更新于 2026-11-08。任何新会话/压缩恢复后，**先读本文件**，再读 `decision_log.md`。
> 规则见 `AGENTS.md` §9。

## 1. 当前目标

把 C-MAPSS RUL 保形区间方法做到可投 IEEE TIM：方法成立、证据完整、论文成稿。

## 2. 已完成（可验证）

| 项 | 产物 | 状态 |
|---|---|---|
| 主实验 4 子集 × 5 折 × 2 seeds | `experiments/punet_matrix/` 40 格 | ✅ |
| 追加 seeds 2024 / 13 | `experiments/punet_matrix/*/f*_s{2024,13}/` | ✅ 79/80 |
| 名义水平配对 | 19/20 胜、中位 −8.7% | ✅ |
| 覆盖对齐配对 | 29/40 胜、中位 −8.4% | ✅ |
| 跨域 FEMTO 6 轴承 | `experiments/femto/` | ✅ 5/6、−12.2% |
| A4 置换消融 20 折 | 均值 +0.2% | ✅ |
| 覆盖诊断 20 折 | 引擎级最小 0.50–0.71 | ✅ |
| 论文 Markdown / LaTeX / PDF | `docs/manuscripts/` | ✅ 编译通过 |
| 图 3 张 | `docs/figures/` | ✅ |

## 3. 已排除的方向（不要重复）

| 方向 | 结论 | 证据文件 |
|---|---|---|
| 上下文中心修正（D002） | 不可跨引擎迁移，覆盖被破坏 | `docs/tables/d002_centre.csv` |
| 宽度放大（D001） | a=1 已最优 | `docs/tables/width_scaling.csv` |

| 方向 | 结论 | 证据文件 |
|---|---|---|
| 非对称校准 | 最优=对称，无收益 | `docs/tables/asymmetric.csv` |
| 分层校准（mu / sigma 分箱） | 全部更差 | `docs/tables/mondrian.csv` |
| 引擎级校准 | 等价于放宽区间 | `docs/tables/engine_vs_inflation.csv` |
| seed 集成 | n=2 有效、n=4 退化 | `docs/tables/ensemble_scaling.csv` |
| 局部化 CQR | 40 格上无收益，FD003 更差 | `docs/tables/cells_all.csv` |

## 3.5 论文主张（D003 后冻结）

正面、限定范围的效率主张：
- 名义水平 19/20 胜、中位 −8.7%
- 覆盖对齐 29/40 胜、中位 −8.4%
- 增益集中于 FD001/FD002；FD003/FD004 接近打平（作为**适用范围**陈述，不作为负结果）
- 跨域 FEMTO 5/6、−12.2%
- 不主张"均匀改善"，不主张"方法学新颖性"

构建命令：`powershell -NoProfile -File scripts/build_manuscript.ps1`（含 7 章结构断言）

## 4. 待执行

1. 等待 Codex 对 D001 的决策，按其结论执行
2. 补最后 1 格（FD004 f4 s13）
3. 若决定改损失：重训 40 格 × 1 seed
4. 论文按最终结果更新并重编译

## 5. 恢复命令

```bash
ssh a40 'cd ~/deepseek-rul && ls experiments/punet_matrix/*/f*_s*/preds_point.csv | wc -l'
python scripts/table_all_methods.py
python scripts/make_tim_tables.py
```

## 6. 环境事实

- 远端 `a40`：无外网、无 git；Python `/home/yndx/miniconda3/envs/a40_env/bin/python`
- 本地：无 torch，只有 Windows + pandoc + tectonic（LaTeX 编译）
- 数据集：C-MAPSS 在 `data/`；FEMTO 在远端 `~/femto`（3.1 GB）

## 7. 未验证/待确认

- 桌面 profile 的 `reasoningEffort` 是 `high` 而非约定的 `max`（见 AGENTS.md §2）
- 论文参考文献仍是占位 [1]–[15]，无 BibTeX

## 8. 长期自主科研配置（2026-11-08 第二轮）

- AGENTS.md 新增 §11–§17：科研使命与 Q1 标准、AUTO_PROCEED、子代理编队、GPU 政策、
  证据与 Wiki、不可改动边界、AGENT_GUIDE 缺失记录。
- 新建 ARIS Research Wiki：docs/wiki/{index,literature,hypotheses,experiments,failures,progress}.md
- 实测：ARIS 官方 AGENT_GUIDE.md 本机不存在；未虚构引用，见 AGENTS.md §17。
- 本轮未启动任何实验。

**恢复入口更新**：新会话读 research_state.md → decision_log.md → results_summary.md → docs/wiki/index.md。

