from pathlib import Path
p=Path('TIM_PAPER_PACKAGE.md'); s=p.read_text()
old = "### 3.3 可辨识性（IDENTIFIABILITY.md）"
new = """### 3.2b 统计功效与指标尾部（RESULTS_NARRATIVE.md §8、§11）
- FD002（260 台）FULL−M0 的 95% 区间半宽 0.376 循环 ≈ 2.9% 相对效应；3% 量级
  处于可检测边缘。FD004（249 台）中心值仅 1.1%，无法与零区分。
- NASA 分数在 FD004 上被 1–5 台发动机主导（OPS 的 90% 来自 5 台，单台 5256），
  因此 NASA 排序在小机队上不稳定，必须同时报告尾部集中度。
- 测量条件跨折（2 折）：MC1 下 FULL 仍更好（−2.8%），**MC2（1σ 通道偏移）下
  FULL 反而更差（+7.5%）**——先验注入在标定偏差下放大误差。

### 3.3 可辨识性（IDENTIFIABILITY.md）"""
if old in s and '3.2b' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
