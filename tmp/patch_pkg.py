from pathlib import Path
p=Path('TIM_PAPER_PACKAGE.md'); s=p.read_text()
old = "### 3.2 三种使用方式（ABLATIONS.csv / RESULTS.csv）"
new = """### 3.1b FD004 主表（249 台，49,048 窗口，seed 101）
M0 13.398（MAE 8.810、NASA 8.980、Bias −0.171）；OPS 13.676（8.721、30.644、−0.108）；
FULL 13.253（8.662、11.251、−1.038）。FULL−M0 = +0.145 [−0.240,+0.518]（跨零），
且 FULL 的 NASA 劣于 M0——**既有方法的优势在 FD004 上不是四指标一致的**。

### 3.2 三种使用方式（ABLATIONS.csv / RESULTS.csv）"""
if old in s and '3.1b' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
