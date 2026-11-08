from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = '## 最终状态'
new = '''## F. 实验完成度（2026-11-05 23:34）

所有队列任务已完成（Q2/Q3/Q5 COMPLETE，无残留训练进程）：
- FD002：M0/OPS/FULL/NORM 全部 5 折，其中 M0/FULL/NORM 为 3 种子；
- FD001/FD003：M0/FULL 各 5 折（seed 101）；
- FD004：M0/OPS/FULL/NORM 各 5 折（seed 101）；
- 测量条件：clean/MC1/MC2 各 5 折；
- 交付：12 行主表、90 个逐窗口预测、6 图 6 表、194 项 MANIFEST。

## 最终状态'''
if old in s and '## F. 实验完成度' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')