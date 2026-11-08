from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = "| 测量条件实验 | MC1 +14%、MC2 +142%、MC3 平坦（FD002 fold 2 / 全折） | PASS（FULL 臂与更多折待补） |"
new = ("| 测量条件实验 | MC1 +14%、MC2 +142%、MC3 平坦；FULL 在 clean/MC1/MC2 下均优于 M0 且缺测下优势扩大（−5.3% → −7.3%） | PASS（1 折，更多折排队） |")
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor not found')
