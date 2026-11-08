from pathlib import Path
p=Path('RESULTS_NARRATIVE.md'); s=p.read_text()
old = "## 10. 测量条件（2 折合并，FD002 fold 0+2）"
new = """## 10. 测量条件（跨折合并；clean 3 折、MC1/MC2 2 折）

clean（3 折，156 台）：M0 13.243、FULL 12.838（−3.1%）。
MC1（2 折）：M0 15.289、FULL 14.862（−2.8%）。
MC2（2 折）：M0 36.937、FULL 39.692（+7.5%，FULL 更差）。

（以下为 2 折版本的详细读法，3 折 clean 结论与之一致。）"""
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
