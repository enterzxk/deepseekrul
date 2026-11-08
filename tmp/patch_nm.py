from pathlib import Path
p=Path('NOVELTY_MATRIX.md'); s=p.read_text()
s=s.replace("只有\"加性注入\"带来小幅收益（FD002 外层验证 3.1%）","只有\"加性注入\"带来小幅收益（FD002 外层验证 3.3%，三种子 3.41%）")
s=s.replace("外层验证 2.9% 的融合增益在官方测试上消失","外层验证 3.0% 的融合增益在官方测试上消失")
p.write_text(s); print('patched')
