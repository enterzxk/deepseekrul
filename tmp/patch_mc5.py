from pathlib import Path
p=Path('RESULTS_NARRATIVE.md'); s=p.read_text()
s=s.replace("| MC2（3 通道 +1σ） | 35.976 | 35.951 | −0.1%（持平） | 95.2 | **215.0** |",
            "| MC2（3 通道 +1σ） | 33.105 | 33.168 | +0.2%（持平） | 73.8 | **163.8** |")
s=s.replace("## 10. 测量条件（跨折合并；clean/MC1 4 折、MC2 3 折）",
            "## 10. 测量条件（跨折合并；4 折）")
p.write_text(s); print('patched')
