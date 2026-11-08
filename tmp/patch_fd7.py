from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
s=s.replace("| FD002 | 4 | 14.513 | 13.054 | 显著更差（+1.034 [+0.595,+1.468] vs M0） |",
            "| FD002 | 4 | 14.523 | 13.054 | 显著更差（+1.034 [+0.595,+1.468] vs M0） |")
s=s.replace("G_FD002 = 1 − 14.513/13.054 = **−11.2%**（4 折）",
            "G_FD002 = 1 − 14.523/13.054 = **−11.3%**（4 折）")
p.write_text(s); print('patched')
