from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
s=s.replace("| 四指标 + 区间 | RESULTS.csv（FD002/FD004 全表） | PASS |",
            "| 四指标 + 区间 | RESULTS.csv（四子集全表，11 行） | PASS |")
s=s.replace("| 3 种子 | FD002 202/303 运行中；FD004 仅 seed 101 | **PARTIAL** |",
            "| 3 种子 | FD002 seed 202（5 折 M0 / 4 折 FULL）、303（3 折 M0 / 2 折 FULL）运行中；FD001/FD003/FD004 仅 seed 101 | **PARTIAL** |")
p.write_text(s); print('patched')
