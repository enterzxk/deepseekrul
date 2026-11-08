from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
s=s.replace("| 3 种子 | FD002 seed 202（5 折 M0 / 4 折 FULL）、303（3 折 M0 / 2 折 FULL）运行中；FD001/FD003/FD004 仅 seed 101 | **PARTIAL** |",
            "| 3 种子 | FD002 seed 101/202 满 5 折（−3.38% / −3.13%），303 为 3 折（−3.73%）；FD001/FD003/FD004 仅 seed 101 | **PARTIAL（接近完成）** |")
p.write_text(s); print('patched')
