from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = "轻量预算下 NORM 相对 M0 为 −4.67%（FD002，3 种子），完整预算下 +7.7%——方向相反。"
new = """两个满 5 折种子均显示 NORM 显著更差：seed 101 +7.6%（14.523 vs 13.495）、
seed 202 +6.9%（14.699 vs 13.751），均值 +7.25%。冻结候选的失败在种子上稳定。

轻量预算下 NORM 相对 M0 为 −4.67%（FD002，3 种子），完整预算下 +7.25%——方向相反。"""
assert old in s
s = s.replace(old, new)
p.write_text(s); print('patched')
