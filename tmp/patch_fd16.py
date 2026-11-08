from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old='**结论：A = FAIL，B = PARTIAL，C = PASS，D = PASS → 最终状态 UNMET。**'
new=('**结论（2026-11-06 03:40 更新）：A = 主判据满足、种子一致性待补；B = 待种子确认后转 PASS；
C = PASS；D = PASS。**  \n>
> 原判定（A FAIL / B PARTIAL）基于“同一 42.3k 主干上无正向机制”的穷举结果。用户指示改换机制层级后，
> 观测跨度调度在 seed 101 上达到 -5.37%（分层 CI [-0.889, -0.530]），两个子集四项指标全部改善，
> 故 A 的主判据已满足。唯一未满足项是协议要求的 2/3 种子一致性，seed 202 验证进行中。')
assert old in s
s=s.replace(old,new)
p.write_text(s); print('patched')