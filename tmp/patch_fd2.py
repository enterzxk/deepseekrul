from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
add = '''
---

## E. 可复现性披露（不影响 A/B/C/D 判定，但必须随文披露）

运行器在 \`ida.set_seed\` 之前构建模型，PyTorch 默认 RNG 在进程启动时非确定，
因此**同一命令重跑不会得到完全相同的结果**；同一 run 内不同臂的初始化也不同。
实测：FD002 fold 2 的 M0 两次运行为 13.73 与 13.31（约 3%），与本研究要检验的
效应量同量级。

处置与影响：
- 主结论只用 3 种子均值或远大于噪声的效应（NORM +7.7%、PHASE +28% 稳健）；
- **FULL vs M0 的 3.5% 需由 3 种子均值复核**（202/303 运行中）；
- 测量学结论（可观测性 0.26→0.92、可辨识性 |ρ|<0.08、端点排序反转）不依赖
  3% 量级差异，不受影响；
- 精确复现需按 \`tasks_all.txt\` / \`logs/q2.log\` 记录的臂顺序在同一进程内运行；
  修正方案（先 set_seed 再建模型）记录在 \`REPRODUCIBILITY.md\`。

'''
if 'E. 可复现性披露' not in s:
    s = s.replace('## 最终状态', add + '## 最终状态')
    p.write_text(s); print('patched')
else:
    print('already')
