from pathlib import Path
p=Path('PROTOCOL.md'); s=p.read_text()
old = '处置：主结论只用 3 种子均值或远大于噪声的效应；A 判定不变；FULL vs M0 的 3.5%\n需由 3 种子均值复核。已完成运行不重跑（时间与成本不允许），修正方案与复算顺序\n记录在 \\`REPRODUCIBILITY.md\\`。'
new = '处置：主结论只用 3 种子均值或远大于噪声的效应；A 判定不变；FULL vs M0 的 3.5%\n已由 3 种子均值复核为 **3.41%（sd 0.30%）**，逐折噪声（FULL 在 f1 上跨种子相差\n1.24 循环）不影响合并结论。已完成运行不重跑（时间与成本不允许），修正方案与\n复算顺序记录在 REPRODUCIBILITY.md。'
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor missing')