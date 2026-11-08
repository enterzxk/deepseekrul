from pathlib import Path
p=Path('REPRODUCIBILITY.md'); s=p.read_text()
old = '2. 因此**单种子比较不可靠**：seed 101 的 FULL 优势（−3.3%）部分来自它在 f1/f2 上\n   的有利初始化；seed 202 的对应优势只有 −1.9%。\n3. 这与 \\`REPRODUCIBILITY.md\\` 第 1 节的初始化缺陷一致；结论必须以 3 种子均值为准，\n   且应报告逐折范围。'
new = '2. 因此**单折比较不可靠**：FULL 在 f1 上跨种子相差 1.24 循环（约 10%）。\n   但**满 5 折的种子均值是稳定的**：seed 101 −3.38%、seed 202 −3.13%、seed 303 −3.73%\n   （3 折），三种子均值 −3.41%（sd 0.30%）。\n3. 结论：初始化噪声影响**逐折**结果，不影响**多种子合并**结论；因此主表必须报告\n   3 种子均值与逐折范围，单折数字只作诊断。'
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor missing')