from pathlib import Path
p=Path('KEY_FINDINGS.md'); s=p.read_text()
old = '## 1b. FD001 适用范围（100 台，seed 101）'
new = '''## 1a. FD002 三种子（满 5 折，主结论口径）

| 臂 | RMSE 均值 | sd | MAE | NASA | Bias |
|---|---:|---:|---:|---:|---:|
| M0 | 13.65 | 0.13 | 9.441 | 4.095 | +1.198 |
| OPS | 13.38 | —（1 种子） | 9.291 | 3.904 | +0.776 |
| **FULL（参考）** | **13.19** | 0.14 | 9.151 | 3.577 | +0.169 |
| NORM（冻结候选） | 14.59 | 0.11 | 10.29 | 4.633 | +0.918 |

逐种子 FULL−M0：−3.38% / −3.13% / −3.59%，均值 **−3.37%（sd 0.23%）**。
NORM 相对 FULL：**+1.536 循环（−11.8%）**，三种子一致。

## 1b. FD001 适用范围（100 台，seed 101）'''
if old in s and '## 1a.' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')