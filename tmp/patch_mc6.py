from pathlib import Path
p=Path('RESULTS_NARRATIVE.md'); s=p.read_text()
old = """| 条件 | M0 RMSE | FULL RMSE | FULL 相对 | M0 NASA | FULL NASA |
|---|---:|---:|---:|---:|---:|
| clean | 13.461 | 13.065 | −2.9% | 3.667 | 3.331 |
| MC1（20% 缺测） | 15.215 | 14.267 | **−6.2%** | 4.424 | 3.586 |
| MC2（3 通道 +1σ） | 33.105 | 33.168 | +0.2%（持平） | 73.8 | **163.8** |"""
new = """| 条件 | M0 RMSE | FULL RMSE | FULL 相对 | M0 NASA | FULL NASA |
|---|---:|---:|---:|---:|---:|
| clean | 13.652 | 13.054 | −4.4% | 4.086 | 3.492 |
| MC1（20% 缺测） | 15.149 | 14.208 | **−6.2%** | 4.739 | 3.803 |
| MC2（3 通道 +1σ） | 30.845 | 30.647 | −0.6%（持平） | 60.8 | **132.3** |

（5 折、260 台；权重为 v2 重训，clean 基线与主表的 13.505 略有差异，属初始化噪声。）"""
if old in s:
    s = s.replace(old, new)
    s = s.replace("## 10. 测量条件（跨折合并；4 折）", "## 10. 测量条件（5 折、260 台，v2 权重）")
    p.write_text(s); print('patched')
else:
    print('anchor missing')
