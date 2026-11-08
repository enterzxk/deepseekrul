from pathlib import Path
p=Path('RESULTS_NARRATIVE.md'); s=p.read_text()
old_start = s.find("## 10. 测量条件")
old_end = s.find("## 11. 统计功效")
if old_start != -1 and old_end != -1:
    new = """## 10. 测量条件（跨折合并；clean/MC1 4 折、MC2 3 折）

| 条件 | M0 RMSE | FULL RMSE | FULL 相对 | M0 NASA | FULL NASA |
|---|---:|---:|---:|---:|---:|
| clean | 13.461 | 13.065 | −2.9% | 3.667 | 3.331 |
| MC1（20% 缺测） | 15.215 | 14.267 | **−6.2%** | 4.424 | 3.586 |
| MC2（3 通道 +1σ） | 35.976 | 35.951 | −0.1%（持平） | 95.2 | **215.0** |

读法（4 折版本，取代此前的 2 折结论）：
1. **缺测条件下 FULL 的相对优势从 2.9% 扩大到 6.2%**——先验注入提供了一条
   不依赖完整窗口的信息通路，这是 C 项最有价值的正向结果。
2. 1σ 通道偏移下 RMSE 基本持平（−0.1%），但 **FULL 的 NASA 分数是 M0 的 2.3 倍**
   （215 vs 95），说明偏移条件下 FULL 的尾部高估更严重；RMSE 持平掩盖了这一点。
3. 2 折时曾观察到 FULL 在 MC2 下 +7.5% 更差，4 折后该结论不成立——**跨折结论
   必须等折数足够**，这也是本项目对自身早期结论的修正记录。

"""
    s = s[:old_start] + new + s[old_end:]
    p.write_text(s); print('replaced')
else:
    print('anchors not found')
