from pathlib import Path
p=Path('KEY_FINDINGS.md'); s=p.read_text()
old = "## 2. 消融"
new = """## 1b. FD001 适用范围（100 台，seed 101）

M0 RMSE 10.274 / MAE 7.240 / NASA 1.86 / Bias +0.80；
FULL 10.235 / 7.190 / 1.78 / −0.33；相对 −0.38% / −0.70% / −4.34%。
三项均在协议允许退化内（RMSE/MAE ≤ +1.0%、NASA ≤ +2.0%）→ **FD001 护栏 PASS**。
逐折方向不一致（f3 +3.2%、f4 +2.2%）。

## 2. 消融"""
if old in s and '1b.' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
