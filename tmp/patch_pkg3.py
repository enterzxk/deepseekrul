from pathlib import Path
p=Path('TIM_PAPER_PACKAGE.md'); s=p.read_text()
old = "### 3.2b 统计功效与指标尾部"
new = """### 3.2a 三种子复核（FD002）
seed 101 −3.38%（5 折）、202 −2.90%（4 折）、303 −3.73%（3 折）；
均值 **−3.34%（sd 0.41%）**。方向一致、幅度稳定，说明 FULL 相对 M0 的优势是真实的，
不是单种子初始化运气。但折数不同，全 5 折后需重算。注意这不改变 A 判定——A 比较的是
新机制 NORM（−11.3%/−11.6%）对参考。

### 3.2b 统计功效与指标尾部"""
if old in s and '3.2a' not in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('skip')
