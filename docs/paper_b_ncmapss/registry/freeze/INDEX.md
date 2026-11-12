# 冻结清单索引（`freeze/`）

一份清单对应一次「开封前声明」。

**不变量：`freeze/*.json` 有且仅有一份**（即当前 live 清单）。作废的清单移到
`freeze/superseded/`，**字节不动**——移动路径不改内容，所以它的摘要照样能按签名时的规则复算。
这条不变量让「哪份算数」不需要解析任何字段，一条 `ls` 就够，闸门也不必去猜 `status`。

| 路径 | 状态 | `freeze_ts_utc` | 哈希规则 | `freeze_content_sha256` | 格数 | 说明 |
|---|---|---|---|---:|---:|---|
| `freeze/NCM-FREEZE-20261112T1955Z.json` | **active** | 2026-11-12T19:55:04Z | **v17** | `476bdaea731098f97ea4b161168bd92d521549d7ba9637ac71ac66da45259cc0` | 40 | 第 14 代锚 `497afe23…` 下的测试批。v17 把 `freeze_ts_utc` 纳入哈希；补 `sequencing` / `completeness` / `epoch_star_values`。 |
| `freeze/superseded/NCM-FREEZE-20261112T1945Z.json` | superseded | 2026-11-12T19:45:00Z | v16 | `3c27009e3f5e9c6f6c456046f8b2a7d4672dfe242cc6a38f44156aec439b3932` | 40 | 未用于跑批。作废原因：`freeze_ts_utc` 不在哈希内，可被改早而不被发现。 |

## 三条规矩

**① 作废不回改内容。** `superseded` 的清单**字节一个都没动**——它的摘要是按当时的规则签的，
往里加一个 `status` 字段就等于让摘要失效。所以作废只体现为**移动目录**，状态记在本索引、
`REGISTRY_NOTES.md` 和继任清单的 `supersedes` 字段里。改历史文件等于让「这份清单当时写了什么」不可考。

**② 摘要按签名时的规则算，不是按现行规则。** 上表 `v16` 那一行只能这样复算：

```python
body = {k: v for k, v in d.items()
        if k not in ("freeze_ts_utc", "freeze_content_sha256", "eval_start_utc")}
sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
```

现行规则（v17）排除的只有 `eval_start_utc` 和哈希自身。规范化形式是
**`ensure_ascii=False` + 紧凑分隔符 `(',', ':')` + `sort_keys=True` + UTF-8**，
闸门和签名器必须用同一个形式——`json.dumps` 默认 `ensure_ascii=True`，一换就会把已提交的清单判成被改过。

```bash
python scripts/nc_freeze.py verify docs/paper_b_ncmapss/registry/freeze/NCM-FREEZE-20261112T1955Z.json
# hash / completeness / epoch_star / ordering 四项，全 OK 才许开跑
```

**③ 清单里为什么要有 `epoch_star_values`。** §6.10 说 `epoch*` 由 dev 六折估计、
**在开封前冻结**。如果这 40 个数不在被哈希的清单里，这句话就没法执行——谁都可以在开封后
按自己的算法重算一遍，而且看起来完全合规。所以这 40 个数在签名时从封存台账机械复算写入，
运行器**读取**它们，不许自己再算；闸门应拒绝「运行器算出的 `epoch*` ≠ 清单里的值」的开封请求。

`floor_statistics` 同批写入，用来复算「撞地板」那条统计：N0 的 24 折里 **14 折**落在 20
（DS02 5/6、DS05 3/6、DS06 4/6、DS07 2/6）。
