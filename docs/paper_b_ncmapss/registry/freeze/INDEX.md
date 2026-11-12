# 冻结清单索引（`freeze/`）

一份清单对应一次「开封前声明」。

**不变量：`freeze/*.json` 有且仅有一份**（即当前 live 清单）。作废的清单移到
`freeze/superseded/`，**字节不动**——移动路径不改内容，所以它的摘要照样能按签名时的规则复算。
这条不变量让「哪份算数」不需要解析任何字段，一条 `ls` 就够，闸门也不必去猜 `status`。

| 路径 | 状态 | `freeze_ts_utc` | 哈希规则 | `freeze_content_sha256` | 格数 | 协议锚 |
|---|---|---|---|---:|---:|---|
| `freeze/NCM-FREEZE-20261112T2001Z.json` | **active** | 2026-11-12T20:01:43Z | **v17** | `5c9b15a5ef443b362ba1a556849bb904546a499d5d7d06d61a4ebe82c024af7b` | 40 | 第 **15** 代 `7e4fc04` / `6ec1c014…` |
| `freeze/superseded/NCM-FREEZE-20261112T1955Z.json` | superseded | 2026-11-12T19:55:04Z | v17 | `476bdaea731098f97ea4b161168bd92d521549d7ba9637ac71ac66da45259cc0` | 40 | 第 14 代 `5a6b1e7` / `497afe23…` |
| `freeze/superseded/NCM-FREEZE-20261112T1945Z.json` | superseded | 2026-11-12T19:45:00Z | v16 | `3c27009e3f5e9c6f6c456046f8b2a7d4672dfe242cc6a38f44156aec439b3932` | 40 | 第 14 代 `5a6b1e7` / `497afe23…` |

两份作废的清单都**没用于跑批**。逐条作废原因写在继任清单的 `supersedes` 字段里：

- `1945Z` → `1955Z`：`freeze_ts_utc` 不在哈希内，可被改早而不被发现（v16 的洞）。
- `1955Z` → `2001Z`：绑到第 15 代锚。14 代 §6.10 的「N0 撞地板」计数写错（`20/24` 应为
  `14/24`），正文已改，清单的 `protocol_blob` 必须跟着换——否则运行器的闸门会拿清单的 blob
  与本次核到的锚比，对不上就拒（这是它该做的）。
  **40 格 `epoch_star_values`、格清单、数据哈希、特征白名单、训练计划、分析规则逐字节未变。**

## 四条规矩

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
python scripts/nc_freeze.py verify docs/paper_b_ncmapss/registry/freeze/NCM-FREEZE-20261112T2001Z.json
# hash / completeness / epoch_star / ordering 四项，全 OK 才许开跑
```

**③ 清单里为什么要有 `epoch_star_values`。** §6.10 说 `epoch*` 由 dev 六折估计、
**在开封前冻结**。如果这 40 个数不在被哈希的清单里，这句话就没法执行——谁都可以在开封后
按自己的算法重算一遍，而且看起来完全合规。所以这 40 个数在签名时从封存台账机械复算写入，
运行器**读取**它们，不许自己再算；闸门拒绝「运行器算出的 `epoch*` ≠ 清单里的值」的开封请求。

**④ 聚合计数必须带分组明细。** `floor_statistics` 里 N0 那一条写成
`14/24（DS02 5/6 + DS05 3/6 + DS06 4/6 + DS07 2/6）`，不是裸的 `14/24`。
理由见 `REGISTRY_NOTES.md` v19：裸聚合写错过三次，每次都是分子分母来自不同的聚合，
而两个版本在正文里长得一模一样。
