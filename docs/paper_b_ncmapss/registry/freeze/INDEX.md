# 冻结清单索引（`freeze/`）

一份清单对应一次「开封前声明」。**只有一份是 `active`**，其余是 `superseded`。

| freeze_id | 状态 | `freeze_ts_utc` | 哈希规则 | `freeze_content_sha256` | 格数 | 说明 |
|---|---|---|---|---:|---:|---|
| `NCM-FREEZE-20261112T1955Z` | **active** | 2026-11-12T19:55:04Z | **v17** | `476bdaea731098f97ea4b161168bd92d521549d7ba9637ac71ac66da45259cc0` | 40 | 第 14 代锚 `497afe23…` 下的测试批。v17 把 `freeze_ts_utc` 纳入哈希；补 `sequencing` / `completeness` / `epoch_star_values`。 |
| `NCM-FREEZE-20261112T1945Z` | superseded | 2026-11-12T19:45:00Z | v16 | `3c27009e3f5e9c6f6c456046f8b2a7d4672dfe242cc6a38f44156aec439b3932` | 40 | 未用于跑批。作废原因：`freeze_ts_utc` 不在哈希内，可被改早而不被发现。 |

## 两条规矩

**① 作废不回改。** `superseded` 的清单**字节一个都没动**——它的摘要是按当时的规则签的，
往里加一个 `status` 字段就等于让摘要失效。作废状态只记在本索引、`REGISTRY_NOTES.md`
和继任清单的 `supersedes` 字段里。改历史文件等于让「这份清单当时写了什么」不可考。

**② 摘要按签名时的规则算，不是按现行规则。** 上表 `v16` 那一行的摘要只能这样复算：

```python
body = {k: v for k, v in d.items()
        if k not in ("freeze_ts_utc", "freeze_content_sha256", "eval_start_utc")}
sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
```

现行规则（v17）的复算走一条命令：

```bash
python scripts/nc_freeze.py verify docs/paper_b_ncmapss/registry/freeze/NCM-FREEZE-20261112T1955Z.json
# hash / completeness / epoch_star / ordering 四项，全 OK 才许开跑
```

## 清单里为什么要有 `epoch_star_values`

§6.10 说 `epoch*` 由 dev 六折估计、**在开封前冻结**。如果这 40 个数不在被哈希的清单里，
这句话就没法执行——谁都可以在开封后按自己的算法重算一遍，而且看起来完全合规。
所以这 40 个数在签名时从封存台账机械复算写入，运行器**读取**它们，不许自己再算。

`floor_statistics` 同批写入，用来复算「撞地板」那条统计：N0 的 24 折里 **14 折**落在 20
（DS02 5/6、DS05 3/6、DS06 4/6、DS07 2/6）。
