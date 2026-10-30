# SUPERSEDED — P9 numbers are not citable

Date: 2026-10-29

Every number in this directory (`p9_summary.md`, `p9_table_main.tex`, metrics
JSON) was produced by the synergy head. `CRGraphTransformer._smooth_terminal`
ordered the tail window old→new while its weights and the `-k` offset are defined
new→old, so the head consumed a reversed tail.

Consequence: **13.209 (single, FD001) and 12.868 (5-seed ensemble) are void.**
The files are kept unchanged as an audit trail; they must not be regenerated
silently or cited.

See `docs/director/REVIEW_20261022_R3.md` §J and `paper/E5_CONSISTENCY.md`.
