# Literature & Protocol Benchmark — IEEE TIM Track

Last updated: 2026-10-21
Owner: research agent (this repository)

> **Verification policy (binding).** A number enters this file only after the
> source was actually read. Anything I could not verify is marked
> `UNVERIFIED — PDF required` and is **never** quoted in a paper draft. Two
> tables are kept strictly apart: **cross-paper reference** numbers (different
> protocols) and **same-protocol reproduction** numbers produced in this repo.

---

## 1. Why the protocol columns matter more than the numbers

C-MAPSS RMSE is routinely reported under protocols that are not comparable:

| Axis | Common variants | Effect on RMSE |
|---|---|---|
| RUL cap | 125 vs 130 vs none | 0.1-0.4 |
| Window length | 30 / 50 / 60 / full sequence | 0.2-0.8 |
| Operating-condition handling | raw 14 sensors / per-condition normalisation / clustering | **1-3 on FD002/FD004** |
| Evaluation point | last window only vs sliding-window mean vs full-sequence | 0.2-0.6 |
| Cycle feature | absolute cycle index included vs not | leaks unit age; 0.3-1.0 |
| Training engines | 100% vs 80/20 split | 0.5-1.0 |
| Model selection | validation only vs test-peeking | unbounded |

A 12.3 on FD002 under "per-condition normalisation + last-window + 125 cap"
is a different result from a 12.3 under "raw sensors + sliding mean". Every
entry below therefore carries the protocol flags, and an entry with unknown
flags is marked as such.

---

## 2. Cross-paper reference library (2024-2026)

Legend: **V** = verified from the paper/abstract text; **P** = partial (abstract
only, protocol flags unknown); **U** = unverified, PDF required.
Protocol columns: L = window, cap = RUL cap, CN = condition normalisation,
E = evaluation mode (LW = last window, SW = sliding mean).

### 2.1 Named targets

| Method | Venue / year | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE | L | cap | CN | E | Status |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| HDA-GNN | IEEE TIM 2025 | UNVERIFIED — PDF required | UNVERIFIED | UNVERIFIED | UNVERIFIED | ? | ? | ? | ? | U |
| ST-KDFormer | IEEE TIM 2025 | UNVERIFIED — PDF required | UNVERIFIED | UNVERIFIED | UNVERIFIED | ? | ? | ? | ? | U |
| LVDACNN | IEEE TIM 2026 | UNVERIFIED — PDF required | UNVERIFIED | UNVERIFIED | UNVERIFIED | ? | ? | ? | ? | U |

The three papers above were located on IEEE Xplore during this session
(abstract pages 11004129, 11217208, 11359311 / 11216598 respectively), but the
full texts are paywalled and no numeric result could be extracted from the
abstract snippets returned. **No number is invented here.** To fill these
cells, drop the PDFs into `docs/lit/` and the table can be completed in
minutes; until then these rows must not appear in a manuscript table.

### 2.2 Representative method families to benchmark against

| Family | Representative work (2024-2026) | What it claims | Protocol risk | Status |
|---|---|---|---|---|
| Patch / channel-independent Transformer | PatchTST-style RUL variants | channel independence beats channel mixing | some report per-dataset tuned L | U (PDF required) |
| Inverted attention | iTransformer-style RUL variants | attention across variates, not time | often full-sequence eval | U |
| State-space | Mamba / selective SSM RUL variants | linear-time long context | rarely reports FD003/FD004 | U |
| Dynamic graph | HDA-GNN, ST-KDFormer family | condition-adaptive sensor coupling | graph usually pooled | U |
| Dual-task HI | knee-point / HI dual-branch works | health-stage classification helps RUL | class boundary often tuned on test | U |
| CNN-Transformer hybrids | LVDACNN family | local conv + global attention | sliding-mean evaluation common | U |

**Rule for the paper:** only rows with a complete protocol column may be cited
in the comparison table; rows with `?` go into a supplementary "reported
under heterogeneous protocols" note.

---

## 3. Same-protocol reproduction in this repository

Protocol S1 (ours): window 50, stride 1, RUL cap 125, Min-Max on training
engines only (FD002/FD004 additionally use 6-cluster operating-condition
normalisation), last-window evaluation, 5 seeds 2027-2031, 100% training
engines, engine-level split only for schedule selection.

| Model | FD001 RMSE | FD002 RMSE | FD003 | FD004 | Params | Source |
|---|---:|---:|---|---|---:|---|
| TCN | 16.646 +/- 0.413 | 19.958 +/- 1.542 | sealed | sealed | 145,237 | P10 |
| BiLSTM | 14.158 +/- 0.290 | 14.027 +/- 0.112 | sealed | sealed | 144,701 | P10 |
| Vanilla Transformer | **13.289 +/- 0.132** | **12.707 +/- 0.178** | sealed | sealed | 144,146 | P10 |
| P9 CR-GraphTransformer | 13.209 +/- 0.318 | 13.490 +/- 0.414 | sealed | sealed | 143,415 | P9 |
| P10 native sensor-graph | 13.753 +/- 0.191 | 13.709 +/- 0.437 | sealed | sealed | 141,707 | P10 |

FD003/FD004 are **sealed blind sets** and will stay sealed until Stage 4.

---

## 4. Data audit (Stage 1, 2026-10-21)

| File | Lines | Columns | Mode before audit | Mode after audit |
|---|---:|---:|---|---|
| train_FD001.txt | 20,631 | 26 | rw-r--r-- | rw-r--r-- |
| test_FD001.txt | 13,096 | 26 | rw-r--r-- | rw-r--r-- |
| RUL_FD001.txt | 100 | 1 | rw-r--r-- | rw-r--r-- |
| train_FD002.txt | 53,759 | 26 | rw-r--r-- | rw-r--r-- |
| test_FD002.txt | 33,991 | 26 | rw-r--r-- | rw-r--r-- |
| RUL_FD002.txt | 259 | 1 | rw-r--r-- | rw-r--r-- |
| train_FD003.txt | 24,720 | 26 | rw-r--r-- | rw-r--r-- |
| test_FD003.txt | 16,596 | 26 | ---------- | **sealed** |
| RUL_FD003.txt | 100 | 1 | ---------- | **sealed** |
| train_FD004.txt | 61,249 | 26 | rw-r--r-- | rw-r--r-- |
| test_FD004.txt | 41,214 | 26 | ---------- | **sealed** |
| RUL_FD004.txt | 248 | 1 | ---------- | **sealed** |

Line counts match the canonical NASA distribution (FD001 100 test units,
FD002 259, FD003 100, FD004 248). FD003/FD004 test+RUL are `chmod 000` and
Stage 2/3 code paths never open them.

---

## 5. What I will report in the paper table

1. Same-protocol block (Section 3) for all four subsets after Stage 4.
2. Cross-paper block (Section 2) restricted to rows with complete protocol
   columns, with a footnote naming the protocol differences.
3. An explicit statement that no cross-paper number was produced by this repo.
