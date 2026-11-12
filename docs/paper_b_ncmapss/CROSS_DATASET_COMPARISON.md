# Cross-dataset comparison: C-MAPSS and N-CMAPSS DS02

Status: **rules only**. No cross-dataset number exists yet, and none may be
produced before both sides have a frozen analysis rule.

## The trap this document exists to prevent

The C-MAPSS audit established two facts that make naive comparison invalid:

1. C-MAPSS test labels are **clipped at 125**, and that is the
   literature-consistent convention. Raw-label numbers are a stricter, separate
   criterion (§2b of the audit).
2. C-MAPSS performance is dominated by the cycle axis and the benchmark is
   saturated: the whole method space spans ~1.9 RMSE on FD001.

DS02 has a different label structure, a different unit definition (flights, not
truncated windows) and no obvious cap. Placing a DS02 RMSE next to a C-MAPSS
RMSE as if they measured the same thing would be the single easiest way to make
this paper wrong.

## Rules

1. **Never put a C-MAPSS number and a DS02 number in the same table column.**
   Side-by-side is allowed only with both protocols named in the header.
2. **State the label convention in every DS02 row.** If DS02 has no cap, write
   "no cap" explicitly — the absence of a cap is a protocol statement, not a
   default.
3. **Never compare an absolute RMSE across the two datasets.** Compare
   *relative* effects instead: the delta between rungs, or the delta against the
   N0 age-only floor, within each dataset.
4. **The transfer claim is about ranking, not about level.** The claim this
   paper can support is "the ordering of N1/N2/N3 is preserved", or "it is not".
   It is not "our method achieves X on both benchmarks".
5. **Report the N0 floor for both datasets.** It is the only quantity that is
   meaningfully comparable in spirit — how much of the task is just the clock.

## What a valid transfer claim looks like

> On C-MAPSS (capped labels, FD001–FD004, three seeds) the DSPT base sits 1.49
> RMSE below a vanilla Transformer with the same inputs. On DS02 (LOEO over six
> dev units, end-of-flight prediction) the same comparison gives [value] with a
> per-fold spread of [value]. The ordering is [preserved / not preserved].

Anything weaker than a per-fold paired statement is not a transfer result.

## Falsification

- If the ranking reverses between datasets, report the reversal and do not
  average it away.
- If the DS02 result is within the cross-fold spread, the transfer claim is
  rejected — "no evidence of transfer" is the correct conclusion, not "trend
  towards transfer".
