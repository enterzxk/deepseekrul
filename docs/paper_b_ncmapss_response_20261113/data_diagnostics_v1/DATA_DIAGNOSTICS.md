# Development sensor/condition diagnostics

Descriptive audit; no RUL labels or TEST decoded. Not a RUL comparison or module success claim.

| Subset | Median condition R² | Median abs raw-age correlation | Median abs residual-age correlation |
|---|---:|---:|---:|
| DS02 | 0.9998 | 0.1021 | 0.8242 |
| DS05 | 0.9998 | 0.1205 | 0.7188 |
| DS06 | 0.9998 | 0.1174 | 0.6943 |
| DS07 | 0.9998 | 0.1154 | 0.6755 |

residual includes model error, operating dynamics, initial offsets and degradation; correlation with age is not a physical HI proof

All fits use only the five training engines. Complete fold/sensor values and fitted coefficients accompany this file.
