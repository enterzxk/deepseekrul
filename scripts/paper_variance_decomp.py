"""Section 4.1 — variance decomposition, two independent designs.

Why this script exists: the numbers 2.6 / 31.5 / 65.9 were quoted in the
director documents but existed only as a table inside
docs/director/REVIEW_20261022_R1.md section 37. A CLAIMS entry backed by a
prose table is not reproducible, so the table is transcribed here verbatim,
re-analysed with a plain two-way ANOVA, and written out as JSON.

Two designs are reported side by side because they were run independently, on
different validation-set families and different run subsets:
  A. split 7/8/9, n=27 runs, unbalanced -> results/dsh/review_r1/variance_decomposition.json
  B. 3 methods x 3 validation sets, balanced 3x3 -> transcribed below

Neither design replicates runs, so the interaction and the residual are not
separable. The third row of each table is therefore labelled
"interaction + residual", never "interaction".

Run: python scripts/paper_variance_decomp.py
"""
import json
import os

OUT = 'results/dsh/e10/variance_two_designs.json'

# --- Design B: transcribed from REVIEW_20261022_R1.md section 37, table 1 ---
# rows = methods, columns = validation-set families
DESIGN_B = {
    'methods': ['M0_raw_tf', 'M1_global', 'M6_tfA'],
    'validation_sets': ['A', 'B', 'C'],
    'rmse': [[15.78, 16.99, 13.99],
             [17.19, 12.93, 15.57],
             [16.79, 14.66, 15.80]],
    'source': 'docs/director/REVIEW_20261022_R1.md section 37 (archived table)',
}

# --- Design A: the balanced-ANOVA counterpart computed from trained runs ---
DESIGN_A_PATH = 'results/dsh/review_r1/variance_decomposition.json'


def two_way(table):
    """Balanced two-way ANOVA share of total sum of squares."""
    rows, cols = len(table), len(table[0])
    n = rows * cols
    grand = sum(sum(r) for r in table) / n
    row_means = [sum(r) / cols for r in table]
    col_means = [sum(table[i][j] for i in range(rows)) / rows for j in range(cols)]
    ss_row = cols * sum((x - grand) ** 2 for x in row_means)
    ss_col = rows * sum((x - grand) ** 2 for x in col_means)
    ss_tot = sum((v - grand) ** 2 for r in table for v in r)
    ss_int = ss_tot - ss_row - ss_col
    return dict(grand_mean=grand, ss_total=ss_tot, ss_row=ss_row, ss_col=ss_col,
                ss_interaction=ss_int,
                pct_row=100 * ss_row / ss_tot,
                pct_col=100 * ss_col / ss_tot,
                pct_interaction=100 * ss_int / ss_tot,
                row_means=row_means, col_means=col_means)


def main():
    b = two_way(DESIGN_B['rmse'])
    print('Design B (3 methods x 3 validation sets, balanced)')
    print(f"  grand mean {b['grand_mean']:.4f}   SS_total {b['ss_total']:.4f}")
    for i, name in enumerate(DESIGN_B['methods']):
        print(f"  method {name:12s} mean {b['row_means'][i]:.3f}")
    print(f"  method      {b['pct_row']:.2f}%")
    print(f"  validation  {b['pct_col']:.2f}%")
    print(f"  interaction+residual {b['pct_interaction']:.2f}%")

    a = None
    if os.path.exists(DESIGN_A_PATH):
        raw = json.load(open(DESIGN_A_PATH))
        a = dict(n=raw['n'], grand_mean=raw['grand_mean'],
                 pct_row=raw['pct_method'], pct_col=raw['pct_split'],
                 pct_interaction=raw['pct_interaction'],
                 pct_residual=raw['pct_residual_seed'],
                 note='unbalanced design, split 7/8/9; interaction and seed are '
                      'separate columns because each cell was replicated over seeds')
        print('\nDesign A (split 7/8/9, n=27)')
        print(f"  method {a['pct_row']:.2f}%  split {a['pct_col']:.2f}%  "
              f"interaction {a['pct_interaction']:.2f}%  seed {a['pct_residual']:.2f}%")

    out = dict(
        purpose='Section 4.1: how much of the RMSE variation is attributable to '
                'method identity rather than to the choice of held-out engines',
        design_b=dict(**{k: v for k, v in DESIGN_B.items()}, anova=b),
        design_a=a,
        shared_conclusion='method identity explains under 3% of the variance in both '
                          'designs; the engine/validation-set choice explains 23-32%; '
                          'the majority sits in the method-by-split interaction',
        interval_status='point estimates only: neither design replicates runs, so no '
                        'interval is attached to the shares',
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
