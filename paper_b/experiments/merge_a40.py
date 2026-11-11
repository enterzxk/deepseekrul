"""Merge the a40 attribution results into the lab3090 result table."""
import sys
from pathlib import Path

import pandas as pd


def main():
    main_csv = Path(sys.argv[1] if len(sys.argv) > 1 else 'paper_b/results/all_runs.csv')
    a40_csv = Path(sys.argv[2] if len(sys.argv) > 2 else 'paper_b/results/a40_all_runs.csv')
    out = Path(sys.argv[3] if len(sys.argv) > 3 else 'paper_b/results/all_runs_merged.csv')
    a = pd.read_csv(main_csv)
    b = pd.read_csv(a40_csv)
    b = b[b.tag.str.startswith('ATTR_')]
    m = pd.concat([a, b], ignore_index=True).drop_duplicates(['tag', 'subset', 'seed'], keep='last')
    m.to_csv(out, index=False)
    print('merged', len(m), 'rows ->', out)


if __name__ == '__main__':
    main()
