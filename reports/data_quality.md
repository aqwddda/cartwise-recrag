# Musical_Instruments Data Quality Report

## Dataset Scope

| Metric | Value |
|---|---:|
| Catalog items | 24,587 |
| Users | 57,439 |
| Retained interactions | 511,836 |
| Retained review evidence candidates | 240,525 |
| Items with retained reviews | 24,587 |
| Maximum retained reviews for one item | 10 |

## Chronological Splits

| Split | Source rows | Retained rows | Filtered outside catalog | Deduplicated rows |
|---|---:|---:|---:|---:|
| train | 396,958 | 396,958 | 0 | 0 |
| valid | 57,439 | 57,439 | 0 | 0 |
| test | 57,439 | 57,439 | 0 | 0 |

Chronology validation passed across 172,317 shared-user comparisons.

## Missing Fields

| Artifact | Field | Missing rows | Missing ratio |
|---|---|---:|---:|
| items | title | 3 | 0.01% |
| items | brand | 91 | 0.37% |
| items | price | 8,103 | 32.96% |
| items | description | 6,941 | 28.23% |
| source catalog reviews | text | 1,770 | 0.08% |

## Filtering Summary

| Metric | Value | Ratio |
|---|---:|---:|
| Metadata items absent from source metadata | 0 | 0.00% |
| Duplicate metadata rows ignored | 0 | n/a |
| Catalog review rows with empty text ignored | 1,770 | 0.08% |
| Eligible review rows removed by per-item cap or final deduplication | 1,877,058 | 88.64% |
| Retained low-rating review rows | 45,485 | 18.91% |

## Notes

- `parent_asin` is the unified item key.
- Official leave-last-out train, validation, and test files remain separate.
- Training interactions are never augmented with validation or test interactions.
- Review evidence candidates retain at most the configured number of reviews per item.
