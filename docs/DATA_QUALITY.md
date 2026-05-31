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

## Product Review Distribution

The `5core` scope below includes all raw reviews associated with products present
in the official `5core` dataset. These values are calculated before applying the
review evidence limit of 10 reviews per product.

| Metric | All raw products | Products present in `5core` |
|---|---:|---:|
| Products in scope | 213,593 | 24,587 |
| Products with reviews | 213,571 | 24,587 |
| Products without reviews | 22 | 0 |
| Raw reviews | 3,017,439 | 2,119,353 |
| Mean reviews per reviewed product | 14.13 | 86.20 |
| Minimum reviews per reviewed product | 1 | 5 |
| P25 reviews per reviewed product | 1 | 17 |
| Median reviews per reviewed product | 2 | 32 |
| P75 reviews per reviewed product | 7 | 72 |
| P90 reviews per reviewed product | 22 | 168 |
| P95 reviews per reviewed product | 47 | 294 |
| P99 reviews per reviewed product | 199 | 978.14 |
| Maximum reviews per reviewed product | 9,334 | 9,334 |
| Mean rating | 4.26 | 4.30 |
| Reviews with empty text | 2,830 | 1,770 |

| Coverage metric | Value |
|---|---:|
| `5core` product share of all raw products | 11.51% |
| `5core` product share of raw products with reviews | 11.51% |
| Raw review share associated with `5core` products | 70.24% |

| Rating | All raw product reviews | Reviews for products present in `5core` |
|---|---:|---:|
| 1 | 264,352 (8.76%) | 166,842 (7.87%) |
| 2 | 131,441 (4.36%) | 87,284 (4.12%) |
| 3 | 197,131 (6.53%) | 133,729 (6.31%) |
| 4 | 400,387 (13.27%) | 284,337 (13.42%) |
| 5 | 2,024,128 (67.08%) | 1,447,161 (68.28%) |

## Verified Purchase and High-Rating Statistics

The official `5core` interaction CSV files do not contain `verified_purchase`.
The interaction-level values below were obtained by joining each `5core`
interaction back to the raw review dataset using `user_id`, `parent_asin`,
`rating`, and `timestamp`. All `511,836` interactions were matched.

| Scope | Total rows | Verified purchase rows | Verified purchase ratio | Rating > 4 rows | Rating > 4 ratio |
|---|---:|---:|---:|---:|---:|
| Raw reviews | 3,017,439 | 2,780,515 | 92.15% | 2,024,128 | 67.08% |
| `5core` interactions joined to raw reviews | 511,836 | 461,477 | 90.16% | 361,351 | 70.60% |
| Raw reviews for the 24,587 products present in `5core` | 2,119,353 | 1,967,350 | 92.83% | 1,447,161 | 68.28% |

## Notes

- `parent_asin` is the unified item key.
- Official leave-last-out train, validation, and test files remain separate.
- Training interactions are never augmented with validation or test interactions.
- Review evidence candidates retain at most the configured number of reviews per item.
