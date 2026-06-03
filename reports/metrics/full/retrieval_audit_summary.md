# Phase 6 Retrieval Audit Summary

## Overall

| channel | queries | results | mean_relevance@10 | excellent_rate@10 | acceptable_rate@10 | bad_rate@10 | ndcg@10 | mrr_strict@10 | mrr_loose@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5 | 30 | 300 | 1.656667 | 0.756667 | 0.900000 | 0.100000 | 0.924928 | 0.866667 | 0.927778 |
| blair | 30 | 300 | 1.620000 | 0.736667 | 0.883333 | 0.116667 | 0.921301 | 0.838095 | 0.906667 |
| bm25 | 30 | 300 | 1.413333 | 0.643333 | 0.770000 | 0.230000 | 0.912412 | 0.880556 | 0.950000 |

## Pairwise Wins

| channel | compared_model | queries | wins | losses | ties |
| --- | --- | --- | --- | --- | --- |
| e5 | blair | 30 | 15 | 11 | 4 |
| blair | e5 | 30 | 11 | 15 | 4 |
| e5 | bm25 | 30 | 15 | 8 | 7 |
| bm25 | e5 | 30 | 8 | 15 | 7 |
| blair | bm25 | 30 | 15 | 10 | 5 |
| bm25 | blair | 30 | 10 | 15 | 5 |

## Notes

- Scores use the human relevance labels 0, 1, and 2.
- Pairwise wins compare per-query `ndcg@10`, then `mean_relevance@10`.
- This summary intentionally does not split results by query language.
