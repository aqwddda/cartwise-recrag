# CartWise Project Context

## Project Goal

CartWise is an evidence-grounded conversational e-commerce recommender system built as a resume project.

Chinese title: 基于 RAG 的可解释个性化电商导购系统

The system combines traditional recommendation models with retrieval-augmented generation. Recommendation models decide which items to recommend. RAG retrieves product knowledge and review evidence. The LLM parses user intent and generates grounded explanations, but it must not invent products, prices, or unsupported claims.

## Core User Experience

Example request:

> 推荐 500 元以内、适合办公室视频会议、佩戴舒适的耳机。我之前喜欢降噪产品，不要入耳式。

The system returns a Top 5 product list with:

- match reasons based on current requirements and long-term preferences
- product price and important attributes
- real review evidence citations
- potential drawbacks supported by reviews
- support for follow-up requests such as 更便宜、换一批、不要这个品牌

## Recommendation Pipeline

1. Parse the natural-language request into structured session constraints.
2. Build a user profile from long-term interaction history and current-session requirements.
3. Retrieve candidates from multiple channels:
   - collaborative filtering: BPR or LightGCN
   - semantic retrieval: dense embedding and BM25
   - graph expansion: bought-together neighbors
   - popularity fallback for cold-start users
4. Merge candidates and normalize scores.
5. Apply hard filters in code, such as category, price range, and excluded attributes.
6. Re-rank a small candidate set with weighted features or a CrossEncoder.
7. Retrieve review evidence separately for each final candidate item.
8. Ask the LLM to generate explanations using only the supplied candidates and evidence.
9. Update the session profile after follow-up requests and repeat the relevant stages.

## Data Model

Use one category subset from Amazon Reviews 2023 instead of the full dataset.

Required data:

- user-item interactions, ratings, and timestamps
- product title, brand, price, category, description, and attributes
- review text, score, and timestamp
- bought-together item relationships

Keep two separate retrieval indexes:

- product index: candidate retrieval
- review index: evidence retrieval for selected products

## Suggested Technology Stack

- recommendation baseline: Popularity and PyTorch Geometric LightGCN, with BPR as an optional comparison
- optional sequential model: SASRec
- text embedding: BGE-M3 or a lightweight Sentence Transformer
- hybrid retrieval and metadata filtering: Qdrant plus BM25
- second-stage ranking: Sentence Transformers CrossEncoder
- backend: FastAPI
- demo UI: Streamlit for the first version
- experiment tracking: MLflow or CSV plus Markdown reports
- LLM: an OpenAI-compatible API behind a replaceable adapter

## Evaluation

Use chronological train, validation, and test splits to prevent future-data leakage.

Offline metrics:

- Recall@10
- NDCG@10
- HitRate@10
- cold-start Recall@10
- citation accuracy for generated explanations
- P95 online latency

Ablation sequence:

1. Popularity baseline
2. BPR or LightGCN
3. LightGCN plus semantic retrieval
4. Add bought-together graph expansion
5. Add second-stage reranking
6. Compare cold-start performance and explanation grounding

## Scope Control

First release:

- one Amazon product category only
- LightGCN, semantic retrieval, and popularity fallback
- structured filters
- review evidence citations
- FastAPI and Streamlit demo
- reproducible evaluation report

Second release:

- bought-together graph expansion
- CrossEncoder reranking
- multi-turn session state
- cold-start analysis

Avoid complex multi-agent orchestration until the complete recommendation pipeline is working and measurable.

## Resume Description Draft

CartWise：基于 Amazon Reviews 2023 构建可解释个性化电商导购系统，融合 LightGCN 协同过滤、Dense/BM25 混合检索、商品共购图扩展和 Cross-Encoder 重排，实现支持多轮约束更新与评论证据引用的 RAG 推荐链路；通过 Recall@10、NDCG@10 和冷启动实验评估各模块增益，并使用 FastAPI 与 Qdrant 完成在线服务化。

## Repository Name

Suggested repository name: `cartwise-recrag`
