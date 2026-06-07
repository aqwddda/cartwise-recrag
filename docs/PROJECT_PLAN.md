# CartWise 项目计划

## 概览

CartWise 是一个第一期 MVP，用于乐器电商领域的可解释自然语言商品推荐。它使用 Amazon Reviews 2023 `Musical_Instruments` 数据，结合用户历史、当前购物意图、结构化过滤、商品检索和评论证据。

## 已完成的第一期范围

CartWise 当前支持：

- 自然语言购物请求。
- 用于检索的中译英查询翻译。
- 面向显式约束的 LLM 意图解析。
- BM25 商品检索。
- 使用 Qdrant 的 Dense 商品检索。
- LightGCN 已知用户推荐。
- Popularity fallback recommendation。
- Weighted RRF candidate fusion。
- 基于 metadata 的硬过滤，支持 price、brand、category、color 和 material 等约束。
- 针对最终商品候选的 Review Evidence RAG。
- 基于检索评论的中文推荐理由和潜在缺点文本。
- 引用校验和确定性模板回退。
- FastAPI 后端端点。
- Streamlit 单轮前端。
- 用于延迟分析的阶段级 timing diagnostics。

## 数据和产物

系统围绕 `parent_asin` 作为商品 key 构建。生成的数据、索引、模型、Qdrant storage 和 reports 都是本地产物，不应提交到 Git。

预期本地产物类别：

- `data/`：原始和已处理 Amazon Reviews 数据。
- `models/`：已训练 LightGCN checkpoints 和 mappings。
- `artifacts/`：生成索引、检查报告和 Qdrant 相关产物。
- `reports/`：指标和实验输出。

## 架构

```text
用户查询
  -> 查询翻译 / 意图解析
  -> BM25 + Dense + LightGCN + Popularity recall
  -> 硬过滤
  -> Weighted RRF fusion
  -> 最终商品候选
  -> Review Evidence RAG
  -> 引用校验后的解释 / 回退
  -> FastAPI 响应
  -> Streamlit UI
```

## 核心模块

| 层级           | 模块                      | 作用                                               |
| -------------- | ------------------------- | -------------------------------------------------- |
| Query          | `cartwise/query`          | 查询翻译、意图解析、过滤类型                       |
| Catalog        | `cartwise/catalog`        | 共享 product document 构建                         |
| Retrieval      | `cartwise/retrieval`      | BM25、Dense、LightGCN、Popularity、filters、fusion |
| Recommendation | `cartwise/recommendation` | 编排 intent、recall、filtering 和 fusion           |
| Evidence       | `cartwise/evidence`       | Review retrieval、explanation generation、fallback |
| Application    | `cartwise/application`    | 组合 recommendation 和 evidence services           |
| API            | `cartwise/api`            | FastAPI routes 和 HTTP schemas                     |
| UI             | `cartwise/ui`             | Streamlit page 和 HTTP API client                  |
| Scripts        | `scripts/`                | Data、index、model、audit 和 smoke utilities       |

## 运行时流程

1. FastAPI startup 构建 `RecommendationApplicationService`。
2. Application factory 加载 product metadata、BM25 index、LightGCN checkpoint、Dense encoder、Qdrant collections 和 LLM clients。
3. `POST /api/v1/recommend` 校验 request 并调用 application service。
4. `RecommendationService` 解析 intent、召回 candidates、应用 filters，并融合 candidates。
5. `EvidenceService` 为最终 candidates 检索 review evidence，并生成 citation-checked explanations。
6. FastAPI 返回稳定 response schema。
7. Streamlit 展示 product cards、evidence、explanation 和 diagnostics。

## 评估范围

第一期支持针对 Popularity 和 LightGCN 的 offline historical-interaction evaluation。自然语言 query-level evaluation 还不是严格 benchmark，因为源数据集不包含真实购物 queries 或 query-product relevance labels。

当前定性检查聚焦于：

- 返回商品是否匹配 request 和 constraints。
- Review evidence 是否属于被推荐商品。
- Explanation text 是否 grounded in product metadata 和 retrieved reviews。
- 当 LLM output invalid 时 fallback behavior 是否安全。
- 后端 latency 花在哪里。

## 已知约束

- 端到端延迟主要由 Evidence retrieval 和远程 LLM explanation 主导。
- Streamlit UI 目前只支持单轮。
- 当前没有实现 product image pipeline。
- Redis、persistent session store、login、deployment workflow 或 async task queue 都不属于第一期。
- 完整后端 readiness 前必须本地准备 Qdrant collections、data files 和 model checkpoints。
- Query-level benchmark construction 是未来工作。

## 后续改进方向

未来工作应保持聚焦和渐进：

- Conversational recommendation：允许用户在结果后 refine requirements。
- Single-product Evidence QA：只使用某个 selected product 的 reviews 回答后续问题。
- Evidence retrieval optimization 和更低 response latency。
- 更紧凑的 explanation prompts 或更快的 explanation models。
- Query-level evaluation set construction。
- 更好的 recommendation explanation quality audits。
- 在 licensing 和 storage behavior 清晰后，可选加入 product images。
