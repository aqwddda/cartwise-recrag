# CartWise 开发里程碑

本文档记录已完成的第一期里程碑。它不是逐步的内部执行日志。

## 第一期目标

构建一个本地、可复现的 MVP，用于乐器电商领域的可解释自然语言商品推荐。

系统结合了协同推荐、词面搜索、dense retrieval、metadata filtering、review evidence retrieval 和 LLM 生成解释。它是研究/演示系统，不是生产服务。

## 已完成里程碑

### 1. 数据准备

CartWise 使用 Amazon Reviews 2023 `Musical_Instruments` 数据。预处理 pipeline 围绕 `parent_asin` 规范化商品，创建 train/validation/test 交互拆分，将 item metadata 与 reviews 连接，并为后续 evidence citation 构建稳定 review identifiers。

主要输出预期位于 `data/processed/` 下，包括 item metadata、interaction splits、review records 和 filter alias mappings。这些生成产物不提交到 Git。

### 2. 热门度基线

Popularity 推荐器统计每个 item 的训练交互次数，并推荐用户尚未交互过的热门 items。它提供冷启动基线，也为 LightGCN 提供离线比较点。

主要模块：

- `cartwise/retrieval/popularity.py`
- `scripts/pipeline/evaluate_popularity.py`

### 3. LightGCN 个性化

LightGCN 使用 PyTorch Geometric 在 user-item interaction graph 上实现。训练脚本保存模型权重、ID mappings 和 user history，用于 Top K inference 和 evaluation。

主要模块：

- `cartwise/retrieval/lightgcn.py`
- `scripts/pipeline/train_lightgcn.py`

### 4. 商品 BM25 和 Dense 检索

商品 metadata 被渲染为共享 product document。BM25 为 brand、model、instrument type 和 accessory names 等 terms 提供词面匹配。Dense retrieval 使用 sentence-transformer style encoders 和 Qdrant collections 做语义匹配。

中文输入会在检索前翻译为英文。英文输入直接进入检索。

主要模块：

- `cartwise/catalog/documents.py`
- `cartwise/retrieval/bm25.py`
- `cartwise/retrieval/dense.py`
- `scripts/pipeline/build_product_bm25_index.py`
- `scripts/pipeline/build_product_dense_index.py`

### 5. LLM 查询处理和硬过滤

LLM intent parser 提取显式约束，例如 product terms、budget、brand preference、excluded brands、color 和 material。Parser 不排序商品，也不生成推荐。解析值会映射到 `FilterConstraints`，硬过滤在代码中执行约束。

主要模块：

- `cartwise/query/llm.py`
- `cartwise/query/types.py`
- `cartwise/retrieval/filters.py`

### 6. Fusion 排序

CartWise 使用 source-aware weighted reciprocal rank fusion 合并 BM25、Dense、LightGCN 和 Popularity candidates。Fusion layer 对商品去重，记录所有 candidate sources，应用 filtering policies，并返回排序后的最终候选列表。

主要模块：

- `cartwise/retrieval/fusion.py`

### 7. Review Evidence RAG

Review Evidence RAG 与商品召回分离。系统先决定推荐哪些商品，然后只为这些商品检索 review evidence。检索到的 reviews 会通过 `parent_asin` 校验；生成解释只能引用检索到的 review IDs。无效 LLM 输出会回退到确定性模板解释。

主要模块：

- `scripts/pipeline/build_evidence_index.py`
- `cartwise/evidence/rag.py`
- `cartwise/evidence/service.py`
- `cartwise/evidence/types.py`

### 8. 服务层

推荐链被组织为显式服务边界：

- `RecommendationService` 处理 query intent、recall、filtering 和 fusion。
- `EvidenceService` 检索 review evidence 并生成 grounded explanations。
- `RecommendationApplicationService` 将 recommendation 和 evidence 组合为稳定的 application-level result。

主要模块：

- `cartwise/recommendation/service.py`
- `cartwise/evidence/service.py`
- `cartwise/application/service.py`
- `cartwise/application/factory.py`

### 9. FastAPI 后端

FastAPI 暴露 liveness、readiness 和 recommendation endpoints。Routes 只调用 application service，不在 request handlers 中组装 retrievers、models、Qdrant clients 或 LLM clients。重资源由 composition root 在 startup 期间构建。

主要模块：

- `cartwise/api/main.py`
- `cartwise/api/schemas.py`

Endpoints：

- `GET /health/live`
- `GET /health/ready`
- `POST /api/v1/recommend`

### 10. Streamlit 前端

Streamlit UI 是 FastAPI 的轻量 HTTP 客户端。它不导入或调用 retrieval、recommendation、evidence、Qdrant、model 或 LLM 内部模块。

主要模块：

- `cartwise/ui/api_client.py`
- `cartwise/ui/app.py`

### 11. 性能诊断

后端会记录 application service、recommendation service、retrieval/fusion、evidence retrieval 和 LLM explanation 阶段的阶段级 timing diagnostics。第一次低风险优化加入了重复 evidence retrieval queries 的 query-vector caching，并阻止 Streamlit reruns 重新触发 recommendation requests。

当前已知瓶颈：当 `top_k` 较大时，Evidence retrieval 和 LLM explanation 主导端到端延迟。

### 12. Wrapper 清理

在测试和 legacy harness imports 迁移到新 module paths 后，旧 re-export wrappers 已被移除：

- Removed: `cartwise/core/llm.py`
- Removed: `cartwise/core/evidence_rag.py`
- Retained: `cartwise/core/config.py`

## 第一期验收状态

第一期已作为本地 MVP 完成。它支持完整单轮流程：

```text
自然语言需求
-> query translation / intent parsing
-> BM25 + Dense + LightGCN + Popularity recall
-> hard filtering + weighted RRF fusion
-> 针对最终候选的 review evidence retrieval
-> grounded explanation / fallback
-> FastAPI response
-> Streamlit presentation
```

## 下一方向

第二期应聚焦 Conversational Recommendation 和 single-product Evidence QA：

- 细化现有推荐请求。
- 针对被推荐商品提出后续问题。
- 只使用当前结果集中的 evidence 比较商品。
- 在不过早改变第一期 recall/fusion 边界的前提下，提升 evidence retrieval latency 和 explanation quality。
