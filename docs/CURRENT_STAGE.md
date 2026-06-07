# CartWise 当前阶段

## 状态

第一期 MVP 已完成。

CartWise 现在拥有一个可运行的单轮推荐 MVP：

- 自然语言购物需求输入。
- 基于 LLM 的查询翻译和意图解析。
- BM25、Dense、LightGCN 和 Popularity 召回。
- 加权 RRF 融合和基于 metadata 的硬过滤。
- 针对最终候选商品的 Review Evidence RAG。
- 带引用校验和模板回退的 grounded 中文解释。
- FastAPI 后端和 Streamlit 前端。

遗留兼容 wrapper `cartwise/core/llm.py` 和 `cartwise/core/evidence_rag.py` 已经移除。`cartwise/core/config.py` 仍是当前配置模块。

## 候选下一阶段

推荐下一阶段：第二期 Conversational Recommendation。

候选目标：

- 让用户 refine 上一次推荐请求。
- 支持围绕已推荐商品的后续问题。
- 保持所有后续 evidence retrieval 都限制在当前结果集内。
- 保持现有商品召回、融合、过滤和 Evidence RAG 边界。

## 当前不做事项

- 不做 Redis 或持久化多会话状态。
- 不做部署、登录、数据库或管理控制台。
- 不做复杂 Agent 或多 Agent 编排。
- 除非明确要求，不改 Fusion weights、过滤规则、Qdrant collection names、模型参数、prompts、stage 0 fixtures 或 data/index artifacts。
- 在存在 labeled query evaluation set 之前，不做 query-level benchmark claims。

## 开始新工作前

1. 运行 `git status --short`，如存在无关改动则停止。
2. 阅读 `README.md`、`docs/DECISIONS.md` 和本文件。
3. 确认任务是第二期工作还是小范围维护任务。
4. 没有用户明确批准时，不要重建 data、indexes、Qdrant collections 或 models。
