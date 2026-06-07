# CartWise

CartWise 是一个基于 RAG 的可解释个性化电商导购系统。项目使用 Amazon Reviews 2023 `Musical_Instruments` 数据，将用户历史偏好、当前自然语言需求、结构化硬约束和真实评论证据结合起来，返回可追溯的商品推荐解释。

## 当前状态

当前源码已经完成一期功能闭环：推荐链路结构重构、服务层提取、FastAPI 接入、Streamlit 单轮演示页面、阶段级延迟诊断和第一轮低风险延迟优化。Streamlit 只通过 HTTP 调用 FastAPI，不直接导入 retrieval、recommendation、evidence、Qdrant、模型对象或 LLM client。

阶段 10 已完成第一轮延迟可观测性和小范围优化：后端记录阶段级 `cartwise_timing` 日志，Evidence review query 向量支持进程内缓存，Streamlit 默认 `top_k=3` 并通过 `st.session_state` 避免页面 rerun 重复请求。进一步将端到端推荐压到 5 秒内的优化计划已记录在 `docs/FUTURE_IMPROVEMENTS.md` 的 `FI-12`。

一期收尾时不强行构建自然语言 query-level benchmark。Amazon Reviews 2023 当前数据没有真实导购 query 或 query-product 人工相关性标注，因此自然语言 query 链路只作为 demo、延迟诊断和人工审查场景；query-level 标注评估集建设已作为 `FI-13` 记录到 `docs/FUTURE_IMPROVEMENTS.md`。

最近验证状态：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 154 passed, 3 warnings

.\.venv\Scripts\python.exe -m pytest tests/test_api_dependencies.py tests/test_api_lifespan.py tests/test_api.py -q --basetemp="$env:TEMP\cartwise-pytest-api"
# 21 passed
```

## 核心能力

- Query 翻译与意图解析：中文 query 先转英文，英文 query 直接进入检索；意图解析只抽取显式约束。
- 商品召回：BM25、Dense、Popularity、LightGCN。
- 过滤与融合：代码层硬过滤和 weighted RRF Fusion。
- 评论证据 RAG：只在最终候选商品范围内检索评论证据。
- 结构化解释：LLM 输出经引用校验，不合法时模板回退。
- 行为回归：阶段 0 legacy regression fixture 固定旧链路关键结构行为。

## 当前架构分层

```text
cartwise/
  application/      # RecommendationApplicationService 和 API composition root
  api/              # FastAPI 推荐接口、schema、lifespan 和 readiness
  catalog/          # 商品文档构造共享逻辑
  core/             # config.py 仍有效；llm.py/evidence_rag.py 是兼容 wrapper
  evidence/         # Evidence RAG、EvidenceService 和证据类型
  query/            # Query LLM adapter 和 FilterConstraints 等 query 类型
  recommendation/   # RecommendationService 和推荐服务类型
  retrieval/        # Dense、BM25、Popularity、LightGCN、filters、fusion
  ui/               # Streamlit HTTP client 和单轮演示页面
scripts/
  tools/
    audit_retrieval.py
    run_stage8_smoke.py
    stage8_smoke_adapter.py
```

正式业务入口是 `cartwise.application.RecommendationApplicationService`。FastAPI 路由调用该服务，而不是在路由里重新拼装 Dense、BM25、Popularity、LightGCN、Fusion、Qdrant 或 LLM。

## 主要模块

- `cartwise.recommendation.service.RecommendationService`：正式推荐链路编排，负责意图解析、过滤约束、Dense、BM25、Popularity、LightGCN 和 Fusion。
- `cartwise.evidence.service.EvidenceService`：基于最终候选商品批量检索评论证据并生成解释。
- `cartwise.application.service.RecommendationApplicationService`：串联 RecommendationService 和 EvidenceService，输出应用层结构化结果。
- `scripts.tools.stage8_smoke_adapter.Stage8SmokeAdapter`：只服务历史 Stage 8 smoke 工具，不属于正式业务服务，FastAPI 不得调用。

## 运行入口

常用脚本入口：

```powershell
.\.venv\Scripts\python.exe -m scripts.tools.audit_retrieval --scope full --channels fusion --query "guitar tuner for beginners" --top-k 5

.\.venv\Scripts\python.exe -m scripts.tools.run_stage8_smoke --scope full --query "guitar tuner for beginners" --top-k 5 --dense-k 10 --bm25-k 10 --device cuda
```

FastAPI 入口：

```text
GET  /health/live
GET  /health/ready
POST /api/v1/recommend
```

启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn cartwise.api.main:app --reload
```

默认 app 会在启动期构造真实 `RecommendationApplicationService`。如果本机 Qdrant、collection、数据文件、BM25、LightGCN 模型或 LLM Key 缺失，`/health/ready` 会返回 not ready 和初始化错误。

Streamlit UI 入口：

```powershell
.\.venv\Scripts\python.exe -m streamlit run cartwise/ui/app.py
```

## 可复现 Demo 流程

以下命令从仓库根目录运行。Python 命令统一使用虚拟环境中的 `.\.venv\Scripts\python.exe`。下载依赖、模型或数据时，如需代理，优先使用 `http://127.0.0.1:9508`；访问 `127.0.0.1` 和 `localhost` 时应绕过代理。

准备数据和样本：

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.preprocess_amazon_reviews
.\.venv\Scripts\python.exe -m scripts.pipeline.build_dev_sample
```

训练和评估推荐模型：

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.evaluate_popularity --scope full
.\.venv\Scripts\python.exe -m scripts.pipeline.train_lightgcn --scope full
```

构建商品和评论索引：

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.build_product_dense_index --scope full
.\.venv\Scripts\python.exe -m scripts.pipeline.build_product_bm25_index --scope full
.\.venv\Scripts\python.exe -m scripts.pipeline.build_evidence_index --scope full
```

启动后端和前端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn cartwise.api.main:app --reload
.\.venv\Scripts\python.exe -m streamlit run cartwise/ui/app.py
```

API 示例请求：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/recommend `
  -ContentType "application/json" `
  -Body '{"query":"推荐适合家庭录音的便携麦克风支架","top_k":3}'
```

演示验收重点：

- `/health/ready` 返回 ready 后再发起推荐。
- Streamlit 输入自然语言需求后返回 Top K 商品卡片。
- 推荐结果包含召回来源、推荐理由、潜在缺点和评论证据。
- 评论引用只来自对应商品的检索证据。
- LLM 不可用、输出非法或引用非法时触发模板回退。

## 一期评估边界

一期可以严谨记录基于历史交互划分的推荐指标，例如 Popularity 和 LightGCN 的 `Recall@10`、`NDCG@10` 和 `HitRate@10`。自然语言 query 到最终推荐结果的严格离线评估暂不纳入一期，因为当前数据没有 query-level 标注集。

自然语言导购链路的一期验收重点是：

- API 和 Streamlit 可以端到端跑通。
- 结构化硬过滤不会返回明显违反约束的商品。
- Evidence RAG 只在最终候选商品范围内检索评论。
- 中文解释和潜在缺点有对应评论证据或模板回退。
- 延迟瓶颈通过 `cartwise_timing` 日志可定位。

## 测试命令

完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

服务层和回归测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_application_service.py tests/test_recommendation_service.py tests/test_evidence_service.py tests/regression/test_legacy_regression.py
```

API 测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_dependencies.py tests/test_api_lifespan.py
```

## 开发规则摘要

- 开发前先读 `docs/CURRENT_STAGE.md`、`docs/DECISIONS.md`、`docs/DEVELOPMENT_STEPS.md` 和 `docs/PROJECT_PLAN.md`。
- 新业务代码使用 `cartwise.query.llm`、`cartwise.evidence.rag`、`cartwise.recommendation` 和 `cartwise.application`。
- 不要从 `cartwise.core.llm` 或 `cartwise.core.evidence_rag` 写新调用；它们当前只是兼容 wrapper。
- 不要把 `smoke_search_only`、`mode` 或其他历史 smoke 分支重新引入正式服务契约。
- 不要修改阶段 0 fixture 来掩盖行为变化。
- Streamlit 只能通过 HTTP 调用 FastAPI，不直接导入 retrieval、recommendation、evidence 或模型对象。

更多当前阶段要求见 `docs/CURRENT_STAGE.md`。
