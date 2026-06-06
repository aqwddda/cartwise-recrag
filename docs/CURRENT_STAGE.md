# CartWise 当前阶段

本文档是 Agent/Codex 执行任务前的第一入口。当前阶段的具体实现规则以本文档为准；`docs/DEVELOPMENT_STEPS.md` 记录历史阶段和下一步开发顺序，`docs/PROJECT_PLAN.md` 记录 MVP 路线，关键架构决策见 `docs/DECISIONS.md`。

## 上一阶段摘要

阶段 0 到阶段 7 的结构重构和收尾已完成。当前源码已经建立 Query、Catalog、Retrieval、Recommendation、Evidence 和 Application 的分层结构：

- `cartwise.query`：查询翻译、意图解析和查询约束类型。
- `cartwise.catalog`：商品文档构造等 catalog 共享逻辑。
- `cartwise.retrieval`：BM25、Dense、Popularity、LightGCN、过滤和 Fusion。
- `cartwise.recommendation`：正式推荐链路服务，编排意图解析、约束生成、四路召回、过滤和 Fusion。
- `cartwise.evidence`：评论证据 RAG、EvidenceService 和证据类型。
- `cartwise.application`：未来 API 应调用的最上层业务入口，串联 RecommendationService 与 EvidenceService。

历史 Stage 8 smoke 的 search-only 行为已经迁移到脚本侧 `scripts/tools/stage8_smoke_adapter.py`。该 adapter 不属于正式业务服务，未来 FastAPI 不得调用它。

## 当前阶段

当前阶段：阶段 10：Streamlit 前端。

阶段 9 FastAPI 接入已经通过代码层验收：HTTP schema、fake service 测试、真实 `RecommendationApplicationService` 启动期 composition root、readiness 状态和 API 回归测试均已完成。阶段 10 的目标是实现只通过 HTTP 调用 FastAPI 的 Streamlit 单轮演示页面。不要在阶段 10 中重写推荐、召回、Evidence、Application Service、Prompt、Fusion、过滤规则、Qdrant collection、模型参数或阶段 0 fixture。

## 当前真实边界

- `RecommendationService` 负责正式推荐链路：原始 query -> query 翻译/意图解析 -> `FilterConstraints` -> Dense/BM25/Popularity/LightGCN 召回 -> 过滤与 Fusion。
- `EvidenceService` 只基于最终候选商品执行评论证据检索和解释生成；它保持多候选批量调用语义，不逐商品触发多次解释调用。
- `RecommendationApplicationService` 是未来 FastAPI 的主要业务入口，负责顺序调用 RecommendationService 和 EvidenceService，并组织结构化应用结果。
- `cartwise/core/llm.py` 和 `cartwise/core/evidence_rag.py` 当前只是兼容 wrapper。新业务代码不得继续从这两个路径导入。
- `cartwise/core/config.py` 仍是有效配置模块，不属于废弃文件。
- 正式服务契约中不得重新引入 `smoke_search_only`、`mode` 或其他历史 smoke 分支。

## 阶段 10 目标

实现面向用户的 Streamlit 单轮推荐演示页，而不是后端调试面板。Streamlit 只能作为 HTTP 客户端调用 FastAPI，不得导入 retrieval、recommendation、evidence、模型对象、Qdrant client、LLM client、RecommendationService、EvidenceService 或 RecommendationApplicationService。页面只实现单轮推荐展示，不做登录、数据库、Redis、多轮会话、Agent、部署能力或复杂前端功能。

本阶段建议新增或修改文件限定为：

```text
cartwise/ui/app.py
cartwise/ui/api_client.py
cartwise/ui/README.md
tests/test_ui_api_client.py
```

如果已有 Streamlit 入口文件，应优先复用，不要创建重复入口。不要修改 FastAPI 文件。若发现现有 API 字段无法支撑页面展示，应先停止并汇报，不要擅自改后端。

Streamlit 前端只能调用：

```text
GET /health/ready
POST /api/v1/recommend
```

默认后端地址使用：

```text
http://127.0.0.1:8000
```

同时必须支持通过环境变量 `CARTWISE_API_BASE_URL` 或 Streamlit sidebar 配置覆盖。

## 阶段 10 UI 要求

页面应像一个“智能乐器购物推荐助手”，而不是 API response viewer。首屏应包含清晰标题、简短说明、搜索输入区和推荐按钮。页面文案可以使用英文界面，保持专业、简洁。

顶部区域显示：

```text
CartWise
AI-powered music gear recommendations from product data and reviews
```

搜索区域放在页面上方，包含较大的 query 输入框。输入框 placeholder：

```text
Describe what you need, e.g. “a quiet guitar practice setup for my apartment”
```

`top_k` 数字输入默认 5，范围 1 到 50。`user_id` 放在 sidebar 的 Advanced options 中，不要放在主搜索区干扰普通用户。后端 API 地址也放在 sidebar 中。sidebar 同时显示 backend readiness 状态。

后端 readiness 展示应用户友好。如果 `/health/ready` 是 ready，显示 “Backend ready”。如果后端未启动、连接失败或 ready 返回 503，显示清晰提示：

```text
Backend is not ready. Start FastAPI with:
uvicorn cartwise.api.main:app --reload
```

并将后端返回的 `initialization_error` 放在展开区域中。页面不得因 readiness 失败崩溃。

点击推荐按钮后显示 spinner：

```text
Finding products and reading review evidence...
```

请求成功后，页面顶部显示简洁摘要：

```text
Found 5 recommendations in 32.1s
```

如果 `search_query` 与用户原始 query 不同，可以在小字中显示 “Search query used: ...”，但不要让它抢占主视觉。

推荐结果卡片应突出商品标题、品牌、价格、排名、推荐理由、潜在不足和评论证据摘要。不要把后端字段原样堆出来。`search_query`、`applied_constraints`、`source_scores`、`raw response` 这类调试信息应放在可展开区域中，而不是默认展示在主页面。

每个商品主卡片默认展示：

```text
rank
title
brand
price
reason
potential_cons
```

对于价格为空，显示 `Price unavailable`，不要显示 `None`。对于品牌为空，显示 `Unknown brand` 或直接省略。标题过长时可以正常换行，不要截断到无法理解。

评论证据放在卡片内部的 `Review evidence` expander 中。每条证据显示 rating、chunk_text 或 text、score，不展示过多 metadata。没有 evidence 时显示：

```text
No review evidence returned for this item.
```

`source_ranks`、`source_scores`、`sources` 这些算法细节不要作为主视觉，但可以放到 “Retrieval details” expander 里。普通用户默认看到的是推荐理由，而不是 dense/bm25 分数。

如果 results 为空，显示正常空状态：

```text
No recommendations found. Try a broader query or remove constraints.
```

如果 diagnostics 非空，用 “System notes” 折叠区域展示，用于调试 fallback、部分失败或可恢复错误。不要让 diagnostics 作为主页面重点。页面底部可以提供默认关闭的 “Developer details” expander 展示 raw response。

## 阶段 10 API Client 要求

`cartwise/ui/api_client.py` 负责封装 HTTP 请求，不要在 `app.py` 中散落写请求细节。轻量 API client 至少包含：

```text
check_ready()
recommend(query, user_id=None, top_k=5)
```

API client 应设置合理 timeout，例如 90 秒，因为当前后端可能包含 Evidence RAG 和 LLM 解释生成。client 应正确处理连接失败、超时、非 2xx 响应、后端返回 503、422 和 500，并把错误转换成 UI 可展示的结构化错误信息。不要让底层 HTTP 异常直接冒泡到 Streamlit 页面。

API 返回错误时，页面应给出友好提示。422 表示输入不合法，应提示用户检查 query 和 top_k。503 表示后端未就绪或依赖不可用，应提示检查 FastAPI、Qdrant、模型、索引或 LLM key。500 表示后端内部错误，应提示查看后端日志。不要直接把 Python traceback 显示给用户。

## API 设计规则

- 请求 schema 只暴露当前后端已有能力，例如 `query`、可选 `user_id`、`top_k`。
- 请求 schema 不得暴露 `smoke`、`debug`、`mode`、Fusion 内部权重、召回通道开关或其他内部算法模式。
- 响应 schema 应从 Application Service 结果转换得到，不得直接原样返回完整 `ApplicationRecommendationResult`。
- 响应应避免暴露完整中间对象、完整候选池、大型 evidence payload 或内部诊断结构。
- FastAPI 路由不得直接调用 Dense、BM25、LightGCN、Popularity、Fusion、Qdrant 或 LLM。
- FastAPI lifespan 或独立 composition root 负责一次性初始化 Dense 模型、BM25 索引、Popularity、LightGCN、Qdrant client、LLM client、RecommendationService、EvidenceService 和 Application Service。
- `/health/live` 只表达进程存活。
- `/health/ready` 表达 Qdrant、模型、索引、LLM 配置和服务实例是否可用；不可用时应返回可诊断状态，不应让健康检查崩溃。

## 禁止事项

- 不启动 Web 服务作为本阶段文档更新的一部分。
- 不删除 `cartwise/core/llm.py`、`cartwise/core/evidence_rag.py` 或 `cartwise/core/config.py`。
- 不修改阶段 0 冻结 fixture 来掩盖行为变化。
- 不重建 Qdrant collection、模型、索引或数据。
- 不升级依赖。
- 不把 Stage8SmokeAdapter 接入 FastAPI 或 Application Service。

## 下一步任务

1. 已阅读 `cartwise/ui/README.md`、`cartwise/api/schemas.py`、`cartwise/api/main.py` 和 `README.md` 中的 API 启动说明。
2. 已新增 `cartwise/ui/api_client.py`，封装 `GET /health/ready` 和 `POST /api/v1/recommend`。
3. 已新增 `cartwise/ui/app.py`，实现 Streamlit 页面通过 HTTP 调用 FastAPI。
4. 已实现页面展示用户 ID、query、Top K、推荐商品、召回来源、推荐理由、潜在缺点、评论证据、请求耗时和错误状态。
5. 已新增 `tests/test_ui_api_client.py`，只测试 API client，不启动 Streamlit 页面，不依赖真实 FastAPI、Qdrant 或 LLM。
6. 已更新 `cartwise/ui/README.md`，说明如何启动后端和前端，以及 Streamlit 只通过 HTTP 调用 FastAPI 的边界。
7. 已保持 Streamlit 与后端边界清晰：UI 未直接导入或调用 RecommendationService、EvidenceService、Qdrant、LLM 或模型对象。

## 阶段 10 完成摘要

阶段 10 Streamlit 前端代码层任务已完成：

- `cartwise/ui/api_client.py`：新增轻量 HTTP client，默认后端地址为 `http://127.0.0.1:8000`，支持 `CARTWISE_API_BASE_URL` 在页面侧覆盖；client 使用 `httpx` 调用 `/health/ready` 和 `/api/v1/recommend`，设置 90 秒 timeout，并通过 `trust_env=False` 避免本地 127.0.0.1 请求走环境代理。
- `cartwise/ui/app.py`：新增 Streamlit 单轮推荐页面，主页面展示 query 输入、Top K、推荐卡片、推荐理由、潜在缺点、评论证据、请求耗时、错误状态、System notes 和 Developer details；sidebar 展示 API base URL、backend readiness 和 Advanced options 中的 user_id。
- `cartwise/ui/README.md`：更新后端和前端启动说明，明确必须先确认 `/health/ready` ready，并说明 UI 只通过 HTTP 调用 FastAPI。
- `tests/test_ui_api_client.py`：新增 API client 单元测试，覆盖 ready 成功、ready 503、连接失败、recommend 成功、recommend 422、recommend 503、recommend 500 和 recommend timeout。

本阶段未修改 FastAPI 后端接口、Application Service、RecommendationService、EvidenceService、retrieval、Prompt、Fusion、过滤规则、Qdrant collection、模型参数或阶段 0 fixture。未启动真实 Web 服务；端到端联调仍取决于本机 Qdrant、collection、数据文件、BM25、LightGCN 模型和 LLM Key 是否齐全。

## 阶段 9 验收摘要

阶段 9 已完成：

- 新增 `cartwise/api/schemas.py`，定义 `RecommendRequest`、推荐响应、证据响应、诊断响应和 health 响应 schema。
- 更新 `cartwise/api/main.py`，提供 `create_app(application_service=...)` 测试注入入口、`GET /health/live`、`GET /health/ready` 和 `POST /api/v1/recommend`。
- 保留兼容 `/health`，并继续让 Qdrant health check 绕过环境代理。
- API 路由只调用注入的 Application Service，并通过 `ApplicationRecommendationRequest` 与应用层交互。
- `POST /api/v1/recommend` 已处理请求校验、服务未初始化、业务 `ValueError`、重资源不可用类异常和未知异常。
- `GET /health/ready` 当前以 app 中是否存在 Application Service 实例作为就绪核心判断，不在 readiness 请求中初始化重资源。
- 新增 `tests/test_api.py`，使用 fake Application Service 返回真实 `ApplicationRecommendationResult` 形状，覆盖 schema、路由、错误处理、readiness 和内部对象裁剪。
- 新增 `cartwise/application/factory.py`，提供真实 `build_application_service()` composition root，按启动期一次性构造 `RecommendationApplicationService`。
- `cartwise/api/main.py` 已接入 FastAPI lifespan：默认 app 启动时构造真实服务，成功后写入 `app.state`；失败时保留初始化错误并让 `/health/ready` 返回 503，不在请求路径重复初始化。
- 真实 builder 复用正式 `RecommendationService`、`EvidenceService` 和 `RecommendationApplicationService`，不导入或调用 `Stage8SmokeAdapter`。
- 新增 `tests/test_api_dependencies.py` 和 `tests/test_api_lifespan.py`，通过 monkeypatch fake 掉重资源，验证 builder 依赖关系、startup success/failure、fake service 注入和请求复用启动期服务。
- FastAPI builder 默认设备已改为 `cpu`，仍可通过 `ApplicationServiceBuildConfig(device="cuda")` 显式请求 CUDA。
- Product Dense Qdrant collection 命名已抽到轻量模块 `cartwise/retrieval/collection_names.py`，API builder 不再为命名导入 `cartwise.retrieval.dense`。
- Evidence Qdrant collection 命名已抽到轻量模块 `cartwise/evidence/collections.py`，API builder 与 `scripts/pipeline/build_evidence_index.py` 共用同一命名规则。
- `cartwise/application/factory.py` 已延迟导入 pyarrow、OpenAI、Qdrant、Dense、BM25、LightGCN、Evidence RAG 等生产依赖；fake-service API 测试不需要触发真实 builder 或加载重资源。

阶段 9 未启动真实 uvicorn 服务做外部联调；真实 ready 状态仍取决于本机 Qdrant、collection、数据文件、BM25、LightGCN 模型和 LLM Key 是否齐全。该项留到阶段 10 UI 联调前检查，不通过修改业务逻辑兜底。

## 验收命令

阶段 10 初始验收命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_dependencies.py tests/test_api_lifespan.py
```

完成 Streamlit 页面后，应至少回归 API 测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_dependencies.py tests/test_api_lifespan.py
```

完成 API client 后，应先运行 UI client 测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ui_api_client.py -q --basetemp="$env:TEMP\cartwise-pytest-ui"
```

如修改了 API schema、Application Service 边界或共享响应字段，应运行完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 最近成功状态

阶段 10 Streamlit API client 测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ui_api_client.py -q --basetemp="$env:TEMP\cartwise-pytest-ui"
```

结果：

```text
8 passed
```

阶段 10 API 回归测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_dependencies.py tests/test_api_lifespan.py -q --basetemp="$env:TEMP\cartwise-pytest-api"
```

结果：

```text
21 passed
```

阶段 9 collection 命名轻量模块收尾测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api_dependencies.py tests/test_api_lifespan.py tests/test_api.py -q --basetemp="$env:TEMP\cartwise-pytest-api"
```

结果：

```text
21 passed
```

阶段 9 collection 命名轻量模块收尾后完整测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp="$env:TEMP\cartwise-pytest-full"
```

结果：

```text
145 passed, 3 warnings
```

阶段 9 builder 收尾测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_dependencies.py tests/test_api_lifespan.py -q --basetemp="$env:TEMP\cartwise-pytest-api"
```

结果：

```text
20 passed
```

阶段 9 builder 收尾后完整测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp="$env:TEMP\cartwise-pytest-full"
```

结果：

```text
144 passed, 3 warnings
```

阶段 9 真实 builder 和 lifespan 测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py tests/test_api_dependencies.py tests/test_api_lifespan.py -q --basetemp="$env:TEMP\cartwise-pytest-api"
```

结果：

```text
16 passed, 3 warnings
```

阶段 9 真实 builder 接入后完整测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp="$env:TEMP\cartwise-pytest-full"
```

结果：

```text
140 passed, 3 warnings
```

阶段 9 API fake-service 测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -q --basetemp="$env:TEMP\cartwise-pytest-api"
```

结果：

```text
11 passed
```

阶段 9 API 接口层改动后完整测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp="$env:TEMP\cartwise-pytest-full"
```

结果：

```text
135 passed, 3 warnings
```

结构重构收尾后完整测试通过：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
124 passed, 3 warnings
```

阶段 0 确定性回归通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/regression/test_legacy_regression.py
```

结果：

```text
2 passed, 3 warnings
```

最近成功 Stage 8 smoke：

```powershell
.\.venv\Scripts\python.exe -m scripts.tools.run_stage8_smoke --scope full --query "guitar tuner for beginners" --top-k 5 --dense-k 10 --bm25-k 10 --device cuda
```

最近成功报告：

```text
artifacts/reports/stage8_smoke/20260606_001528_guitar_tuner_for_beginners.json
artifacts/reports/stage8_smoke/20260606_001528_guitar_tuner_for_beginners.txt
```
