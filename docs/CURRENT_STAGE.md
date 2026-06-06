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

当前阶段：阶段 9：FastAPI 接入。

阶段 9 的目标是实现正常项目级别的 FastAPI 单轮推荐接口，把已经提取出的 Application Service 接到 HTTP 层。不要把本阶段做成临时 demo，也不要扩展新的推荐算法、召回通道、多轮会话、Streamlit 页面、Redis、Agent、CrossEncoder 或部署能力。

## 当前真实边界

- `RecommendationService` 负责正式推荐链路：原始 query -> query 翻译/意图解析 -> `FilterConstraints` -> Dense/BM25/Popularity/LightGCN 召回 -> 过滤与 Fusion。
- `EvidenceService` 只基于最终候选商品执行评论证据检索和解释生成；它保持多候选批量调用语义，不逐商品触发多次解释调用。
- `RecommendationApplicationService` 是未来 FastAPI 的主要业务入口，负责顺序调用 RecommendationService 和 EvidenceService，并组织结构化应用结果。
- `cartwise/core/llm.py` 和 `cartwise/core/evidence_rag.py` 当前只是兼容 wrapper。新业务代码不得继续从这两个路径导入。
- `cartwise/core/config.py` 仍是有效配置模块，不属于废弃文件。
- 正式服务契约中不得重新引入 `smoke_search_only`、`mode` 或其他历史 smoke 分支。

## 阶段 9 目标

优先实现以下接口：

```text
GET  /health/live
GET  /health/ready
POST /api/v1/recommend
```

可保留现有 `/health` 作为兼容 liveness 入口，但 readiness 必须独立表达重资源是否已经准备好。

阶段 9 应新增或完善：

```text
cartwise/api/schemas.py
cartwise/api/main.py
tests/test_api.py
```

实现顺序：

1. 定义 FastAPI schema，先用 fake Application Service 覆盖 HTTP 行为和错误处理。
2. 设计依赖注入和 composition root，使重资源在启动生命周期或显式构造阶段一次性初始化。
3. 接入真实 `RecommendationApplicationService`，路由层只调用 Application Service。
4. 覆盖正常请求、非法输入、LLM fallback、Qdrant 或重资源不可用、readiness 状态等测试。

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

- 不实现 Streamlit 页面。
- 不启动 Web 服务作为本阶段文档更新的一部分。
- 不删除 `cartwise/core/llm.py`、`cartwise/core/evidence_rag.py` 或 `cartwise/core/config.py`。
- 不修改阶段 0 冻结 fixture 来掩盖行为变化。
- 不重建 Qdrant collection、模型、索引或数据。
- 不升级依赖。
- 不把 Stage8SmokeAdapter 接入 FastAPI 或 Application Service。

## 下一步任务

1. 在真实本机环境下启动 `uvicorn cartwise.api.main:app --reload`，检查 Qdrant、模型、BM25、LightGCN、Evidence collection 和 LLM 配置是否全部可用。
2. 使用真实 API 请求验证 `/health/ready` 和 `POST /api/v1/recommend`，记录成功样例或初始化失败原因。
3. 若真实环境缺少数据、索引、模型、Qdrant collection 或 LLM Key，只补齐运行环境，不修改推荐、召回、Evidence 或 Application 业务逻辑。

## 阶段 9 当前进展

已完成 FastAPI 接口层第一步：

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
- Evidence Qdrant collection 命名已合并到轻量公共模块 `cartwise/evidence/types.py`，API builder 与 `scripts/pipeline/build_evidence_index.py` 共用同一命名规则。
- `cartwise/application/factory.py` 已延迟导入 pyarrow、OpenAI、Qdrant、Dense、BM25、LightGCN、Evidence RAG 等生产依赖；fake-service API 测试不需要触发真实 builder 或加载重资源。

## 验收命令

阶段 9 初始验收命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py
```

阶段 9 完成前应同时回归服务层和阶段 0 基线：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_application_service.py tests/test_recommendation_service.py tests/test_evidence_service.py tests/regression/test_legacy_regression.py tests/test_api.py
```

如修改了推荐、Evidence 或 Application 服务边界，应运行完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 最近成功状态

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
