# CartWise 当前阶段

本文档是 Agent/Codex 执行任务前的第一入口。当前阶段的具体实现规则以本文档为准；`docs/DEVELOPMENT_STEPS.md` 只作为历史阶段索引和粗粒度阶段说明参考。全局架构见 `docs/PROJECT_PLAN.md`，关键决策见 `docs/DECISIONS.md`。

## 上一阶段摘要

阶段 8 已完成：已实现独立评论证据 Dense 索引脚本、候选商品范围内的评论证据 RAG、结构化中文解释生成、引用校验、模板回退和只读真实链路 smoke 报告；当前 Qdrant 仍按用户要求只读，未执行 collection 重建。

## 当前阶段

当前阶段：阶段 9：完整 API。

阶段 9 只实现 FastAPI 单轮推荐接口，把现有阶段 7 fusion 链路和阶段 8 评论证据解释链路封装成后端 API。不要提前实现阶段 10 Streamlit、多轮会话、Redis、复杂 Agent、CrossEncoder、评论 BM25 或二期图扩展。

## 当前目标

完成阶段 9 的三个模块：

```text
cartwise/api/schemas.py
cartwise/api/main.py
tests/test_api.py
```

- `cartwise/api/schemas.py`：定义健康检查、推荐请求和推荐响应的 Pydantic schema。
- `cartwise/api/main.py`：提供 FastAPI app、`GET /health` 和 `POST /api/v1/recommend`。
- `tests/test_api.py`：覆盖健康检查、正常推荐请求、非法输入、LLM fallback 和 Qdrant 不可用等 API 行为。

## 阶段 9 详细规则

### 接口范围

- 阶段 9 只提供单轮推荐请求，不维护会话状态。
- 本阶段接口为：

```text
GET  /health
POST /api/v1/recommend
```

- 先使用可控假数据或可注入依赖验证接口 schema、错误处理和响应结构，再接入真实推荐链路。
- 不新增阶段 9 之外的前端页面或多轮反馈接口。

### 推荐链路

- 中文推荐请求必须先经过既有最小 LLM 翻译层转成英文 query，再进入 Dense/BM25 检索链路；翻译失败时明确报错，不把中文原文静默传给英文检索模型。
- 英文 query 直接进入 Dense/BM25，不调用翻译。
- 商品召回、硬过滤、候选融合继续复用阶段 7 已有模块。
- 评论证据和中文解释继续复用阶段 8 `cartwise/core/evidence_rag.py`。
- 评论 RAG 只能基于 fusion 后的最终候选商品检索证据，不能引入新的商品候选。
- Qdrant 当前只能读取，不得在 API 启动或请求处理中重建、覆盖或修改 collection。

### 响应要求

推荐响应至少能表达：

- `parent_asin`
- 商品标题、品牌、价格
- 融合分数或排序分数
- 召回来源，例如 `dense`、`bm25`
- 中文推荐理由
- 中文潜在缺点
- 引用评论证据，包括 `review_id`、`rating` 和评论摘录
- 当前请求应用到的结构化约束

当过滤后不足 5 个商品时，返回实际数量，不返回违反硬约束的商品。

### 异常与回退

- `/health` 不应因 Qdrant 或 LLM 不可用而崩溃；依赖不可用时返回 HTTP `200`，并在状态字段中标记为 `unavailable`。
- 推荐请求参数非法时返回明确的 4xx 响应。
- LLM 解释不可用、输出非法或引用非法时，必须返回阶段 8 的确定性模板解释。
- Qdrant 不可用时，推荐接口应返回可诊断错误或受控 fallback，不能抛出未处理异常。

### 验收标准

- 可以提交中文推荐请求。
- 可以返回 Top 5 商品、召回来源、评论证据和中文解释。
- 请求中的价格、品牌和属性约束通过后端链路执行。
- LLM 不可用时仍然返回合法结果和模板解释。
- `/health` 能报告 API、Qdrant、推荐模型和 LLM 状态。
- API 测试覆盖正常请求、非法请求、LLM fallback 和 Qdrant 不可用。

## 当前卡点

无新的阶段 9 卡点。开始编码前需要先读取现有配置、LLM 翻译层、阶段 7 fusion 模块和阶段 8 evidence RAG 模块，确定 API 依赖注入方式。

## 下一步任务

1. 阅读 `cartwise/core/config.py`、阶段 7 召回/融合模块、阶段 8 `cartwise/core/evidence_rag.py` 和现有测试风格。
2. 新建 API schema 和 FastAPI app，先用测试注入的假推荐服务跑通接口。
3. 接入真实单轮推荐链路，并保持 Qdrant 只读。
4. 增加 `tests/test_api.py` 覆盖阶段 9 验收场景。

## 验收命令

阶段 9 初始验收命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py
```

阶段 9 完成前应同时回归阶段 7/8 关键测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_fusion.py tests/test_evidence_rag.py tests/test_api.py
```

## 最近成功状态

阶段 8 验收命令已通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_evidence_index.py tests/test_evidence_rag.py tests/test_fusion.py
```

阶段 8 最近成功 smoke：

```powershell
.\.venv\Scripts\python.exe -m scripts.tools.run_stage8_smoke --scope full --query "guitar tuner for beginners" --top-k 5 --dense-k 10 --bm25-k 10 --device cuda
```

最近成功报告：

```text
artifacts/reports/stage8_smoke/20260605_214621_guitar_tuner_for_beginners.json
artifacts/reports/stage8_smoke/20260605_214621_guitar_tuner_for_beginners.txt
```

## 提交说明

阶段 9 完成后建议提交：

```powershell
git add .
git commit -m "feat: add recommendation api"
```

提交前必须重新检查 `git status`，不要提交 `.env`、数据文件、索引文件、模型权重、缓存文件或真实 API Key。
