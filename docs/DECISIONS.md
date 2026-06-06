# CartWise 关键决策记录

本文档只记录已经确定、会影响后续开发实现的关键决策。它用于减少 Agent/Codex 被旧方案、未来方案或重复文档干扰。若本文档与 `docs/PROJECT_CONTEXT.md` 存在差异，以本文档和 `docs/DEVELOPMENT_STEPS.md` 为准。

## 文档优先级

执行任务前优先读取 `docs/CURRENT_STAGE.md`，确认当前阶段和下一步。具体开发步骤以 `docs/DEVELOPMENT_STEPS.md` 对应阶段为准。全局设计以 `docs/PROJECT_PLAN.md` 为准。后续想法只记录在 `docs/FUTURE_IMPROVEMENTS.md`，默认不得主动实现。`docs/PROJECT_CONTEXT.md` 只用于项目背景和简历表达，不作为当前开发 spec。

## 一期范围

一期目标是本机可复现的最小完整闭环：Amazon Reviews 2023 `Musical_Instruments` 数据处理、Popularity、LightGCN、Dense/BM25 商品检索、weighted RRF 候选融合、代码层硬过滤、评论证据 RAG、LLM 意图解析与解释生成、FastAPI、Streamlit、离线指标和本机延迟报告。一期不实现复杂 Agent、CrossEncoder、Redis 持久化会话、商品共购图扩展、SASRec 或独立商品图模型。

## 数据与主键

一期只使用 Amazon Reviews 2023 的 `Musical_Instruments` 单品类数据，统一使用 `parent_asin` 作为商品主键。只将官方 5-core 中出现的商品纳入一期可推荐目录，再从原始元数据和评论中补齐商品文本、价格、品牌、详情和评论证据。训练、验证和测试按时间划分，验证和测试边不得进入训练图。

## 商品检索

商品索引文本字段顺序固定为 `Title -> Brand -> Main Category -> Categories -> Features -> Details -> Description`。E5、BLaIR 和 BM25 复用同一份基础商品文档。E5 和 BLaIR 分别建立独立 Qdrant collection，不能混用向量。阶段 7 主推荐链路默认只使用 E5 作为 Dense 通道；BLaIR 保留为阶段 6 召回审核和对比实验通道。BM25 建立本地持久化索引。

## Query 处理

英文 query 直接进入 Dense 和 BM25，不调用 LLM 翻译。中文 query 先调用最小 LLM 翻译层直译为英文，再进入相同检索链路。翻译提示词只要求返回英文译文，不做结构化输出、复杂查询改写或推荐理由生成。翻译失败时明确报错，不将中文原文静默传给英文检索模型。

## LLM 意图解析

阶段 7 的 LLM intent parser 只抽取显式硬约束，包括 `product_terms`、`brands`、`excluded_brands`、`min_price`、`max_price`、`color_tags` 和 `material_tags`。它不改写 Dense/BM25 召回 query，不生成推荐理由，不读取完整品牌表或类目表，也不使用本地正则补充价格解析。LLM 输出必须经过 Pydantic schema 校验；未命中离线映射表的品牌和商品核心词直接丢弃。

## 候选融合与过滤

带自然语言 query 的导购推荐以 Dense 和 BM25 为主要搜索召回通道。LightGCN 和 Popularity 只提供个性化与热门度补充，不能无条件绕过当前 query 相关性进入最终候选池。候选融合使用 weighted RRF；已知用户默认权重为 Dense `0.45`、BM25 `0.25`、LightGCN `0.25`、Popularity `0.05`，冷启动用户默认权重为 Dense `0.65`、BM25 `0.30`、Popularity `0.05`。

## 评论证据 RAG

商品候选检索和评论证据检索必须分离。先决定推荐商品，再为最终商品检索评论证据。评论证据一期默认采用 `70-14` 方案：每个商品最多保留 70 条文本非空评论，其中最多优先保留 14 条 `rating <= 3` 的中低评分评论。评论证据索引使用独立 Qdrant collection 和 E5 embedding，按 `chunk_size=384`、`chunk_overlap=64` 切分；在线解释时每个最终商品先召回 10 个评论 chunk，目标收集 5 条不同 `review_id`，不足时最多扩到 20 个 chunk，并优先补入 1 条 `rating <= 3` 的中低评分评论。推荐理由和潜在缺点必须基于商品字段或检索评论，输出前必须校验商品 ID 和引用 ID；LLM 输出只能引用当前商品已检索到的 `review_id`，非法输出必须模板回退。

## 结构重构后的服务边界

当前源码已经完成阶段 0 到阶段 7 的结构重构，正式业务边界按以下层次执行：

- `cartwise.query` 负责 query 翻译、意图解析和查询约束类型。
- `cartwise.catalog` 负责商品文档构造等 catalog 共享逻辑。
- `cartwise.retrieval` 负责 BM25、Dense、Popularity、LightGCN、过滤和 Fusion 算法。
- `cartwise.recommendation.RecommendationService` 负责正式推荐链路编排。
- `cartwise.evidence.EvidenceService` 负责对最终候选商品批量执行评论证据检索和解释生成。
- `cartwise.application.RecommendationApplicationService` 是未来 FastAPI 的最上层业务入口。

该边界的目的不是增加抽象层，而是防止 API、UI 或脚本重新拼装底层召回器、Qdrant、LLM 和 Fusion 逻辑。FastAPI 路由必须调用 Application Service，不得直接调用 Dense、BM25、LightGCN、Popularity、Fusion、Qdrant 或 LLM。

## 兼容 wrapper 决策

`cartwise/core/llm.py` 和 `cartwise/core/evidence_rag.py` 当前只作为兼容 re-export wrapper 保留。保留原因是旧测试、legacy regression harness 和潜在历史调用方仍需要旧路径稳定存在；立即删除会把结构重构和调用方清理混在一起，增加回归风险。新业务代码必须使用 `cartwise.query.llm` 和 `cartwise.evidence.rag`，不得继续从 wrapper 路径导入。

`cartwise/core/config.py` 仍是有效配置模块，不属于废弃文件。配置迁移到顶层 `cartwise/config.py` 可以作为 MVP 后清理项，但在 FastAPI 接入前不应为了目录形式移动配置，避免扩大 import 变更面。

## Stage 8 smoke 兼容适配器

历史 `run_stage8_smoke.py` 的 search-only 行为不属于正式推荐服务契约。该流程不执行 intent parser，不调用 Popularity 或 LightGCN，只使用 Dense、BM25、空 `FilterConstraints`、Fusion 和 Evidence RAG 生成 smoke 报告。

为避免污染正式服务，历史 smoke 行为已经迁移到 `scripts/tools/stage8_smoke_adapter.py::Stage8SmokeAdapter`。该 adapter 只服务脚本工具，不属于 `cartwise` 业务包，未来 FastAPI 不得调用它。正式 `RecommendationService` 和 `RecommendationApplicationService` 不再接收 `mode`、`smoke_search_only` 或等价控制字段。

## EvidenceService 批量解释调用

`EvidenceService` 必须一次性将全部最终候选传入 `explain_candidates`，保持评论证据 RAG 的多候选批量语义。这样可以避免 Top-K 个商品触发 Top-K 次 LLM 调用，也能保持单次 Prompt 中候选和 evidence 的整体校验能力。不得为了简化调用重新改回逐候选解释。

## FastAPI 接入决策

FastAPI 是正式服务入口，不是召回器装配脚本。路由层只能接收 HTTP 请求、调用已经构造好的 Application Service、把 Application Service 结果转换为对外 schema，并返回错误或诊断状态。重资源初始化应由 FastAPI lifespan、composition root 或测试依赖注入一次性完成；不得在每次请求中加载模型、读取大型索引、创建 Qdrant client 或创建 LLM client。

FastAPI 的请求 schema 只暴露 `query`、可选 `user_id`、`top_k` 等当前后端已有能力。不得暴露 smoke、debug、召回通道开关、Fusion 权重或内部算法模式。响应 schema 必须从 Application Service 结果裁剪转换得到，不得直接原样返回完整内部对象。

## 环境与安全

当前开发机使用 Python `3.12.9`，显卡为 GTX 1660 Ti 6GB，已验证 `torch==2.12.0+cu126` 和 `torch-geometric==2.7.0` 可以完成 LightGCN GPU 链路。请求 CUDA 但 CUDA 不可用时必须立即报错，不能静默回退 CPU。下载依赖、访问外部 API 或拉取数据时使用代理 `http://127.0.0.1:9508`，访问 `127.0.0.1` 和 `localhost` 时绕过代理。真实 API Key 只能放入未提交的 `.env` 文件。
