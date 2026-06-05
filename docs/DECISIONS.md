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

商品候选检索和评论证据检索必须分离。先决定推荐商品，再为最终商品检索评论证据。评论证据一期默认采用 `70-14` 方案：每个商品最多保留 70 条文本非空评论，其中最多优先保留 14 条 `rating <= 3` 的中低评分评论。Top 5 商品各自最多返回 3 条证据。推荐理由和潜在缺点必须基于商品字段或检索评论，输出前必须校验商品 ID 和引用 ID。

## 环境与安全

当前开发机使用 Python `3.12.9`，显卡为 GTX 1660 Ti 6GB，已验证 `torch==2.12.0+cu126` 和 `torch-geometric==2.7.0` 可以完成 LightGCN GPU 链路。请求 CUDA 但 CUDA 不可用时必须立即报错，不能静默回退 CPU。下载依赖、访问外部 API 或拉取数据时使用代理 `http://127.0.0.1:9508`，访问 `127.0.0.1` 和 `localhost` 时绕过代理。真实 API Key 只能放入未提交的 `.env` 文件。
