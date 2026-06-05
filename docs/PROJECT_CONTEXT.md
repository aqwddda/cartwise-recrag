# CartWise 背景与展示摘要

本文档只用于记录 CartWise 的项目背景、对外介绍和简历表达，不作为当前开发阶段的执行规范。实际开发优先级以 `docs/CURRENT_STAGE.md`、`docs/DEVELOPMENT_STEPS.md` 和 `docs/DECISIONS.md` 为准；如果本文档与这些开发文档存在差异，以开发文档为准。

## Project Goal

CartWise is an evidence-grounded conversational e-commerce recommender system built as a resume project. 中文标题可以写为：基于 RAG 的可解释个性化电商导购系统。系统结合传统推荐模型、搜索检索和评论证据 RAG：推荐模型决定候选商品，商品检索补充当前自然语言需求相关结果，评论证据索引用于支撑推荐解释，LLM 只负责解析用户意图和基于已验证事实生成解释，不允许虚构商品、价格或无证据结论。

## Core User Experience

用户可以用中文或英文提出购物需求，例如“推荐 50 美元以内、适合初学者的吉他调音器，不要 Fender”。系统返回 Top 5 商品，并展示匹配理由、价格与关键属性、真实评论证据、潜在缺点和召回来源。演示版支持 `更便宜`、`换一批` 和排除品牌等轻量多轮操作；这些状态只保存在 FastAPI 进程内存中，不作为长期用户记忆。

## First Release Scope

一期只使用 Amazon Reviews 2023 的 `Musical_Instruments` 单品类数据，统一以 `parent_asin` 作为商品主键。推荐链路包括 Popularity、PyTorch Geometric LightGCN、Dense 商品检索、BM25 商品检索、weighted RRF 候选融合、代码层硬过滤、评论证据检索、LLM 意图解析与解释生成、FastAPI 接口和 Streamlit 演示页面。一期不实现复杂 Agent、Redis 持久化会话、CrossEncoder 重排、商品共购图扩展或独立商品图模型。

## Current Recommendation Pipeline

当前一期链路为：先将用户自然语言需求解析为显式硬约束；中文 query 先通过最小 LLM 翻译层直译为英文，英文 query 直接进入检索；Dense 和 BM25 负责根据当前 query 召回商品；LightGCN 和 Popularity 只作为个性化与热门度补充，并且不能无条件绕过当前 query 相关性；候选通过 weighted RRF 融合后执行价格、品牌、类目、颜色、材料等代码层硬过滤；最终商品再分别检索评论证据；LLM 只使用已提供商品字段和评论证据生成中文解释；引用 ID 和商品 ID 在输出前再次校验，失败时使用确定性模板回退。

## Data and Indexes

CartWise 保留两类索引：商品索引用于候选召回，输入字段包括标题、品牌、主类目、类目路径、特征、详情和描述；评论证据索引用于最终商品的解释支撑，输入字段包括评论正文、评分、时间、购买验证和 helpful votes。商品索引和评论证据索引必须分离，不能用评论证据检索直接决定推荐商品。评论证据一期默认使用 `70-14` 容量方案：每个商品最多保留 70 条文本非空评论，其中最多优先保留 14 条 `rating <= 3` 的中低评分评论。

## Evaluation

离线推荐评估使用时间顺序划分，避免未来数据泄漏。至少报告 Popularity、LightGCN 和最终融合链路的 Recall@10、NDCG@10 和 HitRate@10。系统层面还需要报告本地推荐检索链路 P50/P95 延迟、包含外部 LLM 的端到端 P50/P95 延迟，以及推荐解释中评论引用的一致性人工抽检结果。

## Future Work

二期或后续实验可以继续验证 bought-together 共购图扩展、CrossEncoder 或 ColBERT 重排、ESCI-US 搜索评估、Dense 模型微调、BM25 优化、Redis 会话存储、冷启动分桶分析和更强 Embedding 模型。这些方向默认只记录在 `docs/FUTURE_IMPROVEMENTS.md` 中，除非用户明确指定对应 FI 编号，否则不得提前进入一期开发。

## Resume Description Draft

CartWise：基于 Amazon Reviews 2023 乐器类 5-core 数据构建可解释个性化电商导购系统，融合 LightGCN 协同过滤、Dense/BM25 混合检索、weighted RRF 候选融合、结构化硬过滤和评论证据 RAG，实现支持轻量多轮需求更新与可追溯评论引用的推荐链路；通过 Recall@10、NDCG@10、HitRate@10、引用一致性抽检和 P95 延迟评估各模块效果，并使用 FastAPI、Qdrant 与 Streamlit 完成本机可复现 Demo。

## Repository Name

Suggested repository name: `cartwise-recrag`
