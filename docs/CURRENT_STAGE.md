# CartWise 当前阶段

本文档是 Agent/Codex 执行任务前的第一入口。当前阶段的具体实现规则以本文档为准；`docs/DEVELOPMENT_STEPS.md` 只作为历史阶段索引和粗粒度阶段说明参考。全局架构见 `docs/PROJECT_PLAN.md`，关键决策见 `docs/DECISIONS.md`。

## 当前阶段

当前阶段：阶段 8：评论证据与解释生成。

阶段 8 只实现评论证据 Dense 索引、候选商品范围内的评论证据 RAG、结构化解释生成和确定性模板回退。不要提前实现阶段 9 API、阶段 10 Streamlit、评论 BM25、CrossEncoder、Redis、复杂 Agent、商品共购图扩展或独立商品图模型。

## 当前目标

完成阶段 8 的三个模块：

```text
scripts/pipeline/build_evidence_index.py
cartwise/core/evidence_rag.py
tests/test_evidence_rag.py
```

- `scripts/pipeline/build_evidence_index.py`：构建独立的评论证据 Dense Qdrant collection。
- `cartwise/core/evidence_rag.py`：在阶段 7 fusion 候选商品范围内检索评论证据，编排 Prompt、调用 LLM、校验结构化输出并在异常时模板回退。
- `tests/test_evidence_rag.py`：覆盖评论证据归属、引用校验、非法输出回退和评论 RAG 边界。

## 已完成

- 已新增 `scripts/pipeline/build_evidence_index.py`，用于读取阶段 2 的 `reviews.parquet` 并构建评论证据索引。
- 离线索引脚本已包含 E5 默认 embedding、独立 collection 命名、Qdrant 写入、稳定 `chunk_id`/point ID、report 和 checkpoint 输出。
- 评论切分已使用 LangChain `RecursiveCharacterTextSplitter`，默认 `chunk_size=384` tokens、`chunk_overlap=64` tokens，并按 embedding tokenizer 计算长度。
- 已新增 `tests/test_build_evidence_index.py`，覆盖稳定 ID、评论文本构造、多 chunk、payload、批量写入和 token 统计。
- `docs/DEVELOPMENT_STEPS.md` 已补充阶段 8 的评论 chunk、在线 RAG、引用和回退规则。
- 已新增 `cartwise/core/evidence_rag.py`，实现候选商品范围内的评论证据检索、渐进式 chunk 召回、低评分评论补充、Prompt 构造、LLM JSON 校验和确定性模板回退。
- 已新增 `tests/test_evidence_rag.py`，覆盖评论检索 query 构造、多 chunk 保留、低评分补充、合法 LLM 引用、非法引用回退和检索越界回退。
- 已修正评论索引 Qdrant payload 字段，使其包含阶段 8 要求的 `title`、`text`、`helpful_vote`、`verified_purchase` 和 `timestamp`。
- 已新增 `scripts/tools/run_stage8_smoke.py`，用于只读运行 Dense/BM25 召回、fusion 排序、评论证据 Qdrant 检索和 DeepSeek 解释生成。
- 已完成一次真实只读 smoke：查询 `guitar tuner for beginners`，读取 full scope 商品和评论 Qdrant collection，返回 Top 5 召回商品，并由 DeepSeek 生成非 fallback 中文解释和可校验 `review_id` 引用。

## 未完成

- 尚未重建 Qdrant 评论证据 collection。当前规则要求 Qdrant 只能读取，不能修改，因此不执行重建或覆盖验证。

## 当前卡点

阶段 8 的离线评论索引脚本、在线评论证据 RAG、单元测试和只读真实链路 smoke 已完成。当前唯一未做的是 Qdrant 评论证据 collection 重建验证；由于当前用户要求 Qdrant 只能读取，不能修改，不应执行重建或覆盖操作。

## 阶段 8 详细规则

### 评论索引规则

- 阶段 8 默认使用 Dense 评论证据索引，不实现评论 BM25 索引。
- `build_evidence_index.py` 读取阶段 2 生成的 `reviews.parquet`，其中评论已经完成 `70-14` 容量控制并带有稳定 `review_id`。
- 评论索引写入 Qdrant 的独立 collection，不能与商品 Dense collection 混用。
- 评论 RAG 使用独立 embedding 配置和独立 Qdrant collection。初始 embedding 模型可以与商品 Dense 同为 E5，但不得复用 `cartwise/retrieval/dense.py` 或 `cartwise/retrieval/bm25.py` 中的商品召回模块。
- 评论索引使用 LangChain `RecursiveCharacterTextSplitter` 切分评论文本，按 embedding tokenizer 计算长度，配置固定为 `chunk_size=384` tokens、`chunk_overlap=64` tokens。
- 未超过 `chunk_size` 的评论保持单 chunk；超过上限的评论切成多个 chunk。多个 chunk 共享同一个 `review_id`，但使用不同 `chunk_id` 写入 Qdrant。
- 评论向量 payload 至少包含 `review_id`、`chunk_id`、`parent_asin`、`rating`、`title`、`text`、`chunk_text`、`helpful_vote`、`verified_purchase` 和 `timestamp`。
- 离线构建时记录索引评论数、商品覆盖数、embedding 模型名、Qdrant collection 名称、构建耗时、chunk 数、发生切分的评论数和 token 长度分布。

### 在线 RAG 规则

- `cartwise/core/evidence_rag.py` 是阶段 8 的评论证据 RAG 模块，负责评论检索、Prompt 编排、LLM 调用、结构化输出校验和模板回退。
- 推荐搜索模块与评论 RAG 模块只通过阶段 7 fusion 后的商品 ID 和商品 metadata 交互。
- 在线检索输入为阶段 7 已用于 Dense/BM25 召回的英文 query、阶段 7 入选商品 `parent_asin` 列表和商品 metadata。中文 query 不在评论 RAG 阶段再次翻译。
- 每个商品的评论检索 query 由英文 query、商品标题和商品类目共同构造，用于提高评论证据与当前商品和用户需求的匹配度。
- 评论检索必须限制在 fusion 传入的商品范围内，不能通过评论索引引入新的商品候选。
- 每个入选商品使用渐进式评论 chunk 召回：先检索 `initial_chunk_k=10` 个 chunk，按检索顺序收集 `final_review_k=5` 个不同 `review_id`；如果不足 5 个，再扩大检索，最多到 `max_candidate_chunk_k=20` 个 chunk。
- 达到 5 个不同 `review_id` 后立即停止；如果检索到 20 个 chunk 后仍不足 5 个 `review_id`，则返回实际可用的评论证据数量。
- 如果同一个 `review_id` 命中多个 chunk，这些命中 chunk 全部保留并传入解释生成 Prompt。
- 评论证据以 Dense 相似度召回为主；每个商品最终评论证据中优先保证至少 1 条 `rating <= 3` 的中低评分评论，用于支持潜在缺点说明。
- 如果主召回结果中没有中低评分评论，则在同一商品和同一 query 下追加一次带 `rating <= 3` 过滤的补充召回。补充结果用于补足不足 5 条的结果，或替换主召回中排名最低的一条证据；如果该商品没有可用中低评分评论，则不强行补充。
- 解释生成 Prompt 使用命中的 `chunk_text` 和评论元数据，不要求放入完整评论正文。

### 解释生成规则

- 阶段 8 在线编排使用 LangChain，但 LangChain 只属于评论 RAG 模块内部。
- LangChain 可以用于评论 retriever 编排、Prompt 模板、LLM 调用和结构化输出解析。
- 不使用 LangChain 接管商品召回、硬过滤、候选融合、API 路由或会话状态。
- 解释生成输入只能来自阶段 7 的融合候选和阶段 8 在候选商品范围内检索到的评论证据。
- 输出必须是结构化结果，每个商品包含 `parent_asin`、`reason`、`potential_cons` 和 `cited_review_ids`。
- `reason` 和 `potential_cons` 必须基于输入商品 metadata 和评论证据，不允许使用候选外商品。
- `cited_review_ids` 只能引用当前商品对应的已检索评论。
- LLM 输出只能引用 `review_id`，不能引用 `chunk_id`。

### LLM 禁止事项

- 不允许增加候选列表外的商品。
- 不允许编造价格、品牌、属性或评论内容。
- 不允许编造评论 ID。
- 不允许引用不属于当前商品的评论。

### 异常处理

以下情况使用固定模板回退：

- LLM 不可用或超时。
- 输出不是合法 JSON。
- 输出不符合结构化 schema。
- 输出引用不存在。
- 输出引用不属于对应商品。
- 输出商品不在候选列表中。
- 评论检索返回候选商品范围外的 `parent_asin`。

### 验收标准

- 每条引用必须属于对应商品。
- 引用 ID 可以稳定复现。
- 评论证据来自实际检索结果。
- 评论 Qdrant collection 与商品 Dense collection 独立。
- 评论 RAG 不复用商品 `dense.py` 和 `bm25.py` 的召回模块。
- LangChain 评论 RAG 只读取传入的候选商品和评论证据。
- 每个解释结果只能引用对应商品的 `review_id`。
- LLM 不可用、输出非法或引用非法时，返回确定性模板解释。

## 下一步任务

下一步：

1. 如需复现只读真实链路，运行 `scripts/tools/run_stage8_smoke.py`，不要传入任何会重建或写入 Qdrant 的参数。
2. 进入阶段 9 前，确认是否接受“Qdrant 只读，不做重建验证”的状态。

涉及文件：

- `scripts/pipeline/build_evidence_index.py`
- `cartwise/core/evidence_rag.py`
- `tests/test_build_evidence_index.py`
- `tests/test_evidence_rag.py`
- `requirements.txt`
- `scripts/paths.py`

验收命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_evidence_index.py tests/test_evidence_rag.py
```

只读真实链路 smoke 命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.tools.run_stage8_smoke --scope full --query "guitar tuner for beginners" --top-k 5 --dense-k 10 --bm25-k 10 --device cuda
```

如需验证真实评论索引构建，应先确认 Qdrant 状态和是否允许重建 collection，然后从仓库根目录执行阶段 8 构建命令。

## 最近成功状态

最近成功命令：`.\.venv\Scripts\python.exe -m pytest tests/test_build_evidence_index.py tests/test_evidence_rag.py tests/test_fusion.py`
最近成功 smoke：`.\.venv\Scripts\python.exe -m scripts.tools.run_stage8_smoke --scope full --query "guitar tuner for beginners" --top-k 5 --dense-k 10 --bm25-k 10 --device cuda`
最近成功产物：`cartwise/core/evidence_rag.py`、`tests/test_evidence_rag.py`、`scripts/tools/run_stage8_smoke.py`，以及已补齐 payload 字段的 `scripts/pipeline/build_evidence_index.py`。
当前卡点：Qdrant 当前只能读取，不能执行重建 collection 或覆盖验证。

## 提交说明

阶段 8 完成后建议提交：

```powershell
git add .
git commit -m "feat: add evidence-backed explanations"
```

提交前必须重新检查 `git status`，不要提交 `.env`、数据文件、索引文件、模型权重、缓存文件或真实 API Key。

## 阶段完成后必须更新

每完成一个阶段或重要子任务，应更新本文件中的当前阶段、最近成功命令、最近成功产物、当前卡点和下一步任务。长日志、完整实验表或大段设计应放入 `reports/`、`artifacts/` 或对应阶段文档。
