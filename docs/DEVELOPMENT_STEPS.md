# CartWise 开发步骤

本文档用于指导 CartWise 一期开发。开发原则是先跑通最小闭环，再逐步替换假数据和占位实现。不要同时开发前端、模型和 RAG，以免出现问题时难以定位。

## 一期目标

完成一个可在本机复现的可解释个性化电商导购系统：

- 使用 Amazon Reviews 2023 的 `Musical_Instruments` 子集。
- 使用 Popularity 和 LightGCN 召回个性化候选。
- 使用 Dense 和 BM25 补充符合当前自然语言需求的候选。
- 使用代码层硬过滤严格执行价格、品牌和属性约束。
- 为最终商品检索真实评论证据。
- 使用 LLM 解析意图并基于已验证事实生成中文解释。
- 使用 FastAPI 提供接口，使用 Streamlit 提供演示页面。
- 支持 `换一批`、`更便宜` 和排除品牌等简单多轮操作。

一期暂时不实现 CrossEncoder、Redis、复杂 Agent 和商品共购图扩展。

## 推荐目录结构

```text
cartwise/
  api/
    main.py
    schemas.py
  core/
    config.py
    session.py
    orchestrator.py
    llm.py
  retrieval/
    popularity.py
    lightgcn.py
    dense.py
    bm25.py
    filters.py
    fusion.py
    evidence.py
  ui/
    app.py
scripts/
  download_data.py
  preprocess.py
  build_dev_sample.py
  report_data_quality.py
  train_lightgcn.py
  build_product_index.py
  build_evidence_index.py
  evaluate.py
tests/
  test_health.py
  test_filters.py
  test_fusion.py
  test_evidence.py
  test_api.py
README.md
requirements.txt
```

## 阶段 1：最小 API

### 编写文件

```text
cartwise/core/config.py
cartwise/api/main.py
tests/test_health.py
```

### 功能

只实现：

```text
GET /health
```

接口返回 API、Qdrant、推荐模型和 LLM 状态。Qdrant 未启动时也必须返回 HTTP `200`，状态写为 `unavailable`，不能让健康检查本身崩溃。

示例：

```json
{
  "api": "ok",
  "qdrant": "unavailable",
  "recommender": "not_loaded",
  "llm": "not_configured"
}
```

### 验收

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn cartwise.api.main:app --reload
```

另开终端：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 提交

```powershell
git add .
git commit -m "feat: add health endpoint"
```

## 阶段 2：数据处理

### 编写文件

```text
scripts/download_data.py
scripts/preprocess.py
scripts/build_dev_sample.py
scripts/report_data_quality.py
```

### 处理规则

- 数据集使用 `Musical_Instruments`。
- 商品主键统一为 `parent_asin`。
- 先处理几百个商品的小样本，再处理完整数据。
- 按时间拆分训练集、验证集和测试集。
- 禁止未来数据进入训练图。
- 关联商品元数据和评论。
- 统计价格、品牌、描述和评论文本缺失率。
- 每个商品最多保留 10 条评论证据候选。
- 使用 Parquet 保存处理中间产物，避免反复解析原始 JSON。

### 输出产物

```text
data/processed/items.parquet
data/processed/interactions_train.parquet
data/processed/interactions_valid.parquet
data/processed/interactions_test.parquet
data/processed/reviews.parquet
```

### 验收

输出并检查：

- 商品数。
- 用户数。
- 交互数。
- 训练、验证和测试集规模。
- 关键字段缺失比例。
- 去重和过滤比例。

### 提交

```powershell
git add .
git commit -m "feat: add data preprocessing pipeline"
```

## 阶段 3：Popularity 基线

### 编写文件

```text
cartwise/retrieval/popularity.py
scripts/evaluate.py
```

### 功能

- 统计训练集中每个商品的交互次数。
- 排除用户已经交互过的商品。
- 返回 Top K 热门商品。
- 计算 `Recall@10`、`NDCG@10` 和 `HitRate@10`。

先用小样本跑通，再处理完整数据。

### 验收

- 给定用户 ID 能返回未交互过的 Top K 商品。
- 生成 Popularity 基线指标 CSV。

### 提交

```powershell
git add .
git commit -m "feat: add popularity baseline"
```

## 阶段 4：硬过滤器

### 编写文件

```text
cartwise/retrieval/filters.py
tests/test_filters.py
```

### 功能

- 支持类目、价格、品牌、颜色和材料五类硬约束。
- 类目使用小型受控词表，从商品标题、`categories` 和
  `details_json["Instrument"]` 中派生 `category_tags`。阶段 4 先覆盖演示用例，
  阶段 6 构建商品索引时再改进词表、覆盖率和归一化策略。
- 价格支持上下限，边界包含在结果内。
- 品牌支持指定品牌和排除品牌。
- 颜色合并 `details_json["Color Name"]` 和 `details_json["Color"]`，
  派生 `color_tags`。
- 材料合并 `details_json["Material Type"]` 和 `details_json["Material"]`，
  派生 `material_tags`。
- 字符串比较使用 `strip + casefold` 归一化。
- 仅当用户明确指定某项约束时才执行对应过滤。存在价格、类目、颜色或材料约束时，
  缺失对应字段的商品必须排除；没有对应约束时允许保留缺失字段的商品。
- 预留 `excluded_parent_asins` 接口，但阶段 4 不使用。当前会话已展示商品排除和
  `换一批` 功能留到后续阶段。
- 候选不足 5 个时返回实际数量，不自动放宽约束。

硬过滤器位于候选召回、融合和排序之后，截取 Top 5 之前。它只执行当前请求中的
明确约束，不负责读取用户 ID、处理用户历史交互序列、召回候选、计算推荐分数或
改变候选顺序。硬过滤器必须独立于 LLM 和推荐模型。

### 验收

测试以下边界：

- 类目受控词表可以识别演示用例中的商品。
- 商品缺失类目标签时，有类目约束则排除，无类目约束则允许保留。
- 价格恰好等于下限。
- 价格恰好等于上限。
- 商品价格缺失时，有价格约束则排除，无价格约束则允许保留。
- 品牌、颜色和材料的大小写与首尾空格差异。
- 商品缺失颜色或材料时，有对应约束则排除，无对应约束则允许保留。
- `excluded_parent_asins` 接口存在但默认不影响结果。
- 过滤后保留候选原始顺序。
- 过滤后候选不足 5 个。

### 提交

```powershell
git add .
git commit -m "feat: add hard filters"
```

## 阶段 5：LightGCN

### 编写文件

```text
scripts/train_lightgcn.py
cartwise/retrieval/lightgcn.py
```

### 步骤

1. 使用 RecBole 在小样本上训练。
2. 验证可以保存模型。
3. 验证可以加载模型并执行推理。
4. 评估 LightGCN 指标。
5. 与 Popularity 指标对比。

当前 CPU 版 PyTorch 足够开发。处理完整数据和正式训练前，再安装官方 CUDA 版 PyTorch。

### 验收

- 给定 `user_id` 可以返回未交互过的 Top K 商品。
- 生成 LightGCN 指标 CSV。
- 能够与 Popularity 基线比较。

### 提交

```powershell
git add .
git commit -m "feat: add lightgcn recommender"
```

## 阶段 6：商品混合检索

### 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install sentence-transformers
```

### 编写文件

```text
cartwise/retrieval/dense.py
cartwise/retrieval/bm25.py
scripts/build_product_index.py
```

### 功能

- 将标题、品牌、描述和属性拼成商品文档。
- 使用 `intfloat/multilingual-e5-small` 生成向量。
- Dense 索引写入 Qdrant。
- BM25 索引保存在本地。
- 支持中文查询召回英文商品。

### 验收查询

```text
适合初学者的吉他调音器
家庭录音使用的便携麦克风支架
```

检查检索结果是否包含语义合理的英文商品。

### 提交

```powershell
git add .
git commit -m "feat: add hybrid product retrieval"
```

## 阶段 7：候选融合

### 编写文件

```text
cartwise/retrieval/fusion.py
tests/test_fusion.py
```

### 功能

使用 weighted Reciprocal Rank Fusion 合并候选：

| 用户类型   | 候选权重                                                      |
| ---------- | ------------------------------------------------------------- |
| 已知用户   | LightGCN `0.45`、Dense `0.30`、BM25 `0.15`、Popularity `0.10` |
| 冷启动用户 | Dense `0.50`、BM25 `0.30`、Popularity `0.20`                  |

处理流程：

```text
多路召回 -> 去重 -> 融合打分 -> 硬过滤 -> Top 5
```

### 验收

- 已知用户可以返回符合约束的 Top 5。
- 冷启动用户可以返回符合约束的 Top 5。
- 同一商品不会重复出现。
- 每个结果记录召回来源。

### 提交

```powershell
git add .
git commit -m "feat: add candidate fusion"
```

## 阶段 8：评论证据

### 编写文件

```text
scripts/build_evidence_index.py
cartwise/retrieval/evidence.py
tests/test_evidence.py
```

### 规则

- 每个商品最多索引 10 条评论。
- 优先保留已验证购买、helpful votes 较高和文本非空的评论。
- 同时保留低评分评论，用于展示潜在缺点。
- 每个入选商品最多返回 3 条评论。
- 使用稳定哈希生成 `review_id`。

### 验收

- 每条引用必须属于对应商品。
- 引用 ID 可以稳定复现。
- 评论证据来自实际检索结果。

### 提交

```powershell
git add .
git commit -m "feat: add review evidence retrieval"
```

## 阶段 9：LLM 和多轮会话

### 编写文件

```text
cartwise/core/llm.py
cartwise/core/session.py
cartwise/core/orchestrator.py
```

### LLM 职责

- 将自然语言解析为价格、品牌和属性约束。
- 基于已有商品和评论生成中文解释。

### LLM 禁止事项

- 不允许增加候选列表外的商品。
- 不允许编造价格和属性。
- 不允许编造评论 ID。

### 异常处理

以下情况使用固定模板回退：

- LLM 超时。
- 输出不是合法 JSON。
- 输出引用不存在。
- 输出商品不在候选列表中。

### 会话功能

- `换一批`：排除本会话已经展示的商品。
- `更便宜`：将价格上限调整为上一批最低价格以下。
- 排除品牌：将品牌加入排除集合。

### 提交

```powershell
git add .
git commit -m "feat: add llm adapter and session state"
```

## 阶段 10：完整 API

### 编写文件

```text
cartwise/api/schemas.py
cartwise/api/main.py
tests/test_api.py
```

### 接口

```text
GET  /health
POST /api/v1/sessions
POST /api/v1/sessions/{session_id}/recommend
POST /api/v1/sessions/{session_id}/feedback
```

先使用假数据验证接口，再接入真实推荐链路。

### 验收

- 可以创建会话。
- 可以提交中文推荐请求。
- 可以执行 `换一批`、`更便宜` 和排除品牌。
- 非法 `session_id` 返回明确错误。
- LLM 不可用时仍然返回合法结果。

### 提交

```powershell
git add .
git commit -m "feat: add recommendation api"
```

## 阶段 11：Streamlit 前端

### 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
```

### 编写文件

```text
cartwise/ui/app.py
```

### 原则

Streamlit 只通过 HTTP 调用 FastAPI。推荐逻辑、过滤器、Qdrant 和 LLM 调用全部放在后端。

### 页面功能

- 用户 ID 输入和冷启动模式。
- 中文对话输入框。
- 当前约束展示。
- Top 5 商品卡片。
- 推荐理由和潜在缺点。
- 评论证据和召回来源。
- `换一批`、`更便宜` 和排除品牌按钮。
- 请求耗时展示。

### 启动

先启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn cartwise.api.main:app --reload
```

再启动前端：

```powershell
.\.venv\Scripts\python.exe -m streamlit run cartwise/ui/app.py
```

### 提交

```powershell
git add .
git commit -m "feat: add streamlit demo"
```

## 阶段 12：正式评估和收尾

### 正式实验

在完整数据上运行：

- Popularity。
- LightGCN。
- LightGCN + Dense/BM25 + 硬过滤。

### 记录指标

- `Recall@10`
- `NDCG@10`
- `HitRate@10`
- 本地链路 P50 和 P95 延迟。
- 包含 LLM 的端到端 P50 和 P95 延迟。
- 50 个推荐结果的评论引用准确性人工抽检。

### 补充文档

完善 `README.md`：

- 项目介绍。
- 环境安装步骤。
- 数据准备步骤。
- 训练命令。
- 索引构建命令。
- 后端和前端启动命令。
- 示例请求。
- 已知限制。
- 演示流程。

### 最终走查

使用小样本从零执行：

```text
安装依赖
-> 数据预处理
-> 训练模型
-> 构建索引
-> 启动 API
-> 启动 Streamlit
-> 请求 Top 5
-> 执行换一批
-> 执行更便宜
-> 排除品牌
```

### 提交

```powershell
git add .
git commit -m "docs: complete reproducible demo guide"
```
