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
  pipeline/
    download_amazon_reviews.py
    preprocess_amazon_reviews.py
    build_dev_sample.py
    evaluate_popularity.py
    train_lightgcn.py
    build_product_dense_index.py
    build_product_bm25_index.py
    build_evidence_index.py
  tools/
    generate_data_quality_report.py
    export_items_preview.py
    try_filters.py
    audit_retrieval.py
  experiments/
    download_esci_examples.py
    analyze_esci_overlap.py
tests/
  test_health.py
  test_filters.py
  test_fusion.py
  test_evidence.py
  test_api.py
README.md
requirements.txt
```

脚本从仓库根目录使用模块方式运行。常用命令示例：

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.preprocess_amazon_reviews
.\.venv\Scripts\python.exe -m scripts.pipeline.train_lightgcn --scope full
.\.venv\Scripts\python.exe -m scripts.pipeline.build_product_dense_index --scope full
.\.venv\Scripts\python.exe -m scripts.pipeline.build_product_bm25_index --scope full
.\.venv\Scripts\python.exe -m scripts.tools.audit_retrieval --scope full --channels e5 blair bm25
```

自动生成且可重复构建的索引报告、分析报告和预览统一写入 `artifacts/`，不提交 Git。
用于对比历史实验的 CSV 指标保留在 `reports/metrics/` 并提交 Git。

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
scripts/pipeline/download_amazon_reviews.py
scripts/pipeline/preprocess_amazon_reviews.py
scripts/pipeline/build_dev_sample.py
scripts/tools/generate_data_quality_report.py
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
scripts/pipeline/evaluate_popularity.py
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

### 使用依赖

```text
CUDA 版 PyTorch
torch-geometric
```

使用 PyTorch Geometric（PyG）提供的
`torch_geometric.nn.models.LightGCN`。直接读取阶段 2 生成的 Parquet
交互数据，并复用阶段 3 已有的时间切分和指标口径。

阶段 5 开始开发前，先根据本机显卡和驱动版本，从 PyTorch 官方安装页面选择匹配的
CUDA 版 PyTorch 安装命令。不要继续使用 CPU 版 PyTorch 训练 LightGCN。安装完成后
执行以下命令，确认 PyTorch 能够识别 CUDA 和本机显卡：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

以上安装命令适用于当前开发机。其他机器必须根据显卡和驱动版本调整 CUDA wheel
通道。验收前必须确认 `torch.cuda.is_available()` 返回 `True`，并记录实际使用的
GPU 名称。

### 编写文件

```text
scripts/pipeline/train_lightgcn.py
cartwise/retrieval/lightgcn.py
tests/test_lightgcn.py
```

### 步骤

1. 从 `interactions_train.parquet` 构建用户和商品 ID 映射。
2. 仅使用训练集交互构建用户-商品二部图，不允许验证集和测试集边进入训练图。
3. 使用 PyG `LightGCN`、BPR loss 和负样本训练，在开发小样本上先跑通 GPU 训练。
4. 保存模型权重、模型配置、用户和商品 ID 映射以及训练历史商品集合。
5. 加载已保存模型并执行 Top K 推理，排除用户已经交互过的商品。
6. 每个 epoch 完成后立即在终端打印一次 loss，便于观察训练过程。
7. 分批评估验证集和测试集，将 LightGCN 指标追加写入 CSV，不覆盖历史实验结果。
8. 在 LightGCN 指标 CSV 中增加 `gcn_parameters` 列，并将同口径 Popularity 基线固定
   放在最前面，便于直接对比。

开发样本用于验证训练、保存、加载、推理和评估链路，不用于证明最终模型增益。
阶段 5 即使用与本机显卡匹配的官方 CUDA 版 PyTorch 完成实现和开发样本训练，
避免阶段 12 正式训练前再修改设备选择、模型保存加载或批量评估代码。全量数据训练和
正式指标对比仍留到阶段 12。

训练脚本必须显式支持 `device` 配置，并在请求使用 CUDA 但 CUDA 不可用时立即报错，
不要静默回退到 CPU。模型保存加载、Top K 推理和批量评估必须使用同一套设备选择逻辑。
Tiny Graph 单元测试可以使用 CPU，以便快速验证算法边界；开发样本训练和阶段验收必须
使用 GPU。

需要对同一个已保存模型补充多个 Top K 指标时，使用仅评估模式，不要重新训练：

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.train_lightgcn --scope full --evaluate-only --k 10 50
```

该命令加载已有 `lightgcn.pt`，将多组指标合并后直接追加写入同一个 LightGCN CSV。

当前开发机已验证 `torch==2.12.0+cu126` 和 `torch-geometric==2.7.0` 可以使用
`NVIDIA GeForce GTX 1660 Ti` 完成开发样本 GPU 训练、保存、加载、Top K 推理和
分批评估。开发样本包含 `40,945` 个训练用户、`500` 个商品和 `101,713` 条训练交互。

### 验收

- 给定 `user_id` 可以返回未交互过的 Top K 商品。
- 未知用户返回空列表，冷启动 Popularity fallback 留到阶段 7。
- 模型保存后重新加载，给定相同输入能够执行推理。
- 生成 `reports/metrics/dev/lightgcn.csv`。
- 能够与 `reports/metrics/dev/popularity.csv` 使用相同指标口径比较。
- `reports/metrics/dev/lightgcn.csv` 首行数据为 Popularity 基线，后续 LightGCN
  实验结果按训练顺序追加，并记录 `gcn_parameters`。
- 每个 epoch 完成后终端立即输出本轮 loss。
- 测试覆盖历史商品排除、未知用户、保存加载和 Tiny Graph CPU 冒烟。
- CUDA 可用性检查通过，开发样本训练使用 GPU 完成，不允许静默回退到 CPU。

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
cartwise/core/llm.py
cartwise/retrieval/dense.py
cartwise/retrieval/bm25.py
scripts/pipeline/build_product_dense_index.py
scripts/pipeline/build_product_bm25_index.py
```

### 功能

- 将商品元数据按以下固定顺序拼成基础商品文档：

  ```text
  Title
  -> Brand
  -> Main Category
  -> Categories
  -> Features
  -> Details
  -> Description
  ```

- E5、BLaIR 和 BM25 复用同一份基础商品文档。`main_category` 只用于提供检索文本，
  不作为清洗目录或硬过滤商品的依据。
- Dense 编码时不使用字符数硬截断。分别使用 E5 和 BLaIR 自己的 tokenizer，按照
  各自模型 token 上限自动截断尾部内容。由于高价值字段位于文档前部，优先保留标题、
  品牌、类目、特征和属性，最后截断较长的描述。
- 构建索引时分别记录两个 tokenizer 的 token 长度分布、发生截断的商品数和截断比例。
- 使用 `intfloat/e5-small-v2` 和 `hyp1231/blair-roberta-base`
  分别生成商品向量，完成零微调对比。
- `e5-small-v2` 使用 `query:` 和 `passage:` 前缀生成归一化向量。该模型只处理英文
  文本。
- `blair-roberta-base` 使用官方模型卡约定的归一化 `[CLS]` 向量。该模型面向英文
  Amazon Reviews 2023 商品检索。
- 两个模型使用独立 Qdrant collection，不能混用向量。
- BM25 索引保存在本地。
- 用户输入包含中文字符时，先调用 LLM 翻译为英文，再将英文查询交给 Dense 和 BM25。
- 中文检测只使用正则表达式 `[\u4e00-\u9fff]` 做轻量判断；翻译提示词只要求将购物
  搜索查询翻译为英文并返回译文，
  不要求 JSON，不解析结构化约束，不做复杂查询改写。
- 翻译提示词保持固定：

  ```text
  Translate the following shopping search query into English.
  Return only the translation without explanation:
  {query}
  ```

- 英文查询直接进入 Dense 和 BM25，不调用 LLM。
- LLM 未配置、超时或翻译结果为空时明确报错，不将中文查询静默传给英文检索模型。
- 阶段 6 不接入 ESCI 数据集，不微调 Embedding 模型。

### 验收查询

```text
guitar tuner for beginners
portable microphone stand for home recording

适合初学者的吉他调音器
```

检查并记录：

- E5 和 BLaIR 对英文查询召回的商品是否语义合理。
- 中文查询能够通过最小 LLM 翻译层转换为英文，并进入相同检索链路。
- 两个模型的 Top K 结果可以由人工对比，选择一期默认 Dense 模型。
- 阶段 6 不建立复杂评估集，不在当前阶段进行 Embedding 微调。

使用统一召回审核工具对 E5、BLaIR 和 BM25 分别执行人工测评。每个进程只加载一个
查询通道，默认每次召回 Top 10，并复用已经加载的模型：

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.build_product_bm25_index --scope full
.\.venv\Scripts\python.exe -m scripts.tools.audit_retrieval --scope full --channels e5
.\.venv\Scripts\python.exe -m scripts.tools.audit_retrieval --scope full --channels blair
.\.venv\Scripts\python.exe -m scripts.tools.audit_retrieval --scope full --channels bm25
```

交互模式优先输入 `query-id <ID>`，从
`artifacts/reports/manual_testing/retrieval_audit_queries.json` 读取人工测评查询。仍可使用
`query <文本>` 临时调试，或使用 `user <用户 ID>` 审核推荐通道：

```text
query-id EN-01
query-id ZH-03
query guitar tuner for beginners
```

每轮 query 召回分别生成可评分 HTML 和机器可读 JSON。HTML 支持对每条结果填写
`0`、`1`、`2` 三档相关度和可选备注，在浏览器本地自动保存进度，并导出评分 CSV。
默认保存到：

```text
artifacts/reports/retrieval_audit/<scope>/<timestamp>_<sequence>_<channel>_<query-id>.html
artifacts/reports/retrieval_audit/<scope>/<timestamp>_<sequence>_<channel>_<query-id>.json
```

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

阶段 7 处理带自然语言 query 的导购推荐。Dense 和 BM25 是当前 query 的主要搜索召回
通道；LightGCN 和 Popularity 提供个性化与热门度补充，但不能绕过 query 相关性门控
直接进入最终候选池。

使用 weighted Reciprocal Rank Fusion 合并候选，默认权重调整为：

| 用户类型   | 候选权重                                                      |
| ---------- | ------------------------------------------------------------- |
| 已知用户   | Dense `0.45`、BM25 `0.25`、LightGCN `0.25`、Popularity `0.05` |
| 冷启动用户 | Dense `0.65`、BM25 `0.30`、Popularity `0.05`                  |

类目相关性分为两个层次：

```text
宽泛商品类型：
  microphone_stand、guitar_strings、tuner 等受控标签
  -> 可用于 LightGCN 和 Popularity 补充候选的 query 相关性门控

细粒度形态和使用场景：
  boom_arm、desk_mounted、broadcast、podcast 等
  -> 保留给 Dense、BM25 和后续排序，不直接作为类目硬过滤条件
```

Dense 和 BM25 已经根据原始 query 检索商品，不执行宽泛商品类型门控，避免由于商品标签
覆盖不完整或同义词归一化不足误删有效搜索结果。例如 `microphone arm` 查询召回的
`desk-mounted broadcast microphone boom stand` 应保留。

LightGCN 和 Popularity 与当前 query 无直接关系。只有在能够高置信度解析出宽泛商品
类型时，才允许使用通过该类型门控的 LightGCN 和 Popularity 候选补充搜索结果。如果
无法高置信度解析宽泛商品类型，则本轮 query 只使用 Dense 和 BM25 候选，不添加
LightGCN 或 Popularity 独有候选。

价格、指定品牌、排除品牌、颜色和材料仍属于用户明确约束，在融合后对全部候选统一
执行硬过滤。类目门控只控制 LightGCN 和 Popularity 补充候选是否与 query 的宽泛商品
类型一致，不替代阶段 4 已有硬过滤器，也不对 Dense 和 BM25 结果做字符串匹配过滤。

处理流程：

```text
原始 query -> Dense + BM25 搜索召回
宽泛商品类型 -> LightGCN + Popularity 补充候选门控
-> 去重 -> weighted RRF -> 明确约束硬过滤 -> Top 5
```

### 验收

- 已知用户可以返回符合约束的 Top 5。
- 冷启动用户可以返回符合约束的 Top 5。
- 同一商品不会重复出现。
- 每个结果记录召回来源。
- Dense 和 BM25 召回结果不会因为宽泛商品类型标签缺失而被门控删除。
- LightGCN 和 Popularity 独有候选只有通过宽泛商品类型门控后才能进入带 query 的候选池。
- 无法高置信度解析宽泛商品类型时，不向带 query 的候选池添加 LightGCN 或 Popularity 独有候选。

### 提交

```powershell
git add .
git commit -m "feat: add candidate fusion"
```

## 阶段 8：评论证据

### 编写文件

```text
scripts/pipeline/build_evidence_index.py
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

- 在阶段 6 最小查询翻译函数的基础上扩展可替换的 LLM 适配层。
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
