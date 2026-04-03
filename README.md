# StartDR

## 定位

StartDR 是当前重新整理的毕业设计实验目录，目标是用尽量短的链路把 `DrugRec` 患者数据、本地 Neo4j 药品知识图谱和检索实验串起来。

当前这套代码不是完整主线工程，重点是：

- 读取 `DrugRec` 患者样本
- 从 Neo4j 导出药品文本语料
- 运行 `bm25`、`pyserini_bm25` 与 `dense` 三类召回实验
- 在冻结候选集上训练当前可用的 `gnn` 药品推荐模型
- 输出离线评测结果，给后续主线整理提供依据

当前仓库采用 `uv + src` 布局：

- 仓库目录名保留为 `StartDR`
- 源码统一位于 `src/drcore/`
- 对外使用 `uv run <脚本名>`，不再依赖目录名作为运行入口

## 当前目录结构

```text
StartDR/
├─ resource/                     # 本地资源目录，默认不入库
│  ├─ DrugRec.jsonl
│  ├─ DrugRec0328/
│  ├─ cache/
│  ├─ DrugRec_处理说明.md
│  └─ split.py
├─ docs/                         # 本地说明目录，默认不入库
├─ output/                       # 本地输出目录，默认不入库
├─ src/
│  └─ drcore/
│     ├─ data/
│     ├─ metrics/
│     ├─ model/
│     ├─ retrieval/
│     ├─ schema/
│     ├─ utils/
│     ├─ model_train.py
│     ├─ patient_candidates.py
│     └─ retriever_eval.py
├─ .vscode/
├─ pyproject.toml
└─ README.md
```

## Git 与本地资源约定

当前 `.gitignore` 的规则是：

- 忽略全部 `__pycache__/`
- 忽略 `.venv/`
- 忽略整个 `resource/`，但保留其中的 `.py` 和 `.md`
- 忽略整个 `output/`，但保留 `output/.gitkeep`
- 忽略整个 `docs/`，但保留 `docs/.gitkeep`
- 忽略 `misc/`

这意味着：

- `resource/` 下的 `DrugRec.jsonl`、切分结果、缓存文件默认视为本地资源
- `output/log/`、`output/retriver/` 默认视为运行产物
- `docs/` 下的实验笔记默认不进入 Git

## 核心模块

### `src/drcore/data/`

外部数据源适配层。

当前包含：

- `jsonl.py`：通用 `jsonl` 读写
- `pkl.py`：通用 `pickle` 读写

### `src/drcore/utils/kg.py`

本地 Neo4j 查询执行层。

- 连接地址固定为 `bolt://localhost:7687`
- 当前代码内认证写死为 `neo4j / password`
- Cypher 查询语句直接定义在当前文件内
- 查询结果会通过 `src/drcore/utils/kg_cache_decorator.py` 缓存到 `resource/cache/`

当前主要函数：

- `list_full_drug_details()`

### `src/drcore/retrieval/bm25.py`

基于 `jieba` 分词和 `rank_bm25` 的最小 BM25 召回实现。

流程是：

1. 从 `utils/kg.py` 读取统一药品详情 `DrugRecMedicine`
2. 拼接 `治疗 / 禁用 / 成分` 文本
3. 对患者 `diagnosis + symptom` 分词
4. 返回按分数排序后的候选药品 `drugid`

### `src/drcore/retrieval/dense.py`

基于 Transformer 编码器的稠密召回实现。

当前实现特点：

- 默认模型为 `DMetaSoul/sbert-chinese-general-v2`
- 自动选择 `cuda` 或 `cpu`
- 首次运行会生成药品向量缓存到 `resource/cache/`
- 查询侧会追加检索 instruction，再与统一药品详情生成的药品向量做相似度排序

### `src/drcore/retrieval/pyserini_bm25.py`

基于 `Pyserini` / Lucene 的 BM25 召回实现。

当前实现特点：

- 首次运行会在 `resource/cache/pyserini_bm25_zh/` 构建本地 Lucene 索引
- 索引文档由统一药品详情中的 `治疗 / 禁用 / 成分` 字段生成
- 查询仍然使用患者 `diagnosis + symptom`
- 检索阶段使用 Lucene BM25，参数显式设为 `k1=1.5`、`b=0.75`

### `src/drcore/retrieval/registry.py`

检索器注册入口。

- `build_retriver(name: str)` 当前实际支持 `bm25`、`pyserini_bm25` 和 `dense`
- `get_retriver_names()` 仅返回当前可实际构造的检索器名称

### `src/drcore/patient_candidates.py`

冻结候选集生成入口。

- 默认输入：`resource/DrugRec0328/{split}.jsonl`
- 默认输出：`resource/patient_candidate/<retriver>_top50/{split}.jsonl`
- 当前流程：读取患者样本，调用检索器，补全药品详情并写出冻结候选集

### `src/drcore/model/gnn/intermediate.py`

GNN 中间文件构建入口。

- 默认输入：`resource/patient_candidate/pyserini_bm25_top50/{split}.jsonl`
- 默认输出：`resource/gnn_data/pyserini_bm25_top50/{split}/`
- 当前流程：读取冻结候选集，按 slot 顺序构建 `case + graph_sample`，并写出 `slot_*.pkl + meta.json`

### `src/drcore/retriever_eval.py`

离线检索评测入口。

- 默认输入：`resource/DrugRec0328/test.jsonl`
- 默认输出：`output/retriver/`
- 当前流程：读取患者样本，调用检索器接口，聚合 `hit / recall / mrr` 并写出 JSON 报告

### `src/drcore/model/`

药品推荐模型目录。

当前包含：

- `drugrec_model.py`：统一训练 / 评测接口
- `gnn/`：当前唯一可用的 `gnn` 模型实现

### `src/drcore/model_train.py`

药品推荐模型训练入口。

- 默认训练输入：`resource/gnn_data/pyserini_bm25_top50/train/`
- 默认验证输入：`resource/gnn_data/pyserini_bm25_top50/dev/`
- 默认输出目录：`output/model/`
- 当前仅支持 `gnn` 模型，训练阶段只读取预先构建好的 slot 化 GNN 中间目录

### `src/drcore/metrics/retrieval.py`

检索阶段纯指标函数目录。

- 当前提供：`hit`、`recall`、`mrr`、聚合函数与失败样本统计
- 不包含 CLI、文件读写或检索器构造逻辑

### `src/drcore/schema/`

集中定义 `TypedDict` 与相关类型别名，当前包含：

- `drugrec.py`：患者样本与统一药品字段
- `drugrec_task.py`：训练任务结果结构与 GNN 局部图结构
- `kg.py`：检索阶段共享的辅助类型
- `metrics.py`：离线评测报告结构
- `retriever.py`：检索器协议与候选结构
- `patient_candidate_set.py`：冻结候选集样本结构

`schema` 的分层生成关系图见 `docs/schema_generation.puml`。
其中 `medicine` / `on_medicine` / `conflict` 与检索阶段药品详情统一复用 `DrugRecMedicine`，业务角色通过字段名、参数名与注释表达。
分拆后的小型 schema 时序图见：

- `docs/schema_sequence_01_patient_ingest.puml`
- `docs/schema_sequence_02_kg_access.puml`
- `docs/schema_sequence_03_retrieval_build.puml`
- `docs/schema_sequence_04_retrieval_execute.puml`
- `docs/schema_sequence_05_patient_candidates.puml`
- `docs/schema_sequence_06_metrics_eval.puml`

### `src/drcore/utils/`

通用支撑模块目录。

当前包含：

- `log.py`：统一初始化控制台日志与文件日志
- `kg.py`：Neo4j 查询执行层
- `paths.py`：项目路径常量

## 数据与本地产物

### `resource/`

本地资源目录，通常包含：

- `DrugRec.jsonl`：规范化后的患者样本
- `DrugRec0328/`：训练、验证、测试切分结果
- `DrugRec0330/`：`0.6 / 0.2 / 0.2` 训练、验证、测试切分结果
- `patient_candidate/`：冻结候选集样本导出目录
- `gnn_data/`：GNN 训练中间文件目录
- `cache/`：Neo4j 查询缓存、稠密检索向量缓存
- `split.py`：数据切分脚本
- `DrugRec_处理说明.md`：数据处理说明

### `docs/`

本地实验说明目录。当前常见内容是 BM25 相关笔记与 schema 生成关系图，但按现有 `.gitignore` 默认不入库。

### `output/`

本地运行输出目录，通常包含：

- `log/`：脚本运行日志
- `retriver/`：离线评测 JSON 结果
- `term_normalization/`：词项整理过程的导出文件

## 环境与运行

首次使用：

```powershell
uv sync
```

运行前至少需要准备：

- Neo4j，本地地址 `bolt://localhost:7687`
- 当前代码默认认证 `neo4j / password`
- `resource/DrugRec.jsonl` 或切分数据
- 当前 `pyproject.toml` 已固定 Windows 环境下的 `torch cu126` 源，`uv sync` 会安装 CUDA 版 `torch`

示例命令：

```powershell
uv run retrieval-bm25
uv run retrieval-dense
uv run python resource/split.py --train 0.6 --test 0.2 --dev 0.2 --output-dir resource/DrugRec0330
uv run patient-candidates --input-dir resource/DrugRec0330 --split test
uv run gnn-data --input-dir resource/patient_candidate/pyserini_bm25_top50 --split train --chunk-size 64
uv run model-train --output-name gnn_pyserini_top50 --epochs 5
uv run retriver-eval --retriver bm25 --top-k 10 --output-name bm25_test_k10
uv run retriver-eval --retriver pyserini_bm25 --top-k 50 --output-name pyserini_bm25_test_k50
uv run retriver-eval --retriver dense --top-k 50 --output-name dense_test_k50
```

`uv run patient-candidates` 当前默认使用 `pyserini_bm25` 生成 `top50` 候选集，并写入 `resource/patient_candidate/<retriver>_top50/{split}.jsonl`。
`uv run gnn-data` 会从冻结候选集构建 `resource/gnn_data/<retriver>_top50/{split}/`，目录内包含 `slot_*.pkl` 和 `meta.json`。
`uv run model-train` 会从该中间目录读取训练样本，并把 checkpoint 与训练报告写入 `output/model/`。

## 当前状态

截至 `2026-04-03`，当前可确认的能力边界是：

- 已有患者读取、Neo4j 查询、BM25 召回、Pyserini BM25 召回、Dense 召回、离线评测这条最短实验链路
- 已补齐冻结候选集上的 `gnn` 药品推荐训练入口，当前 `DrugRecModelName` 只声明并注册 `gnn`
- `DrugRec.jsonl`、KG 全量药品详情与冻结候选集中的 `ingredients` / `interaction` 字段现已统一为单一命名，不再保留别名
- 文档、数据、缓存、输出目录以本地实验资源为主，不按仓库源码管理
- 检索注册表当前仅暴露 `bm25`、`pyserini_bm25` 和 `dense` 三个可实际运行的检索器

如果后续修改了目录、默认路径或脚本入口，必须同步更新本文件。
