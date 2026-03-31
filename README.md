# MINE

## 定位

`MINE/` 是当前重新整理的毕业设计实验目录，目标是用尽量短的链路把 `DrugRec` 患者数据、本地 Neo4j 药品知识图谱和检索实验串起来。

当前这套代码不是完整主线工程，重点是：

- 读取 `DrugRec` 患者样本
- 从 Neo4j 导出药品文本语料
- 运行 `bm25`、`pyserini_bm25` 与 `dense` 三类召回实验
- 输出离线评测结果，给后续主线整理提供依据

请从工作区根目录 `pyneo/` 启动，使用项目限定导入，例如 `python -m MINE.kg`。

## 当前目录结构

```text
MINE/
├─ constant/
│  ├─ __init__.py
│  └─ CYPHER.py
├─ data/                         # 本地数据目录，默认不入库
│  ├─ DrugRec.jsonl
│  ├─ DrugRec0328/
│  ├─ cache/
│  ├─ DrugRec_处理说明.md
│  └─ split.py
├─ docs/                         # 本地说明目录，默认不入库
├─ metrics/
│  └─ retriver.py
├─ model/                        # 预留目录，当前为空
├─ output/                       # 本地输出目录，默认不入库
├─ retrieval/
│  ├─ __init__.py
│  ├─ bm25.py
│  ├─ dense.py
│  └─ registry.py
├─ schema/
│  ├─ __init__.py
│  ├─ drugrec.py
│  ├─ kg.py
│  ├─ metrics.py
│  ├─ patient_candidate_set.py
│  └─ retriever.py
├─ utils/
│  ├─ kg_cache_decorator.py
│  └─ log.py
├─ input_process.py
├─ kg.py
└─ README.md
```

## Git 与本地资源约定

当前 `.gitignore` 的规则是：

- 忽略全部 `__pycache__/`
- 忽略整个 `data/`，但保留其中的 `.py` 和 `.md`
- 忽略整个 `output/`，但保留 `output/.gitkeep`
- 忽略整个 `docs/`，但保留 `docs/.gitkeep`

这意味着：

- `DrugRec.jsonl`、切分结果、缓存文件默认视为本地资源
- `output/log/`、`output/retriver/` 默认视为运行产物
- `docs/` 下的实验笔记默认不进入 Git

## 核心模块

### `input_process.py`

患者数据读取入口，默认读取 `MINE/data/DrugRec.jsonl`。

当前提供：

- `get_patients(number: int)`：读取前若干条
- `get_all_patients()`：读取全部样本
- `load_jsonl(path: Path)`：读取任意 `jsonl`
- `load_jsonl_limit(path: Path, limit: int | None)`：读取前若干条

类型定义见 `MINE/schema/drugrec.py`。

### `kg.py`

本地 Neo4j 查询执行层。

- 连接地址固定为 `bolt://localhost:7687`
- 当前代码内认证写死为 `neo4j / password`
- Cypher 常量定义在 `MINE/constant/CYPHER.py`
- 查询结果会通过 `MINE/utils/kg_cache_decorator.py` 缓存到 `MINE/data/cache/`

当前主要函数：

- `list_drug_ids()`
- `list_simple_drug_details()`
- `list_candidate_text_index_records()`
- `list_full_drug_details()`
- `get_drug_details_by_ids()`

### `retrieval/bm25.py`

基于 `jieba` 分词和 `rank_bm25` 的最小 BM25 召回实现。

流程是：

1. 从 `kg.py` 读取药品轻量语料
2. 拼接 `治疗 / 禁用 / 成分` 文本
3. 对患者 `diagnosis + symptom` 分词
4. 返回按分数排序后的候选药品 `drugid`

### `retrieval/dense.py`

基于 Transformer 编码器的稠密召回实现。

当前实现特点：

- 默认模型为 `DMetaSoul/sbert-chinese-general-v2`
- 自动选择 `cuda` 或 `cpu`
- 首次运行会生成药品向量缓存到 `MINE/data/cache/`
- 查询侧会追加检索 instruction，再与药品向量做相似度排序

### `retrieval/pyserini_bm25.py`

基于 `Pyserini` / Lucene 的 BM25 召回实现。

当前实现特点：

- 首次运行会在 `MINE/data/cache/pyserini_bm25_zh/` 构建本地 Lucene 索引
- 索引文档仍然使用 `治疗 / 禁用 / 成分` 文本
- 查询仍然使用患者 `diagnosis + symptom`
- 检索阶段使用 Lucene BM25，参数显式设为 `k1=1.5`、`b=0.75`

### `retrieval/registry.py`

检索器注册入口。

- `build_retriver(name: str)` 当前实际支持 `bm25`、`pyserini_bm25` 和 `dense`
- `get_retriver_names()` 仍返回 `dual_tower` 占位名，但当前目录下没有对应实现文件，不能直接构造

### `metrics/retriver.py`

离线检索评测入口。

- 默认输入：`MINE/data/DrugRec0328/test.jsonl`
- 默认输出：`MINE/output/retriver/`
- 当前指标：`hit`、`recall`、`mrr`
- 当前报告内容：聚合指标、样本数、失败数

### `schema/`

集中定义 `TypedDict` 与相关类型别名，当前包含：

- `drugrec.py`：患者样本与药品字段
- `kg.py`：知识图谱查询结果结构
- `metrics.py`：离线评测报告结构
- `retriever.py`：检索器协议与候选结构
- `patient_candidate_set.py`：冻结候选集样本结构

### `utils/log.py`

统一初始化控制台日志与文件日志。

- 日志目录固定为 `MINE/output/log/`
- 同一次运行内共用同一份日志文件

## 数据与本地产物

### `data/`

本地数据目录，通常包含：

- `DrugRec.jsonl`：规范化后的患者样本
- `DrugRec0328/`：训练、验证、测试切分结果
- `DrugRec0330/`：`0.6 / 0.2 / 0.2` 训练、验证、测试切分结果
- `patient_candidate/`：冻结候选集样本导出目录
- `cache/`：Neo4j 查询缓存、稠密检索向量缓存
- `split.py`：数据切分脚本
- `DrugRec_处理说明.md`：数据处理说明

### `docs/`

本地实验说明目录。当前常见内容是 BM25 相关笔记，但按现有 `.gitignore` 默认不入库。

### `output/`

本地运行输出目录，通常包含：

- `log/`：脚本运行日志
- `retriver/`：离线评测 JSON 结果
- `term_normalization/`：词项整理过程的导出文件

## 运行方式

运行前至少需要安装与当前脚本直接相关的依赖：

- 公共依赖：`neo4j`、`rich`
- BM25 相关：`jieba`、`rank-bm25`、`pyserini`
- Dense 相关：`torch`、`transformers`、`jaxtyping`

示例命令：

```powershell
python -m MINE.input_process
python -m MINE.kg
python -m MINE.retrieval.bm25
python -m MINE.retrieval.dense
python MINE/data/split.py --train 0.6 --test 0.2 --dev 0.2 --output-dir MINE/data/DrugRec0330
python -m MINE.patient_candidates --input-dir MINE/data/DrugRec0330 --split test
python -m MINE.metrics.retriver --retriver bm25 --top-k 10 --output-name bm25_test_k10
python -m MINE.metrics.retriver --retriver pyserini_bm25 --top-k 50 --output-name pyserini_bm25_test_k50
python -m MINE.metrics.retriver --retriver dense --top-k 50 --output-name dense_test_k50
```

`python -m MINE.patient_candidates` 当前默认使用 `pyserini_bm25` 生成 `top50` 候选集，并写入 `MINE/data/patient_candidate/<retriver>_top50/{split}.jsonl`。

## 当前状态

截至 `2026-03-30`，`MINE/` 当前可确认的能力边界是：

- 已有患者读取、Neo4j 查询、BM25 召回、Pyserini BM25 召回、Dense 召回、离线评测这条最短实验链路
- `DrugRec.jsonl`、KG 全量药品详情与冻结候选集中的 `ingredients` / `interaction` 字段现已统一为单一命名，不再保留别名
- 文档、数据、缓存、输出目录以本地实验资源为主，不按仓库源码管理
- 检索注册表里仍留有 `dual_tower` 占位名，但当前仓库中没有对应实现，不能视为可用能力

如果后续修改了目录、默认路径或脚本入口，必须同步更新本文件。
