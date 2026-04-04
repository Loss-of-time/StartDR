# StartDR

## 定位

当前仓库拆成两个子项目：

- `drretrieval`：只负责检索、离线评测、冻结候选集导出
- `drrerank`：只负责读取候选集 `jsonl`，构建训练中间文件并训练精排模型

两个子项目的唯一交接物是冻结候选集：

- `resource/patient_candidate/<retriever>_top<k>/{split}.jsonl`

## 目录结构

```text
StartDR/
├─ resource/
├─ docs/
├─ output/
├─ src/
│  ├─ drretrieval/
│  │  ├─ patient_candidates.py
│  │  ├─ retriever_eval.py
│  │  └─ core/
│  └─ drrerank/
│     ├─ build_data.py
│     ├─ train.py
│     └─ core/
├─ pyproject.toml
└─ README.md
```

约定：

- `src/drretrieval/` 根目录只保留可运行入口
- `src/drrerank/` 根目录只保留可运行入口
- 非可运行文件统一下沉到各自的 `core/`
- 不再保留统一日志模块，运行信息统一使用 `print` 和 `tqdm`

## drretrieval

职责：

- 读取 `DrugRec` 患者样本
- 从本地 Neo4j 读取药品详情
- 运行 `bm25`、`pyserini_bm25`、`dense` 三类检索
- 评测检索指标
- 导出冻结候选集 `jsonl`

主要入口：

- `patient-candidates`
- `retriever-eval`

默认产物：

- `resource/patient_candidate/<retriever>_top<k>/{split}.jsonl`
- `output/retriever/*.json`

## drrerank

职责：

- 读取 `drretrieval` 产出的候选集 `jsonl`
- 构建 `GNN` 训练中间文件
- 训练当前精排模型

主要入口：

- `rerank-build-data`
- `rerank-train`

默认产物：

- `resource/gnn_data/<retriever>_top<k>/{split}/`
- `output/model/*.pt`
- `output/model/*.json`

## 资源目录

当前本地资源约定：

- `resource/DrugRec.jsonl`：规范化后的患者样本
- `resource/DrugRec0328/`、`resource/DrugRec0330/`：训练验证测试切分
- `resource/patient_candidate/`：冻结候选集
- `resource/gnn_data/`：精排训练中间文件
- `resource/cache/`：Neo4j 查询缓存、稠密检索向量缓存、Pyserini 索引

## 环境

首次使用：

```powershell
uv sync
```

运行前至少需要准备：

- 本地 Neo4j，地址 `bolt://localhost:7687`
- 当前代码默认认证 `neo4j / password`
- `resource/DrugRec.jsonl` 或对应切分数据
- Windows 环境下 `uv sync` 会按 `pyproject.toml` 安装 `torch cu126`

## 示例命令

检索侧：

```powershell
uv run patient-candidates --input-dir resource/DrugRec0330 --split test --retriever pyserini_bm25 --top-k 50
uv run retriever-eval --retriever bm25 --top-k 10 --output-name bm25_test_k10
uv run retriever-eval --retriever pyserini_bm25 --top-k 50 --output-name pyserini_bm25_test_k50
uv run retriever-eval --retriever dense --top-k 50 --output-name dense_test_k50
```

精排侧：

```powershell
uv run rerank-build-data --input-dir resource/patient_candidate/pyserini_bm25_top50 --split train
uv run rerank-build-data --input-dir resource/patient_candidate/pyserini_bm25_top50 --split dev
uv run rerank-train --train-input resource/gnn_data/pyserini_bm25_top50/train --dev-input resource/gnn_data/pyserini_bm25_top50/dev --output-name gnn_pyserini_top50 --epochs 5
```

## 当前状态

截至 `2026-04-04`，当前仓库边界如下：

- 检索与检索后精排训练已经拆成两个子项目
- 两边只通过候选集 `jsonl` 交接，不再共享统一运行入口
- `drrerank` 不再包含 Neo4j 查询、检索器实现、检索评测逻辑
- 文档、路径、脚本入口若后续再改，必须同步更新本文件
