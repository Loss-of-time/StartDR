# StartDR

## 定位

当前仓库拆成两个子项目：

- `drretrieval`：只负责检索、离线评测、导出 TraceDR 风格候选集 `jsonl`
- `drrerank`：只负责读取 TraceDR 风格候选集 `jsonl` 并训练精排模型

两个子项目的唯一交接物是 TraceDR 风格候选集：

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
│     ├─ import_tracedr.py
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
- 导出 TraceDR 风格候选集 `jsonl`

主要入口：

- `patient-candidates`
- `retriever-eval`

默认产物：

- `resource/patient_candidate/<retriever>_top<k>/{split}.jsonl`
- `output/retriever/*.json`

## drrerank

职责：

- 读取 TraceDR 风格候选集 `jsonl`
- 训练当前精排模型

主要入口：

- `rerank-import-tracedr`
- `rerank-train`

默认产物：

- `output/model/*.pt`
- `output/model/*.json`

## 资源目录

当前本地资源约定：

- `resource/DrugRec.jsonl`：规范化后的患者样本
- `resource/DrugRec0716_from_traceDR/`：TraceDR 导出的 `pkl` 候选集
- `resource/DrugRec0328/`、`resource/DrugRec0330/`：训练验证测试切分
- `resource/patient_candidate/`：TraceDR 风格候选集 `jsonl`
- `resource/cache/`：Neo4j 查询缓存、稠密检索向量缓存、Pyserini 索引

## 环境

首次使用：

```powershell
uv sync
```

静态检查：

```powershell
uv run ruff check .
uv run pyright
```

说明：

- `ruff check` 负责 lint 与部分明显错误，不负责完整类型检查
- `pyright` 负责类型检查，当前配置等价于 VS Code `python.analysis.typeCheckingMode = "standard"` 的命令行版本

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
uv run rerank-train --train-input resource/patient_candidate/pyserini_bm25_top50/train.jsonl --dev-input resource/patient_candidate/pyserini_bm25_top50/dev.jsonl --output-name gnn_pyserini_top50 --epochs 5
```

如需直接把 `TraceDR` 的 `pkl` 候选集规范化为同风格 `jsonl`，可执行：

```powershell
uv run rerank-import-tracedr --split train
uv run rerank-import-tracedr --split dev
uv run rerank-import-tracedr --split test
uv run rerank-train --train-input resource/patient_candidate/tracedr_top50/train.jsonl --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl --output-name gnn_tracedr_top50 --epochs 5 --max-evidences 50 --max-entities 100
```

说明：

- `patient-candidates` 与 `rerank-import-tracedr` 最终都输出同一种 `{"people": ..., "top_k_drugs": ...}` `jsonl`
- `rerank-train` 直接读取该 `jsonl` 构图，不再经过 `gnn_data` 中间层
- `rerank-train` 当前支持按 TraceDR 口径截断图规模：默认 `--max-evidences 50`、`--max-entities 100`
- 训练集会跳过“截断后证据中无答案”或“答案实体被截掉”的样本；验证集不会跳过

## 当前状态

截至 `2026-04-04`，当前仓库边界如下：

- 检索与检索后精排训练已经拆成两个子项目
- 两边只通过 TraceDR 风格候选集 `jsonl` 交接，不再共享统一运行入口
- `drrerank` 不再包含 Neo4j 查询、检索器实现、检索评测逻辑
- `ruff check` 与 `pyright` 分工明确，类型问题应以 `pyright` 为准
- 文档、路径、脚本入口若后续再改，必须同步更新本文件

## TODO 可视化训练结果
## TODO gnn 的 baseline 模型
