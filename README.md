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
│     └─ core/
├─ pyproject.toml
└─ README.md
```

约定：

- `src/drretrieval/` 根目录只保留可运行入口
- `src/drrerank/` 根目录只保留跨模型公共入口
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
- 导出 KGDNet、4SDrug 所需离线 `pkl`
- 训练当前精排模型

主要入口：

- `function-graph`
  读取单个 `.py` 文件时输出单文件函数图；读取目录时把目录内全部 `.py` 视为一个整体输出总函数图
- `rerank-4sdrug-export`
- `rerank-4sdrug-train`
- `rerank-kgd-export`
- `rerank-kgd-train`
- `rerank-gat-train`
- `rerank-import-tracedr`
- `rerank-tracedr-train`
- `treeify-function-graph`

默认产物：

- `output/model/*.pt`
- `output/model/*.json`
- `output/function_graph/<时间戳>/*`

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
- `pyproject.toml` 已关闭 pyright 对第三方库源码的深度分析；本地 `.vscode/settings.json` 额外排除了 `.venv`、`misc`、`resource`、`output` 的编辑器级 Python 分析，`misc/` 下的 `TraceDR` 代码按只读参考处理，不参与日常诊断
- `drrerank.core.model.kgd.runtime.get_ehr_data(device, input_dir)` 现已直接返回 `torch_geometric.data.Data` 图对象，供 KGDNet 运行时兼容层使用

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

当前约定：

- `KGD` 与 `4SDrug` 暂时不跑 `DDI` 相关实验
- 原因是当前 `Neo4j` 中药品节点 `id` 与数据集内 `drugid` 尚未完全对齐，继续启用 `DDI` 会引入错误图边
- 当前 `rerank-kgd-export`、`rerank-4sdrug-export` 会默认写出空的 `ddi_A_final.pkl`，不再依赖本地 Neo4j

```powershell
uv run rerank-kgd-export --train-input resource/patient_candidate/tracedr_top50/train.jsonl --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl --test-input resource/patient_candidate/tracedr_top50/test.jsonl --output-dir output/kgd/tracedr_top50
uv run rerank-4sdrug-export --train-input resource/patient_candidate/tracedr_top50/train.jsonl --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl --test-input resource/patient_candidate/tracedr_top50/test.jsonl --output-dir output/4sdrug/tracedr_top50 --batch-sizes 16
uv run rerank-4sdrug-train --input-dir output/4sdrug/tracedr_top50 --output-name foursdrug_top50 --epochs 5 --batch-size 16
uv run rerank-kgd-train --input-dir output/kgd/tracedr_top50 --output-name kgd_top50 --epochs 5
uv run rerank-gat-train --train-input resource/patient_candidate/tracedr_top50/train.jsonl --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl --output-name gat_top50 --epochs 5
uv run rerank-tracedr-train --train-input resource/patient_candidate/tracedr_top50/train.jsonl --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl --output-name tracedr_top50 --epochs 5
uv run rerank-compare-train --models tracedr,gat,kgd,foursdrug --output-prefix tracedr_top50_compare --train-input resource/patient_candidate/tracedr_top50/train.jsonl --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl --test-input resource/patient_candidate/tracedr_top50/test.jsonl --kgd-input-dir output/kgd/tracedr_top50 --foursdrug-input-dir output/4sdrug/tracedr_top50 --epochs 5 --selection-metric mrr --compare-metric mrr
```

如需直接把 `TraceDR` 的 `pkl` 候选集规范化为同风格 `jsonl`，可执行：

```powershell
uv run rerank-import-tracedr --split train
uv run rerank-import-tracedr --split dev
uv run rerank-import-tracedr --split test
```

说明：

- `patient-candidates` 与 `rerank-import-tracedr` 最终都输出同一种 `{"people": ..., "top_k_drugs": ...}` `jsonl`
- `rerank-4sdrug-export` 读取三份 TraceDR 风格候选集 `jsonl`，并写出 `4SDrug` 所需 `voc_final.pkl`、`data_{train,eval,test}.pkl`、`ddi_A_final.pkl`、`sym_train_<batch>.pkl`、`drug_train_<batch>.pkl`、`sym_sets.pkl`、`drug_multihots.pkl`
- 当前实验约定中，`rerank-4sdrug-export` 会默认写出空的 `ddi_A_final.pkl`；原因是 `Neo4j` 药品节点 `id` 与数据集 `drugid` 尚未对齐
- `rerank-4sdrug-train` 读取 `rerank-4sdrug-export` 生成的离线目录，训练 `4SDrug main1` 变体，并按 `--selection-metric` 保存最佳权重；默认仍使用 `JA`
- `rerank-kgd-export` 读取三份 TraceDR 风格候选集 `jsonl`，并将 KGDNet 所需 `pkl` 写入 `--output-dir`
- 当前实验约定中，`rerank-kgd-export` 会默认写出空的 `ddi_A_final.pkl`；原因是 `Neo4j` 药品节点 `id` 与数据集 `drugid` 尚未对齐
- `rerank-kgd-train` 读取 `rerank-kgd-export` 生成的离线目录，并按 `--selection-metric` 输出最佳权重与指标
- `rerank-gat-train` 直接读取 TraceDR 风格候选集 `jsonl`，按 `--selection-metric` 输出最佳权重与指标；可额外传入 `--test-input`
- `rerank-compare-train` 会复用统一训练 runner，顺序执行多个模型并在 `output/model/` 下生成汇总对比报告
- 当前 `drrerank` 训练入口在交互式终端下显示 `tqdm` 进度条；若运行环境不是 TTY（例如 `uv` 子进程日志面板、部分 IDE 终端采集面板），会自动退化为周期性 `print` 进度，避免训练过程没有可见反馈
- 当前项目所有 Hugging Face `AutoModel.from_pretrained(...)` 均固定使用 `use_safetensors=False`，关闭 safetensors 自动转换探测
- 若在 WSL 代理环境下使用 Hugging Face 下载模型，项目依赖已包含 `socksio`，执行 `uv sync` 后即可为 `httpx` 提供 SOCKS 代理支持
- `rerank-tracedr-train` 默认读取 `resource/patient_candidate/tracedr_top50/train.jsonl`，并可额外传入 `--test-input`
- `rerank-4sdrug-export` 当前入口实现位于 `src/drrerank/core/model/foursdrug/export.py`
- `rerank-4sdrug-train` 当前入口实现位于 `src/drrerank/core/model/foursdrug/train.py`
- `rerank-compare-train` 当前入口实现位于 `src/drrerank/core/model/compare.py`
- `rerank-tracedr-train` 当前入口实现位于 `src/drrerank/core/model/tracedr/train.py`
- `rerank-kgd-export` 当前入口实现位于 `src/drrerank/core/model/kgd/export.py`
- `rerank-kgd-train` 当前入口实现位于 `src/drrerank/core/model/kgd/train.py`
- `rerank-gat-train` 当前入口实现位于 `src/drrerank/core/model/gat/train.py`
- KGD 运行时构图入口位于 `src/drrerank/core/model/kgd/runtime.py`
- 函数图分析独立项目位于 `src/function_graph/`
- `function-graph` 现在会在 `json` 与 `md` 产物中附带函数级重构建议，并区分读边界、写边界、状态变异三类 effect
- `function-graph --source <目录>` 会把目录内全部 `.py` 模块视为一个整体分析，并尝试解析目录内模块之间的 `import` / `from ... import ...` 调用关系
- `function-graph` 与 `treeify-function-graph` 默认会在 `output/function_graph/<时间戳>/` 下写入 `json`、`md`、`dot`、`svg` 产物，避免多次运行结果堆在同一层目录

## 当前状态

截至 `2026-04-04`，当前仓库边界如下：

- 检索与检索后精排训练已经拆成两个子项目
- 两边仍通过 TraceDR 风格候选集 `jsonl` 交接，但精排训练侧已经补上统一实验 runner 与对比入口
- `drrerank` 不再包含 Neo4j 查询、检索器实现、检索评测逻辑
- `ruff check` 与 `pyright` 分工明确，类型问题应以 `pyright` 为准
- 文档、路径、脚本入口若后续再改，必须同步更新本文件

## TODO 可视化训练结果
