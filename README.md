# StartDR

## 定位

当前仓库拆成三个子项目：

- `drretrieval`：只负责检索、离线评测、导出 TraceDR 风格候选集 `jsonl`
- `drrerank`：只负责读取 TraceDR 风格候选集 `jsonl` 并训练精排模型
- `drrag`：只负责把候选集与精排结果规整为统一 RAG 协议，并执行硅基流动可解释生成与离线评估

`drretrieval` 与 `drrerank` 的交接物仍是 TraceDR 风格候选集：

- `resource/patient_candidate/<retriever>_top<k>/{split}.jsonl`

`drrag` 在此基础上继续产出统一 RAG 样本、生成结果与评估报告：

- `output/rerank/*.jsonl`
- `output/rag/cases/*.jsonl`
- `output/rag/generation/*.jsonl`
- `output/rag/eval/*.json`

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
│     ├─ foursdrug_export.py
│     ├─ foursdrug_train.py
│     ├─ gat_train.py
│     ├─ import_tracedr.py
│     ├─ kgd_export.py
│     ├─ kgd_train.py
│     ├─ tracedr_train.py
│     └─ core/
│  └─ drrag/
│     ├─ export_cases.py
│     ├─ generate_siliconflow.py
│     ├─ evaluate_generation.py
│     └─ core/
├─ pyproject.toml
└─ README.md
```

约定：

- `src/drretrieval/` 根目录只保留可运行入口
- `src/drrerank/` 根目录只保留 CLI 入口
- `src/drrag/` 根目录只保留 CLI 入口
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
- `rerank-tracedr-export-rank`
- `rerank-tracedr-train`
- `treeify-function-graph`

默认产物：

- `output/model/*.pt`
- `output/model/*.json`
- `output/rerank/*.jsonl`
- `output/function_graph/<时间戳>/*`

## drrag

职责：

- 读取 TraceDR 风格候选集 `jsonl`
- 把 retrieval / rerank 侧对象规整为统一 RAG 协议
- 构造受约束推荐 prompt
- 调用硅基流动 JSON Mode 生成药品可解释结果
- 对生成结果做离线结构与命中评估

主要入口：

- `rag-apply-rerank`
- `rag-export-cases`
- `rag-generate-siliconflow`
- `rag-eval-generation`
- `rag-run-experiment`

默认产物：

- `output/rag/cases/*.jsonl`
- `output/rag/generation/*.jsonl`
- `output/rag/eval/*.json`

## 资源目录

当前本地资源约定：

- `resource/DrugRec.jsonl`：规范化后的患者样本
- `resource/DrugRec0716_from_traceDR/`：TraceDR 导出的 `pkl` 候选集
- `resource/DrugRec0328/`、`resource/DrugRec0330/`：训练验证测试切分
- `resource/model/`：当前实际使用的已训练模型权重
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
uv run rerank-tracedr-export-rank --input resource/patient_candidate/tracedr_top50/test.jsonl --checkpoint resource/model/tracedr_top50.pt
```

如需直接把 `TraceDR` 的 `pkl` 候选集规范化为同风格 `jsonl`，可执行：

```powershell
uv run rerank-import-tracedr --split train
uv run rerank-import-tracedr --split dev
uv run rerank-import-tracedr --split test
```

说明：

- `patient-candidates` 与 `rerank-import-tracedr` 最终都输出同一种 `{"people": ..., "top_k_drugs": ...}` `jsonl`
- `rerank-4sdrug-export` 读取三份 TraceDR 风格候选集 `jsonl`，并写出 `4SDrug` 所需 `voc_final.pkl`、`data_{train,eval,test}.pkl`、`ddi_A_final.pkl`、`sym_train_<batch>.pkl`、`drug_train_<batch>.pkl`、`candidate_train_<batch>.pkl`、`sym_sets.pkl`、`drug_multihots.pkl`
- 当前 `rerank-4sdrug-export` 已改为流式两遍导出；`data_{train,eval,test}.pkl`、`sym_train_<batch>.pkl`、`drug_train_<batch>.pkl`、`candidate_train_<batch>.pkl`、`sym_sets.pkl` 在必要时会按行 pickle 写盘，以降低本地导出峰值内存
- 当前实验约定中，`rerank-4sdrug-export` 会默认写出空的 `ddi_A_final.pkl`；原因是 `Neo4j` 药品节点 `id` 与数据集 `drugid` 尚未对齐
- `rerank-4sdrug-export` 现在会把 `top_k_drugs` 候选药物一并并入 `med_voc`，并把每条样本编码为 `[symptoms, diagnosis, gold_medicines, candidate_medicines]`
- `rerank-4sdrug-train` 读取 `rerank-4sdrug-export` 生成的离线目录，兼容旧版整表 pickle 与新版按行 pickle，训练 `4SDrug main1` 变体，并按 `--selection-metric` 保存最佳权重；训练损失与验证指标都只在 `top_k_drugs` 候选空间内计算，默认仍使用 `JA`
- 若本地已有旧版 `4SDrug` 导出目录，拉取本次更新后需要重新执行一次 `rerank-4sdrug-export`
- `rerank-kgd-export` 读取三份 TraceDR 风格候选集 `jsonl`，并将 KGDNet 所需 `pkl` 写入 `--output-dir`
- 当前实验约定中，`rerank-kgd-export` 会默认写出空的 `ddi_A_final.pkl`；原因是 `Neo4j` 药品节点 `id` 与数据集 `drugid` 尚未对齐
- `rerank-kgd-export` 现在会把 `top_k_drugs` 候选药物一并并入 `med_voc`，并把每条样本编码为 `[symptoms, diagnosis, gold_medicines, candidate_medicines]`
- `rerank-kgd-train` 读取 `rerank-kgd-export` 生成的离线目录，并按 `--selection-metric` 输出最佳权重与指标；训练损失与验证排序都只在 `top_k_drugs` 候选空间内计算
- 若本地已有旧版 `KGD` 导出目录，拉取本次更新后需要重新执行一次 `rerank-kgd-export`
- `rerank-gat-train` 直接读取 TraceDR 风格候选集 `jsonl`，按 `--selection-metric` 输出最佳权重与指标；可额外传入 `--test-input`
- `rerank-tracedr-export-rank` 会读取 TraceDR checkpoint 与候选集 `jsonl`，按病例写出 `ranked_drugs` 排序结果，默认写到 `output/rerank/`
  当前实际使用的模型路径记为 `resource/model/*.pt`
- 当前 `drrerank` 训练入口在交互式终端下显示 `tqdm` 进度条；若运行环境不是 TTY（例如 `uv` 子进程日志面板、部分 IDE 终端采集面板），则不再周期性输出进度文本，仅保留每个 epoch 的摘要日志，避免训练输出过于臃肿
- 当前项目所有 Hugging Face `AutoModel.from_pretrained(...)` 均固定使用 `use_safetensors=False`，关闭 safetensors 自动转换探测
- 若在 WSL 代理环境下使用 Hugging Face 下载模型，项目依赖已包含 `socksio`，执行 `uv sync` 后即可为 `httpx` 提供 SOCKS 代理支持
- `rerank-tracedr-train` 默认读取 `resource/patient_candidate/tracedr_top50/train.jsonl`，并可额外传入 `--test-input`
- `rerank-4sdrug-export` 当前 CLI 入口位于 `src/drrerank/foursdrug_export.py`
- `rerank-tracedr-train` 现支持 4 组关键消融开关：`--num-layers 0`、`--disable-evidence-supervision`、`--evidence-text-mode name_only`、`--exclude-on-medicine`
- `rerank-tracedr-train-adl` 会在训练完成后额外写出 `output/model/<output_name>.adl.json`，其中包含训练配置、GPU 信息、git 状态、耗时、最终指标与产物路径
- `scripts/run_tracedr_train_adl.sh` 适合直接在 ADL 云 GPU 上执行；脚本会自动 `uv sync`、落盘日志并调用 `rerank-tracedr-train-adl`
- `rerank-tracedr-export-rank` 若用于导出消融 checkpoint，对应传入同口径参数：`--num-layers`、`--evidence-text-mode`、`--exclude-on-medicine`
- `rerank-4sdrug-train` 当前 CLI 入口位于 `src/drrerank/foursdrug_train.py`
- `rerank-tracedr-export-rank` 当前 CLI 入口位于 `src/drrerank/tracedr_export_rank.py`
- `rerank-tracedr-train` 当前 CLI 入口位于 `src/drrerank/tracedr_train.py`
- `rerank-kgd-export` 当前 CLI 入口位于 `src/drrerank/kgd_export.py`
- `rerank-kgd-train` 当前 CLI 入口位于 `src/drrerank/kgd_train.py`
- `rerank-gat-train` 当前 CLI 入口位于 `src/drrerank/gat_train.py`
- KGD 运行时构图入口位于 `src/drrerank/core/model/kgd/runtime.py`
- 函数图分析独立项目位于 `src/function_graph/`
- `function-graph` 当前 CLI 入口位于 `src/function_graph/analysis_cli.py`
- `treeify-function-graph` 当前 CLI 入口位于 `src/function_graph/treeify_cli.py`
- `function-graph` 现在会在 `json` 与 `md` 产物中附带函数级重构建议，并区分读边界、写边界、状态变异三类 effect
- `function-graph --source <目录>` 会把目录内全部 `.py` 模块视为一个整体分析，并尝试解析目录内模块之间的 `import` / `from ... import ...` 调用关系
- `function-graph` 与 `treeify-function-graph` 默认会在 `output/function_graph/<时间戳>/` 下写入 `json`、`md`、`dot`、`svg` 产物，避免多次运行结果堆在同一层目录

RAG 侧：

```powershell
uv run rag-export-cases --input resource/patient_candidate/tracedr_top50/test.jsonl --top-k 20
uv run rerank-tracedr-export-rank --input resource/patient_candidate/tracedr_top50/test.jsonl --checkpoint resource/model/tracedr_top50_compare_tracedr.pt
uv run rag-apply-rerank --input resource/patient_candidate/tracedr_top50/test.jsonl --ranked-input output/rerank/test__tracedr_top50_compare_tracedr.jsonl --top-k 20
uv run rag-generate-siliconflow --input output/rag/cases/test__tracedr_rerank__top20.jsonl --input-format rag_case --model Qwen/Qwen3-30B-A3B-Instruct-2507 --top-k 20 --limit 1 --overwrite
uv run rag-eval-generation --input output/rag/generation/test__tracedr_rerank__top20__Qwen__Qwen3_30B_A3B_Instruct_2507__recommend__top20__ev3.jsonl
uv run rag-run-experiment --input resource/patient_candidate/tracedr_top50/test.jsonl --checkpoint resource/model/tracedr_top50_compare_tracedr.pt --top-ks 10,20,50 --variants retrieval_direct,tracedr_rerank --limit 3 --overwrite
```

说明：


ADL 云 GPU 训练可直接执行：

```bash
bash scripts/run_tracedr_train_adl.sh \
  --output-name tracedr_top50_adl \
  --epochs 5 \
  --train-input resource/patient_candidate/tracedr_top50/train.jsonl \
  --dev-input resource/patient_candidate/tracedr_top50/dev.jsonl \
  --test-input resource/patient_candidate/tracedr_top50/test.jsonl
```

若要跑关键消融，只需在同一脚本后追加对应参数，例如：

```bash
bash scripts/run_tracedr_train_adl.sh \
- `rag-export-cases` 会把 TraceDR 风格候选集规整为统一 `RagCase` 协议，默认写到 `output/rag/cases/`
- `rag-apply-rerank` 会把病例级精排结果补到 `RagCase` 的 `rerank_rank` / `rerank_score` 字段；若传入 `--top-k`，会按统一实验排序规则冻结最终候选规模
- `rag-generate-siliconflow` 会调用硅基流动 `chat/completions`，并通过 `response_format={"type":"json_object"}` 强制结构化输出
- `rag-generate-siliconflow` 默认 `limit=1`，避免误触发大批量 LLM 对话；需要显式传更大的 `--limit`
- `rag-generate-siliconflow` 的默认输出文件名现在会编码 `top-k` 与 `max_evidences_per_candidate`，避免不同生成配置复用同一结果文件
- `rag-eval-generation` 只读取已生成结果离线计算结构合法率、字段完整率、证据引用约束与 gold 命中情况，不会重复请求 LLM
- `rag-run-experiment` 会固定输入对比组、`top-k`、模型、提示词和输出命名，自动串起 `rag-export-cases` / `rerank-tracedr-export-rank` / `rag-apply-rerank` / `rag-generate-siliconflow` / `rag-eval-generation`
- `rag-run-experiment` 默认输出：
  `output/rag/cases/<split>__<variant>__top{k}.jsonl`、
  `output/rag/generation/*.jsonl`、
  `output/rag/eval/*__comparison_table.{json,md}`、
  `output/rag/eval/*__case_analysis.md`

## 当前状态

截至 `2026-04-04`，当前仓库边界如下：

- 检索、检索后精排训练与 RAG 生成评估已经拆成三个子项目
- `drretrieval` 与 `drrerank` 仍通过 TraceDR 风格候选集 `jsonl` 交接
- `drrag` 负责在不改动训练主干的前提下，统一 retrieval / rerank / RAG 的样本协议，并接入硅基流动生成
- `drrerank` 不再包含 Neo4j 查询、检索器实现、检索评测逻辑
- `ruff check` 与 `pyright` 分工明确，类型问题应以 `pyright` 为准
- 文档、路径、脚本入口若后续再改，必须同步更新本文件

## TODO 可视化训练结果
