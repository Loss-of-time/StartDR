# drrag

## 定位

`drrag` 只负责三件事：

- 读取 TraceDR 风格候选集 `jsonl`
- 规整为统一 `RagCase` 协议
- 构造 RAG prompt，并调用硅基流动生成药品可解释结果
- 对生成结果做离线结构与命中评估

当前入口：

- `uv run rag-apply-rerank`
- `uv run rag-export-cases`
- `uv run rag-generate-siliconflow`
- `uv run rag-eval-generation`
- `uv run rag-run-experiment`
- `uv run recommend-api`

## SiliconFlow 环境变量

当前项目统一使用：

- `SILICONFLOW_API_KEY`
- 接口地址：`https://api.siliconflow.cn/v1/chat/completions`

获取方式：

- Shell 读取：`echo $SILICONFLOW_API_KEY`
- Python 读取：`os.environ["SILICONFLOW_API_KEY"]`

生效方式：

- 已写入 `~/.zshenv`
- 所有新的 `zsh` 会话都会自动加载该变量

## 硅基流动代金券推荐模型

筛选原则：

- 优先选择适合中文医疗推荐场景的通用指令模型
- 优先选择适合 RAG 回答的非极端思维链模型
- 仅保留对 `drrag` 直接有用的文本、多模态、Embedding、Reranker 模型
- 不纳入语音、图片生成、视频生成、LoRA 微调模型

### 主模型

推荐作为 `drrag` 默认主模型的候选：

- `Qwen/Qwen3-30B-A3B-Instruct-2507`
  结论：默认首选。
  原因：指令跟随与中文能力通常更稳，规模适中，适合作为药物推荐与解释生成的主力模型。

- `deepseek-ai/DeepSeek-V3.2`
  结论：高质量主模型备选。
  原因：更适合追求回答质量与复杂约束遵循的场景，但通常不应默认用于最低成本实验。

- `Qwen/Qwen3-235B-A22B-Instruct-2507`
  结论：高规格上限模型。
  原因：适合最终效果对比或高质量样本抽检，不适合作为大规模离线试验默认模型。

### 思考模型

推荐只用于少量高难样本复核，不建议作为默认批量 RAG 模型：

- `deepseek-ai/DeepSeek-R1`
- `Qwen/Qwen3-235B-A22B-Thinking-2507`
- `Qwen/Qwen3-30B-A3B-Thinking-2507`

判断：

- 这类模型更适合难例分析、错误归因、少量病例复盘
- 不适合当前 `drrag` 的常规小样本生成口径
- 若后续要做思考模型实验，应单独记录提示词、输出长度与额外开销

### 多模态模型

当病例输入未来包含图片、检查单截图或 OCR 文本融合时，可优先考虑：

- `Qwen/Qwen3-VL-32B-Instruct`
  结论：多模态主推。
  原因：能力与成本通常更均衡，适合图文联合理解。

- `Qwen/Qwen3-VL-8B-Instruct`
  结论：轻量多模态备选。
  原因：适合快速验证流程，不适合作为质量上限。

- `Qwen/Qwen2.5-VL-32B-Instruct`
  结论：稳定备选。
  原因：如果后续实验更偏向成熟 VL 路线，可作为对照模型。

### Embedding

如果后续把 `drrag` 扩展到在线检索或向量召回，优先考虑：

- `Qwen/Qwen3-Embedding-4B`
  结论：默认首选。
  原因：能力与资源消耗更平衡，适合先做检索实验。

- `Qwen/Qwen3-Embedding-8B`
  结论：高质量备选。
  原因：适合在召回质量成为主要瓶颈时再启用。

- `BAAI/bge-m3`
  结论：通用中文检索稳妥备选。
  原因：适合做跨模型对照，不依赖 Qwen 技术栈。

### Reranker

如果后续把 `drrag` 扩展到在线重排，优先考虑：

- `Qwen/Qwen3-Reranker-8B`
  结论：默认首选。
  原因：更适合高质量候选重排。

- `Qwen/Qwen3-Reranker-4B`
  结论：成本更低的次选。
  原因：适合先验证在线重排收益。

- `BAAI/bge-reranker-v2-m3`
  结论：稳妥对照组。
  原因：适合做中文检索重排基线。

## 当前实现边界

需要明确：

- 当前 `rag-apply-rerank` 会把病例级精排结果补到统一 `RagCase`，同时补齐候选药与证据的 `rerank_rank/score`，供后续 prompt 优先按精排结果选择
- 当前 `rag-apply-rerank` 支持 `--top-k`，用于按统一实验口径冻结最终进入 RAG 的候选规模
- 当前代码已经接入硅基流动 `chat/completions`
- 当前实现默认使用 OpenAI 兼容请求体，并通过 `response_format={"type":"json_object"}` 强制 JSON 输出
- 当前 `rag-generate-siliconflow` 默认 `limit=1`，目的不是节省功能，而是避免误触发大规模 LLM 对话
- 当前 `rag-generate-siliconflow` 支持 `--workers`，用于并发发起多个病例请求；默认仍为 `1`
- 当前 `rag-generate-siliconflow` 会显示 `tqdm` 总进度条，并把断点续跑时跳过的已完成样本计入初始进度
- 当前 `rag-generate-siliconflow` 会按固定批次阶段写盘；若单条请求发生远端断连，会落为失败记录而不是中断整批任务
- 当前 `rag-eval-generation` 只评估已落盘结果，不会重复发起在线请求
- 当前 `rag-run-experiment` 固定两种输入方案：`retrieval_direct`、`tracedr_rerank`
- 当前 `rag-run-experiment` 会固定 `top-k`、模型、提示词与输出命名，并额外产出结果表与案例分析 Markdown
- 当前 `recommend-api` 只负责在线推荐推理 API，不负责前端页面
- 当前 `recommend-api` 会在服务启动时预加载药品缓存、Pyserini 检索器与 TraceDR checkpoint
- 当前 `recommend-api` 的默认运行口径固定为 `top50` 检索、`top10` 展示、每药 `1` 条基础证据
- 当前 `recommend-api` 返回体中的 `evidence_map` 只保留最终被 `recommendation.items[*].evidence_ids` 实际引用的证据
- 当前 `recommend-api` 返回体中的 `evidence_map` 会优先展示 TraceDR 精排后的证据文本、证据排序与证据分数；若未命中精排证据，则保留 retrieval 默认值
- 当前 `recommend-api` 会在 `evidence_map[evidence_id].graph_relations` 中返回该被引用证据对应的底层图谱关系全集，供前端展开展示；这不是检索阶段全部候选关系，也不是子关系级命中判定结果
