# RAG 诊断与自动优化系统 MVP (Observer Pattern)

本项目是一个"非侵入式的、能够自我进化的 RAG 诊断系统"的最小可行性版本 (MVP)。
它采用了 **Observer（观察者）模式** 设计，完全独立于您的主线 RAG Engine (如 `agent-kms`) 运行。

本项目涵盖了：
1. 没有标准答案时的评估方式 (利用 AI 生成"银标准")。
2. 通过 Ragas 0.4.x 获取包括 `Faithfulness (防幻觉)` 和 `Context Recall (防搜不出来)` 在内的各项核心指标打分。
3. 通过 DSPy 3.x 自动针对打分不高的 Case，倒推并编译出更好、更强限制的高级 Prompt！

## 技术栈版本

| 依赖 | 版本 | 说明 |
|------|------|------|
| Ragas | ≥0.4.0 | RAG 评测框架，使用 `SingleTurnSample` + `EvaluationDataset` API |
| DSPy | ≥2.6.0 | 编译级 Prompt 优化框架，使用 `dspy.configure()` API |
| LangChain OpenAI | ≥0.3.0 | LLM 调用层 |
| Python | ≥3.12.9 | 运行环境 |

## 依赖与环境准备

1. **配置环境变量**：将 `.env.example` 复制一份并重命名为 `.env`。
   ```bash
   cp .env.example .env
   ```
   并在 `.env` 里填入您的大模型 API 密钥。

2. **了解目录结构**：
   - `data/mock_logs.json`: 模拟了现有系统每日吐出来的日志。里头有明显的"幻觉"案例。
   - `src/00_silver_standard_generator.py`: 解决没有标准答案的问题。跑一遍，AI 会严格依照正确文档自动帮您生成 Ground Truth。
   - `src/01_ragas_evaluator.py`: 核心评测场！使用 Ragas 0.4.x API 跑出各项指标打分。
   - `src/02_dspy_optimizer.py`: 自动进化模块。把跑分差的丢给 DSPy，看看它如何编译出带 Few-shot 的优化 Prompt！

## 如何运行

请按顺序在项目根目录 (`rag_diagnostics`) 执行下面三步：

**Step 1: 离线生成"银标准"**
```bash
uv run python src/00_silver_standard_generator.py
```
*(执行完毕后会在 data 目录下生成 `mock_logs_with_refs.json`)*

**Step 2: 运行 Ragas 指标评测**
```bash
uv run python src/01_ragas_evaluator.py
```
*(会输出整体分数 + 逐条 bad cases 分析，详细结果保存在 `data/evaluation_detailed_results.csv`)*

**Step 3: 运行自动调优 (Auto-Prompt Evolution)**
```bash
uv run python src/02_dspy_optimizer.py
```
*(DSPy 将进行提示词自优化，生成的配置保存在 `e2e_optimized_flow.json`)*

---

## 下一步 (对接到实际生产系统)

1. **数据捕获**：在原系统 `FlowEngine` 的 `execute()` 结尾处，将 `{query, contexts, answer}` 落盘为 JSONL 文件或推入消息队列。
2. **替换真实的 LLM**：代码中留有兼容 Vertex AI 等架构的注释，修改对应实例化语句即可。
3. **Hot-Reload Prompt**：当 `02_dspy_optimizer` 输出了更优配置后，主程序通过定时拉取刷新 System Prompt。
