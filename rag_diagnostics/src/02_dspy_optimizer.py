"""DSPy 自动调优引擎 (DSPy Auto-optimizer)
=========================================
利用 DSPy 3.x 的编译级 Prompt 优化能力，结合 Ragas 0.4.x 作为评分裁判，
自动进化出防幻觉、高相关性的 RAG Prompt 配置。

核心思路：
1. 定义一个 RAGAnswer 模块（带 CoT），模拟你的 RAG 生成环节
2. 用 Ragas 的 Faithfulness + AnswerRelevancy 作为联合评分函数
3. DSPy 的 BootstrapFewShot 在多轮尝试中自动淘汰低分生成，
   保留高质量 few-shot 示例，编译成最优 Prompt 配置

用法：
    uv run python src/02_dspy_optimizer.py
"""

import json
import os
import warnings
from typing import Any, Protocol, cast

import dspy  # pyright: ignore[reportMissingTypeStubs] -- DSPy has no py.typed marker
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot  # pyright: ignore[reportMissingTypeStubs]

# Ragas 0.4.x API (using legacy metric imports compatible with evaluate())
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

import nest_asyncio  # pyright: ignore[reportMissingTypeStubs] -- package has no stubs
from ragas import (
    evaluate,  # pyright: ignore[reportUnknownVariableType] -- Ragas callback internals expose Unknown
)
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness

nest_asyncio.apply()  # pyright: ignore[reportUnknownMemberType] -- untyped third-party boundary
load_dotenv()

# 抑制 Ragas deprecation 警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# ==============================================================
# 配置 DSPy 3.x
# ==============================================================
dspy_api: Any = dspy
turbo: Any = dspy_api.LM("openai/gpt-4o-mini", temperature=0.1)
dspy_api.configure(lm=turbo)

# ==============================================================
# 真实系统 Prompt 模板映射
# ==============================================================
# 你的真实 system prompt 模板大致如下:
#
#   "You are an expert knowledge base assistant.
#    Today is {{current_date}}.
#    {{user_instruction}}
#    Answer based on the following context:
#    {{context}}"
#
# DSPy 映射规则:
#   模板中的「固定指令」 → Signature docstring（DSPy 会优化它）
#   模板中的「变量插槽」 → Signature InputField（作为数据传入）
#   生成的「回答」      → Signature OutputField
#
# 这样 DSPy 优化出来的 prompt 可以直接替换你模板中的固定指令部分，
# 而变量插槽保持不变。
# ==============================================================


class RAGAnswer(dspy_api.Signature):  # pyright: ignore[reportUntypedBaseClass] -- DSPy dynamically builds signatures
    """You are an expert knowledge base assistant. Answer the user's
    question using ONLY the provided retrieved context.
    If the context does not contain the answer, say you don't know.

    Use the current date for any time-sensitive questions.
    Follow the user instruction if provided.
    """

    # ---- 对应模板中的动态变量 ----
    context: Any = dspy_api.InputField(
        desc="Retrieved reference documents from the search engine (maps to {{context}} in prompt template)"
    )
    question: Any = dspy_api.InputField(desc="The user's question")
    current_date: Any = dspy_api.InputField(
        desc="Today's date for time-sensitive questions (maps to {{current_date}} in prompt template)",
    )
    user_instruction: Any = dspy_api.InputField(
        desc="Optional per-request instruction to customize behavior (maps to {{user_instruction}} in prompt template)",
    )
    # ---- 要生成的输出 ----
    answer: Any = dspy_api.OutputField(
        desc="A precise, objective answer strictly grounded in the provided context. "
        "Do not fabricate information not present in the documents."
    )


class RAGPipeline(dspy_api.Module):  # pyright: ignore[reportUntypedBaseClass] -- DSPy modules are dynamically typed
    """RAG generation pipeline with template variable support."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType] -- DSPy base is untyped
        self.generate_answer: Any = dspy_api.ChainOfThought(RAGAnswer)

    def forward(
        self,
        question: str,
        context: str,
        current_date: str = "",
        user_instruction: str = "",
    ) -> Any:
        """Generate answer from context with optional template variables."""
        result = self.generate_answer(
            context=context,
            question=question,
            current_date=current_date,
            user_instruction=user_instruction,
        )
        return dspy_api.Prediction(answer=result.answer)


# ==============================================================
# 联合评分函数 (Ragas 0.4.x 作为裁判)
# ==============================================================
class _DSPyExampleLike(Protocol):
    question: str
    context: str | list[str]


class _DSPyPredictionLike(Protocol):
    answer: str


def comprehensive_ragas_metric(
    example: _DSPyExampleLike,
    pred: _DSPyPredictionLike,
    trace: object | None = None,
) -> bool | float:
    """使用 Ragas 对 DSPy 每次尝试生成的答案进行评分。
    综合考虑：
    - Faithfulness (70%): 是否有幻觉/编造
    - AnswerRelevancy (30%): 是否切题

    当 trace 不为 None 时（BootstrapFewShot 优化阶段），
    返回 bool 表示是否达到及格线（用于筛选 few-shot 示例）。
    当 trace 为 None 时（普通评估），返回 float 分数。
    """
    # 构造 Ragas SingleTurnSample
    ctx = example.context
    sample = SingleTurnSample(
        user_input=example.question,
        response=pred.answer,
        retrieved_contexts=[ctx] if isinstance(ctx, str) else ctx,
    )
    dataset = EvaluationDataset(samples=[sample])

    try:
        result: Any = evaluate(  # pyright: ignore[reportUnknownVariableType] -- Ragas overload exposes Unknown internals
            dataset=dataset,
            metrics=cast(
                Any,
                [
                    Faithfulness(),  # pyright: ignore[reportCallIssue] -- dynamic Ragas class export
                    AnswerRelevancy(),  # pyright: ignore[reportCallIssue] -- dynamic Ragas class export
                ],
            ),
            raise_exceptions=False,
            show_progress=False,
        )

        # Ragas 0.4.x: result 是 EvaluationResult, result["metric_name"] 返回分数列表
        scores = cast(dict[str, Any], result.scores[0] if result.scores else {})
        score_f = float(scores.get("faithfulness") or 0.0)
        score_r = float(scores.get("answer_relevancy") or 0.0)
    except Exception as e:
        print(f"  ⚠️ Ragas evaluation error: {e}")
        score_f, score_r = 0.0, 0.0

    # 综合惩罚公式：70% 事实（不能乱编），30% 相关（是否切题）
    total_score = (score_f * 0.7) + (score_r * 0.3)

    # 日志监控
    print(
        f"  🚦 Score: {total_score:.2f} (faithfulness={score_f:.2f}, relevancy={score_r:.2f})"
    )
    print(f"     Q: {example.question[:40]}...")
    print(f"     A: {pred.answer[:50]}...")

    if trace is not None:
        # BootstrapFewShot 优化阶段：综合分 >= 0.7 才通过
        return total_score >= 0.7

    return float(total_score)


# ==============================================================
# 主流程
# ==============================================================
def optimize_pipeline(input_path: str, output_dir: str) -> None:
    """加载日志数据，运行 DSPy 端到端优化。"""
    from datetime import date

    print("📂 Loading log data for optimization...")
    with open(input_path, encoding="utf-8") as f:
        data = cast(list[dict[str, Any]], json.load(f))

    # 构造 DSPy 训练集
    # 每个 Example 都包含对应模板变量的真实值
    trainset: list[Any] = []
    for item in data:
        ex: Any = dspy_api.Example(
            question=item["query"],
            context="\n".join(item["contexts"]),
            # 从日志中读取模板变量，没有则用默认值
            current_date=item.get("current_date", str(date.today())),
            user_instruction=item.get("user_instruction", ""),
        ).with_inputs("question", "context", "current_date", "user_instruction")
        trainset.append(ex)

    print(f"📊 Loaded {len(trainset)} training samples")

    # 配置 BootstrapFewShot 优化器
    teleprompter: Any = cast(Any, BootstrapFewShot)(
        metric=comprehensive_ragas_metric,
        max_bootstrapped_demos=2,  # 最多保留 2 条自动生成的 few-shot 示例
        max_labeled_demos=0,  # 不使用人工标注示例
    )

    rag_system = RAGPipeline()

    print("\n" + "=" * 60)
    print("🚀 Starting DSPy Auto-optimization")
    print("=" * 60)

    compiled_rag: Any = teleprompter.compile(rag_system, trainset=trainset)

    print("\n" + "=" * 60)
    print("🏁 Optimization Complete!")
    print("=" * 60)

    # 保存优化后的配置（JSON 格式，安全、可读、可版本控制）
    output_path = os.path.join(output_dir, "e2e_optimized_flow.json")
    compiled_rag.save(output_path, save_program=False)
    print(f"\n🎉 Optimized pipeline config saved to: {output_path}")
    print("   This JSON contains the optimized prompts and few-shot examples.")
    print("   You can load it back with:")
    print("     program = RAGPipeline()")
    print(f"     program.load('{output_path}')")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "../data/mock_logs_with_refs.json")
    output_dir = os.path.join(base_dir, "..")

    if not os.path.exists(input_file):
        print("⚠️ Silver standard file not found, falling back to raw mock_logs.json")
        input_file = os.path.join(base_dir, "../data/mock_logs.json")

    optimize_pipeline(input_file, output_dir)
