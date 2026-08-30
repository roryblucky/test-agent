"""Ragas 评测引擎 (Ragas Evaluator)
=================================
使用 Ragas 0.4.x API 对 RAG 系统的日志进行全面评测。

指标说明：
- Faithfulness: 幻觉检测 —— 回答中是否有上下文中不存在的编造信息
- AnswerRelevancy: 答非所问检测 —— 回答是否切中要点，没有废话
- ContextRecall: 搜索召回率 —— 检索引擎返回的上下文是否覆盖了所需信息（需要银标准）
- ContextPrecision: 排序精确度 —— 关键信息是否被排在了前面（需要银标准）

用法：
    uv run python src/01_ragas_evaluator.py
"""

import json
import os
import warnings
from typing import Any, cast

from dotenv import load_dotenv

# Ragas 0.4.x API
# NOTE: We use the legacy metric imports (from ragas.metrics) because they are
# compatible with the evaluate() function. The newer collections API requires
# llm_factory() and is designed for the experiment() workflow.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

import nest_asyncio  # pyright: ignore[reportMissingTypeStubs] -- package has no stubs
from ragas import (
    evaluate,  # pyright: ignore[reportUnknownVariableType] -- Ragas callback internals expose Unknown
)
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

nest_asyncio.apply()  # pyright: ignore[reportUnknownMemberType] -- untyped third-party boundary
load_dotenv()

# 抑制 Ragas deprecation 警告（evaluate 在 0.4 中被标记为 deprecated，推荐 experiment）
# 但 evaluate 仍然完全可用，experiment 目前仍在完善中
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


def run_evaluation(input_path: str, output_path: str) -> None:
    """加载带银标准的日志数据，运行 Ragas 全指标评测。"""
    print(f"📂 Loading data with silver standards from {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = cast(list[dict[str, Any]], json.load(f))

    # 构造 Ragas 0.4.x 的 SingleTurnSample 列表
    samples: list[SingleTurnSample] = []
    for item in data:
        sample = SingleTurnSample(
            user_input=item["query"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            # 如果有银标准就传入，用于 ContextRecall / ContextPrecision
            reference=item.get("reference"),
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=cast(Any, samples))
    print(f"📊 Constructed EvaluationDataset with {len(samples)} samples")

    # 初始化评测指标（Ragas 0.4.x 使用类实例而非单例）
    metrics: list[Any] = [  # pyright: ignore[reportUnknownVariableType] -- dynamic Ragas metric exports
        Faithfulness(),  # pyright: ignore[reportCallIssue] -- Ragas dynamic export is a class at runtime
        AnswerRelevancy(),  # pyright: ignore[reportCallIssue] -- Ragas dynamic export is a class at runtime
        ContextRecall(),  # pyright: ignore[reportCallIssue] -- Ragas dynamic export is a class at runtime
        ContextPrecision(),  # pyright: ignore[reportCallIssue] -- Ragas dynamic export is a class at runtime
    ]

    print("🔍 Running Ragas Evaluation... (this calls LLMs, may take a minute)")

    result: Any = evaluate(  # pyright: ignore[reportUnknownVariableType] -- Ragas overload exposes Unknown internals
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    print("\n" + "=" * 60)
    print("📋 [Overall Evaluation Result]")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # 导出逐条数据的打分明细到 CSV
    df = result.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"\n📁 Detailed per-sample scores saved to {output_path}")

    # 打印 bad cases 摘要
    _print_bad_cases(df)


def _print_bad_cases(df: Any, threshold: float = 0.5) -> None:
    """高亮展示低分案例，帮助快速定位问题。"""
    score_cols: list[str] = [
        str(column)
        for column in df.columns
        if column not in ("user_input", "response", "retrieved_contexts", "reference")
    ]
    if not score_cols:
        return

    print(f"\n⚠️  Bad Cases (any score < {threshold}):")
    print("-" * 60)

    has_bad = False
    for idx, row in df.iterrows():
        bad_metrics: list[str] = []
        for col in score_cols:
            val = row.get(col)
            if val is not None and isinstance(val, (int, float)) and val < threshold:
                bad_metrics.append(f"{col}={val:.2f}")
        if bad_metrics:
            has_bad = True
            query = str(row.get("user_input", "N/A"))
            print(f"  [{idx}] Q: {query[:50]}...")
            print(f"       Failures: {', '.join(bad_metrics)}")

    if not has_bad:
        print("  ✅ No bad cases found — all scores above threshold!")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "../data/mock_logs_with_refs.json")
    output_file = os.path.join(base_dir, "../data/evaluation_detailed_results.csv")

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        print("   Please run 00_silver_standard_generator.py first.")
    else:
        run_evaluation(input_file, output_file)
