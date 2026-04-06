"""
Ragas 评测引擎 (Ragas Evaluator)
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

import pandas as pd
from dotenv import load_dotenv

# Ragas 0.4.x API
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

import nest_asyncio

nest_asyncio.apply()
load_dotenv()

# 抑制 Ragas deprecation 警告（evaluate 在 0.4 中被标记为 deprecated，推荐 experiment）
# 但 evaluate 仍然完全可用，experiment 目前仍在完善中
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


def run_evaluation(input_path: str, output_path: str):
    """加载带银标准的日志数据，运行 Ragas 全指标评测。"""
    print(f"📂 Loading data with silver standards from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构造 Ragas 0.4.x 的 SingleTurnSample 列表
    samples = []
    for item in data:
        sample = SingleTurnSample(
            user_input=item["query"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            # 如果有银标准就传入，用于 ContextRecall / ContextPrecision
            reference=item.get("reference"),
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    print(f"📊 Constructed EvaluationDataset with {len(samples)} samples")

    # 初始化评测指标（Ragas 0.4.x 使用类实例而非单例）
    metrics = [
        Faithfulness(),       # 幻觉检测（无需银标准）
        AnswerRelevancy(),    # 答非所问检测（无需银标准）
        ContextRecall(),      # 搜索召回率（需要银标准 reference）
        ContextPrecision(),   # 排序精确度（需要银标准 reference）
    ]

    print("🔍 Running Ragas Evaluation... (this calls LLMs, may take a minute)")

    result = evaluate(
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


def _print_bad_cases(df: pd.DataFrame, threshold: float = 0.5):
    """高亮展示低分案例，帮助快速定位问题。"""
    score_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    if not score_cols:
        return

    print(f"\n⚠️  Bad Cases (any score < {threshold}):")
    print("-" * 60)

    has_bad = False
    for idx, row in df.iterrows():
        bad_metrics = []
        for col in score_cols:
            val = row.get(col)
            if val is not None and isinstance(val, (int, float)) and val < threshold:
                bad_metrics.append(f"{col}={val:.2f}")
        if bad_metrics:
            has_bad = True
            query = row.get("user_input", "N/A")
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
