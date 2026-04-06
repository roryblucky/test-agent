"""
DSPy 自动调优引擎 (DSPy Auto-optimizer)
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

import dspy
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

# Ragas 0.4.x API
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy

import nest_asyncio

nest_asyncio.apply()
load_dotenv()

# 抑制 Ragas deprecation 警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

# ==============================================================
# 配置 DSPy 3.x
# ==============================================================
turbo = dspy.LM("openai/gpt-4o-mini", temperature=0.1)
dspy.configure(lm=turbo)

# ==============================================================
# 定义 RAG 生成签名 (Signature)
# ==============================================================
# 注意：我们去掉了 QueryRewrite 模块。
# 原因：在离线诊断场景下，重写后的搜索词不会被用来重新检索（因为
# context 来自历史日志），QueryRewrite 的输出会被丢弃。
# 保留它只会增加 DSPy 搜索空间的噪声，降低优化效率。
# 当接入真实检索引擎后，可以重新加入 QueryRewrite 模块。
# ==============================================================


class RAGAnswer(dspy.Signature):
    """你是一个专业的银行坐席辅助系统。
    请根据搜索引擎取回的文档，精准、客观、不落窠臼地回答。
    严格要求：不能编造任何文档中没有的信息。
    如果文档中没有相关信息，请明确告知用户。"""

    context = dspy.InputField(desc="从检索引擎取回的相关文档片段")
    question = dspy.InputField(desc="客户的问题")
    answer = dspy.OutputField(desc="严格基于文档的准确客观回答，不编造任何外部信息")


class RAGPipeline(dspy.Module):
    """RAG 生成管线：接收 context + question，输出防幻觉的答案。"""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(RAGAnswer)

    def forward(self, question, context):
        result = self.generate_answer(context=context, question=question)
        return dspy.Prediction(answer=result.answer)


# ==============================================================
# 联合评分函数 (Ragas 0.4.x 作为裁判)
# ==============================================================
def comprehensive_ragas_metric(example, pred, trace=None):
    """
    使用 Ragas 对 DSPy 每次尝试生成的答案进行评分。
    综合考虑：
    - Faithfulness (70%): 是否有幻觉/编造
    - AnswerRelevancy (30%): 是否切题

    当 trace 不为 None 时（BootstrapFewShot 优化阶段），
    返回 bool 表示是否达到及格线（用于筛选 few-shot 示例）。
    当 trace 为 None 时（普通评估），返回 float 分数。
    """
    # 构造 Ragas SingleTurnSample
    sample = SingleTurnSample(
        user_input=example.question,
        response=pred.answer,
        retrieved_contexts=[example.context] if isinstance(example.context, str) else example.context,
    )
    dataset = EvaluationDataset(samples=[sample])

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy()],
            raise_exceptions=False,
            show_progress=False,
        )

        # Ragas 0.4.x: result 是 EvaluationResult, result["metric_name"] 返回分数列表
        scores = result.scores[0] if result.scores else {}
        score_f = scores.get("faithfulness", 0.0)
        score_r = scores.get("answer_relevancy", 0.0)
    except Exception as e:
        print(f"  ⚠️ Ragas evaluation error: {e}")
        score_f, score_r = 0.0, 0.0

    score_f = score_f if score_f is not None else 0.0
    score_r = score_r if score_r is not None else 0.0

    # 综合惩罚公式：70% 事实（不能乱编），30% 相关（是否切题）
    total_score = (score_f * 0.7) + (score_r * 0.3)

    # 日志监控
    print(f"  🚦 Score: {total_score:.2f} (faithfulness={score_f:.2f}, relevancy={score_r:.2f})")
    print(f"     Q: {example.question[:40]}...")
    print(f"     A: {pred.answer[:50]}...")

    if trace is not None:
        # BootstrapFewShot 优化阶段：综合分 >= 0.7 才通过
        return total_score >= 0.7

    return float(total_score)


# ==============================================================
# 主流程
# ==============================================================
def optimize_pipeline(input_path: str, output_dir: str):
    """加载日志数据，运行 DSPy 端到端优化。"""
    print("📂 Loading log data for optimization...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构造 DSPy 训练集
    trainset = []
    for item in data:
        ex = dspy.Example(
            question=item["query"],
            context="\n".join(item["contexts"]),
        ).with_inputs("question", "context")
        trainset.append(ex)

    print(f"📊 Loaded {len(trainset)} training samples")

    # 配置 BootstrapFewShot 优化器
    teleprompter = BootstrapFewShot(
        metric=comprehensive_ragas_metric,
        max_bootstrapped_demos=2,  # 最多保留 2 条自动生成的 few-shot 示例
        max_labeled_demos=0,       # 不使用人工标注示例
    )

    rag_system = RAGPipeline()

    print("\n" + "=" * 60)
    print("🚀 Starting DSPy Auto-optimization")
    print("=" * 60)

    compiled_rag = teleprompter.compile(rag_system, trainset=trainset)

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
