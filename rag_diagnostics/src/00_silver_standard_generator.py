"""
银标准生成器 (Silver Standard Generator)
========================================
解决"没有标准答案"的评测困境。
使用大模型作为裁判，严格依据上下文生成"银标准"参考答案，
为后续 Ragas 评测（如 Context Recall）提供 reference 基准。

用法：
    uv run python src/00_silver_standard_generator.py
"""

import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================
# 初始化 LLM (使用 OpenAI 兼容协议)
# 如果你想使用 Google Vertex，可以把这里的类替换为：
#   from langchain_google_vertexai import ChatVertexAI
#   llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0.0)
# ============================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 定义生成理想答案的 Prompt（使用 ChatPromptTemplate 替代已弃用的 PromptTemplate）
SYSTEM_INSTRUCTION = """你是一个客观严格的金融知识打标员。
你的任务是根据提供的[参考文档(Context)]，以及[用户问题(Query)]，写出一个绝对正确、没有废话、纯客观的[标准答案]。
如果参考文档中无法回答用户问题，请明确输出：\u201c无法根据提供文档回答\u201d。
确保你的答案绝对没有任何主观臆测和外部知识的幻觉。"""

system_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human", "[用户问题(Query)]: {query}\n[参考文档(Context)]: {context_str}\n\n[标准答案]: "),
])

chain = system_prompt | llm


def generate_silver_standards(input_path: str, output_path: str):
    """读取日志数据，为每条记录生成银标准答案。"""
    print(f"📂 Loading data from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data, 1):
        print(f"\n[{i}/{len(data)}] Processing query: {item['query']}")
        context_str = "\n".join(item["contexts"])

        # 让裁判长生成完美的银标准
        response = chain.invoke({"query": item["query"], "context_str": context_str})

        # 将银标准存入 reference 字段
        item["reference"] = response.content.strip()
        print(f"  ✅ Silver Standard: {item['reference'][:80]}...")

    # 导出包含银标准的数据
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 Saved data with silver standards to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "../data/mock_logs.json")
    output_file = os.path.join(base_dir, "../data/mock_logs_with_refs.json")

    generate_silver_standards(input_file, output_file)
