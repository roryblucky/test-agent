"""Intent-driven Config Store for Adaptive RAG.

This module provides the configuration lookup system. Once the agent
identifies a specific user intent, we use this store to fetch:
1. Whether the query needs refinement before searching.
2. Which specific Retriever / Data Source to query.
3. Which specific Reranker model/strategy to apply to the results.
4. The dynamic System Prompt to load into the final generation agent.
"""

from dataclasses import dataclass, field


@dataclass
class DataSourceConfig:
    """Defines a specific data source and its associated reranker."""

    source_name: str
    rerank_model: str | None = None
    is_mcp: bool = False


@dataclass
class IntentConfig:
    """Defines the execution strategy for a specific intent."""

    intent_name: str
    needs_refine: bool
    data_sources: list[DataSourceConfig] = field(default_factory=list)
    system_prompt: str


class IntentConfigStore:
    """Mock store for intent-based configurations.

    In a real system, this could be loaded from a database or YAML config.
    """

    _configs = {
        # Knowledge Base Specific Intent
        "knowledge_query": IntentConfig(
            intent_name="knowledge_query",
            needs_refine=True,
            data_sources=[
                DataSourceConfig(
                    source_name="wiki_kb", rerank_model="strict_knowledge_reranker"
                )
            ],
            system_prompt=(
                "You are an expert knowledge base assistant. Answer the user's "
                "question using ONLY the provided retrieved context. "
                "If the context does not contain the answer, say you don't know."
            ),
        ),
        # Financial or tabular specific Intent
        "finance_query": IntentConfig(
            intent_name="finance_query",
            needs_refine=True,
            data_sources=[
                DataSourceConfig(
                    source_name="sql_finance_db", rerank_model="numerical_reranker"
                )
            ],
            system_prompt=(
                "You are a financial analyst. Analyze the provided structured financial "
                "data and answer the user's question with accuracy and professional tone. "
                "Always cite specific numbers where possible."
            ),
        ),
        # Example of an MCP-driven intent
        "mcp_data_query": IntentConfig(
            intent_name="mcp_data_query",
            needs_refine=False,
            data_sources=[DataSourceConfig(source_name="mcp", is_mcp=True)],
            system_prompt=(
                "You are an intelligent data-gathering agent. Use the tools available "
                "to query the necessary systems, retrieve the information, and then "
                "synthesize a final answer for the user."
            ),
        ),
        # General/Chitchat Intent (No search needed)
        "chitchat": IntentConfig(
            intent_name="chitchat",
            needs_refine=False,
            data_sources=[],
            system_prompt=(
                "You are a helpful and friendly AI assistant. Chat casually with the user "
                "and respond to their greetings or small talk. Keep it concise."
            ),
        ),
        # Summarization Intent (Search not strictly needed, rely on history/context)
        "summarization": IntentConfig(
            intent_name="summarization",
            needs_refine=False,
            data_sources=[],
            system_prompt=(
                "You are an expert summarizer. Analyze the conversation history "
                "or provided text and provide a concise, structured summary highlighting "
                "the key points."
            ),
        ),
    }

    @classmethod
    def get_config(cls, intent_name: str) -> IntentConfig:
        """Fetch the configuration for a given intent.

        Falls back to a 'default' knowledge query strategy if intent is unknown.
        """
        return cls._configs.get(
            intent_name,
            # Default fallback strategy
            IntentConfig(
                intent_name=intent_name,
                needs_refine=True,
                data_sources=[
                    DataSourceConfig(
                        source_name="general_search", rerank_model="default_reranker"
                    )
                ],
                system_prompt=(
                    "You are a helpful assistant. Use the retrieved context to answer "
                    "the question as accurately as possible."
                ),
            ),
        )
