"""Router Agent — ReAct & Agentic Handoff pattern.

Implements a Router Agent using pydantic-ai. Instead of pre-planning
a static execution path, this agent uses tools in a ReAct loop to dynamically
decide its next actions. It can retrieve documents, break down complex queries,
or hand off to specialist agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import AbstractToolset

from app.core.mcp import build_mcp_toolsets
from app.core.model_registry import ModelRegistry
from app.models.domain import Document
from app.providers.base import BaseRankerProvider, BaseRetrieverProvider
from app.services.events import EventEmitter


@dataclass
class SharedState:
    """Dependencies and shared state available to the Router Agent's tools."""

    registry: ModelRegistry
    retriever: BaseRetrieverProvider
    ranker: BaseRankerProvider
    emitter: EventEmitter | None = None
    retrieved_documents: list[Document] = field(default_factory=list)


def create_router_agent(
    registry: ModelRegistry,
    extra_toolsets: list[AbstractToolset] | None = None,
) -> Agent[SharedState, str]:
    """Create the Router Agent with dynamic tools."""
    from app.agents.history_processors import filter_thinking, trim_history

    router = Agent[SharedState, str](
        registry.get_model("pro").model,
        output_type=str,
        model_settings=registry.get_model("pro").settings,
        toolsets=extra_toolsets or [],
        history_processors=[
            trim_history(max_messages=20),
            filter_thinking(),
        ],
        instructions=(
            "You are an intelligent router and task execution agent. Your job is to answer the user's "
            "question by strategically using the available tools in a step-by-step manner.\n\n"
            "Available strategies:\n"
            "- For simple questions or chitchat: Answer directly.\n"
            "- For knowledge queries: Use the `retrieve_knowledge_base` tool to find relevant information before answering.\n"
            "- For complex/multi-part questions: Use `query_breakdown` to split the question, then retrieve info for each part.\n"
            "- For specific domains: Hand off to specialist agents (e.g., `ask_finance_agent` for financial queries).\n\n"
            "Guidelines:\n"
            "- ALWAYS search for relevant documents before answering factual questions.\n"
            "- If a retrieval tool returns empty or insufficient results, DO NOT guess. Use `query_breakdown` to try different search terms.\n"
            "- Cite specific documents or sources in your final answer when applicable.\n"
            "- Be thorough but clear and concise."
        ),
    )

    @router.tool
    async def retrieve_knowledge_base(ctx: RunContext[SharedState], query: str) -> str:
        """Retrieve documents from the knowledge base using the given query."""
        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_start("retrieve_docs", {"query": query})

        # Perform retrieval
        docs = await ctx.deps.retriever.retrieve(query)

        # Optional: Rank documents if a ranker is available
        if ctx.deps.ranker and docs:
            docs = await ctx.deps.ranker.rank(query, docs)

        # Store in shared state for final answer formulation or groundedness check
        ctx.deps.retrieved_documents.extend(docs)

        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_completed(
                "retrieve_docs",
                {"document_count": len(docs), "query": query},
            )

        if not docs:
            return "No documents found for this query. Consider rephrasing or breaking down the query."

        # Return a summarized view of the documents to the LLM
        return "\n\n".join(f"[Doc {d.id}]: {d.content}" for d in docs)

    @router.tool
    async def query_breakdown(ctx: RunContext[SharedState], query: str) -> list[str]:
        """Break down a complex query into simpler sub-queries. Use this when the initial query is too broad."""
        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_start(
                "query_breakdown", {"original_query": query}
            )

        # In a real scenario, this could be another fast LLM call, but here we can just ask the Router
        # or implement a simple heuristic/agent. For simplicity, we ask a fast model to split it.
        fast_agent = ctx.deps.registry.create_agent(
            "fast",
            output_type=list[str],
            system_prompt="Split the user's complex question into 2-3 simpler, distinct search queries. Return ONLY a JSON list of strings.",
        )
        result = await fast_agent.run(query)
        sub_queries = result.data

        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_completed(
                "query_breakdown", {"sub_queries": sub_queries}
            )

        return sub_queries

    @router.tool
    async def ask_finance_agent(ctx: RunContext[SharedState], query: str) -> str:
        """Hand off the query to a specialist finance agent. Use this ONLY for finance, math, or tabular data questions."""
        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_start(
                "ask_finance_agent", {"query": query}
            )

        # Example Agentic Handoff: We create a specialized agent on the fly (or load from registry)
        finance_agent = ctx.deps.registry.create_agent(
            "pro",
            output_type=str,
            system_prompt="You are an expert financial analyst. Analyze the data closely and provide numerical answers.",
        )

        # Pass the current retrieved docs as context to the finance agent if any exist
        context_str = ""
        if ctx.deps.retrieved_documents:
            context_str = "\n".join(
                f"[Doc {d.id}]: {d.content}" for d in ctx.deps.retrieved_documents
            )
            context_str = f"\n\nContext Documents:\n{context_str}"

        # Execute the sub-agent
        result = await finance_agent.run(f"User Query: {query}{context_str}")

        if ctx.deps.emitter:
            await ctx.deps.emitter.emit_step_completed(
                "ask_finance_agent", {"agent_response_length": len(result.data)}
            )

        return result.data

    return router
