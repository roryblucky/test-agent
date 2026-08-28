from __future__ import annotations

from pathlib import Path

import pytest

from app.langgraph_v2.pre_moderation import ModerationDecision
from app.langgraph_v2.provider_adapters import (
    V2ModerationAdapter,
    V2RankerAdapter,
    V2RetrieverAdapter,
    adapt_tenant_providers,
)
from app.langgraph_v2.reranking import RerankingResult
from app.langgraph_v2.retrieval import RetrievalResult
from app.models.domain import Document, ModerationResult
from app.providers.base import (
    BaseModerationProvider,
    BaseRankerProvider,
    BaseRetrieverProvider,
)


@pytest.mark.asyncio
async def test_retriever_adapter_preserves_documents_and_adds_raw_provenance() -> None:
    class Provider(BaseRetrieverProvider):
        async def retrieve(self, query: str, top_k: int = 10, filter_expr: str | None = None) -> list[Document]:
            assert (query, top_k, filter_expr) == ("refined", 10, None)
            return [Document(id="d1", content="evidence")]

    result = await V2RetrieverAdapter(Provider()).retrieve("refined")

    assert isinstance(result, RetrievalResult)
    assert result.documents == [Document(id="d1", content="evidence")]
    assert result.raw_payload == {
        "provider": "Provider",
        "query": "refined",
        "document_count": 1,
    }


@pytest.mark.asyncio
async def test_ranker_adapter_passes_refined_query_without_changing_provider_contract() -> None:
    class Provider(BaseRankerProvider):
        async def rank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
            assert query == "refined"
            assert [document.id for document in documents] == ["d1", "d2"]
            assert top_n == 2
            return [documents[1], documents[0]]

    result = await V2RankerAdapter(Provider()).rank(
        "refined", [Document(id="d1", content="1"), Document(id="d2", content="2")]
    )

    assert isinstance(result, RerankingResult)
    assert [document.id for document in result.documents] == ["d2", "d1"]


@pytest.mark.asyncio
async def test_ranker_adapter_requests_all_documents_by_default() -> None:
    class Provider(BaseRankerProvider):
        async def rank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
            del query
            assert top_n == 6
            return documents

    documents = [Document(id=f"d{index}", content=str(index)) for index in range(6)]
    result = await V2RankerAdapter(Provider()).rank("refined", documents)

    assert result.documents == documents


@pytest.mark.asyncio
async def test_moderation_adapter_converts_existing_result() -> None:
    class Provider(BaseModerationProvider):
        async def check(self, text: str) -> ModerationResult:
            assert text == "query"
            return ModerationResult(
                is_flagged=True, categories={"violence": 0.75}, reason="unsafe"
            )

    result = await V2ModerationAdapter(Provider()).check("query")

    assert result == ModerationDecision(
        is_flagged=True, categories={"violence": 0.75}, reason="unsafe"
    )


def test_tenant_provider_bundle_is_adapted_without_legacy_orchestration_imports() -> None:
    class RetrieverProvider(BaseRetrieverProvider):
        async def retrieve(self, query: str, top_k: int = 10, filter_expr: str | None = None) -> list[Document]:
            del query, top_k, filter_expr
            return []

    class RankerProvider(BaseRankerProvider):
        async def rank(self, query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
            del query, top_n
            return documents

    class ModerationProvider(BaseModerationProvider):
        async def check(self, text: str) -> ModerationResult:
            del text
            return ModerationResult(is_flagged=False)

    class Bundle:
        retriever = RetrieverProvider()
        ranker = RankerProvider()
        moderation = ModerationProvider()

    adapted = adapt_tenant_providers(Bundle())

    assert adapted.retriever is not None
    assert adapted.ranker is not None
    assert adapted.moderation is not None
    package_root = Path(__file__).parents[2] / "app" / "langgraph_v2"
    source = "\n".join(path.read_text() for path in package_root.glob("*.py"))
    for forbidden in ("app.services.handlers", "FlowConfig", "FlowContext", "ExecutorContext"):
        assert forbidden not in source
