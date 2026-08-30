import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.langgraph_v2.authorization import (
    TrustedRequestContext,
    get_trusted_request_context,
)


def test_trusted_request_context_dependency_validates_gateway_headers() -> None:
    app = FastAPI()

    @app.get("/context")
    async def read_context(
        context: TrustedRequestContext = Depends(get_trusted_request_context),
    ) -> dict[str, str]:
        return context.model_dump()

    assert read_context is not None

    with TestClient(app) as client:
        valid = client.get(
            "/context",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": "subject-a"},
        )
        missing = client.get("/context", headers={"X-Application-Id": "tenant-a"})
        empty = client.get(
            "/context",
            headers={"X-Application-Id": "tenant-a", "X-Subject-Id": ""},
        )
        reserved = client.get(
            "/context",
            headers={
                "X-Application-Id": "tenant-a",
                "X-Subject-Id": "__unassigned__",
            },
        )

    assert valid.status_code == 200
    assert valid.json() == {"tenant_id": "tenant-a", "subject_id": "subject-a"}
    assert missing.status_code == empty.status_code == 422
    assert reserved.status_code == 422


def test_reserved_migrated_subject_is_not_a_valid_request_context() -> None:
    with pytest.raises(ValueError):
        TrustedRequestContext(tenant_id="tenant-a", subject_id="__unassigned__")
