from __future__ import annotations

import pytest

from tests.postgres import (
    MissingDisposablePostgres,
    UnsafeDisposablePostgres,
    require_disposable_postgres_url,
)


def test_disposable_postgres_prerequisite_has_an_actionable_failure() -> None:
    with pytest.raises(
        MissingDisposablePostgres,
        match="LANGGRAPH_V2_TEST_DATABASE_URL",
    ):
        require_disposable_postgres_url({})


def test_disposable_postgres_fixture_refuses_a_non_test_database() -> None:
    with pytest.raises(UnsafeDisposablePostgres, match="production"):
        require_disposable_postgres_url(
            {
                "LANGGRAPH_V2_TEST_DATABASE_URL": (
                    "postgresql://postgres:secret@db/production"
                )
            }
        )


def test_disposable_postgres_fixture_accepts_an_explicit_test_database() -> None:
    url = require_disposable_postgres_url(
        {
            "LANGGRAPH_V2_TEST_DATABASE_URL": (
                "postgresql://postgres:secret@db/agent_kms_test_42"
            )
        }
    )

    assert url.endswith("/agent_kms_test_42")
