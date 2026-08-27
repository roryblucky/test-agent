from __future__ import annotations

import os

import psycopg
import pytest

from alembic import command
from app.langgraph_v2.migrations import build_alembic_config

pytestmark = pytest.mark.skipif(
    "LANGGRAPH_V2_TEST_DATABASE_URL" not in os.environ,
    reason=(
        "set LANGGRAPH_V2_TEST_DATABASE_URL to an empty disposable PostgreSQL "
        "database whose name contains a standalone 'test' segment"
    ),
)


def test_application_base_revision_upgrades_and_downgrades(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)

    command.upgrade(config, "head")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        exists_after_upgrade = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)",
            ("langgraph_v2",),
        ).fetchone()
    assert exists_after_upgrade == (True,)

    command.downgrade(config, "base")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        exists_after_downgrade = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)",
            ("langgraph_v2",),
        ).fetchone()
    assert exists_after_downgrade == (False,)
