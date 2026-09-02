from __future__ import annotations

import psycopg

from alembic import command
from app.langgraph_v2.migrations import build_alembic_config


def test_application_schema_has_no_conversation_or_message_history(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)

    command.upgrade(config, "head")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT to_regclass('langgraph_v2.conversations')"
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT to_regclass('langgraph_v2.messages')"
        ).fetchone() == (None,)

    command.downgrade(config, "base")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'langgraph_v2')"
        ).fetchone() == (False,)


def test_incremental_upgrade_drops_the_pre_release_registry(
    langgraph_v2_test_database_url: str,
) -> None:
    """No deployed pre-0016 data exists, so the first-release cut is explicit."""
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0017_drop_history")
    command.upgrade(config, "head")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT to_regclass('langgraph_v2.conversations')"
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT to_regclass('langgraph_v2.messages')"
        ).fetchone() == (None,)

    command.downgrade(config, "base")
