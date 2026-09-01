from __future__ import annotations

from uuid import UUID

import psycopg

from alembic import command
from app.langgraph_v2.migrations import build_alembic_config


def test_history_schema_upgrades_from_base_and_downgrades_cleanly(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)

    command.upgrade(config, "head")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        conversation = connection.execute(
            """
            INSERT INTO langgraph_v2.conversations (
                tenant_id, owner_subject_id, runtime_mode
            ) VALUES ('tenant-a', 'subject-a', 'linear')
            RETURNING conversation_id, next_message_sequence
            """
        ).fetchone()
        assert conversation is not None
        assert isinstance(conversation[0], UUID)
        assert conversation[1] == 1
        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'langgraph_v2'
                  AND table_name IN ('conversations', 'messages')
                """
            ).fetchall()
        }
        assert {"runtime_mode", "request_id", "sequence"} <= columns
        assert {"thread_id", "turn_id", "idempotency_key"}.isdisjoint(columns)

    command.downgrade(config, "base")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'langgraph_v2')"
        ).fetchone() == (False,)


def test_pre_release_history_redesign_starts_with_empty_application_history(
    langgraph_v2_test_database_url: str,
) -> None:
    """No deployed pre-0016 data exists, so the first-release cut is explicit."""
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0015_drop_artifacts")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            """
            INSERT INTO langgraph_v2.conversations (
                tenant_id, conversation_id, owner_subject_id, thread_id
            ) VALUES ('tenant-a', 'pre-release', 'subject-a', 'pre-release-thread')
            """
        )

    command.upgrade(config, "head")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM langgraph_v2.conversations"
        ).fetchone() == (0,)

    command.downgrade(config, "base")
