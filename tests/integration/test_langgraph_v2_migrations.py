from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from alembic import command
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import initial_checkpoint_config, thread_id_for
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
)
from app.langgraph_v2.graph import build_linear_graph
from app.langgraph_v2.migrations import build_alembic_config


def test_turn_migration_backfills_user_deadline_from_authoritative_message(
    langgraph_v2_test_database_url: str,
) -> None:
    """Expand/backfill preserves old Run identity and expires legacy Turns."""
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0008_cancellation_intents")
    tenant_id = "tenant-a"
    conversation_id = "conversation-1"
    run_id = uuid4()
    user_id = uuid4()
    assistant_id = uuid4()
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            "INSERT INTO langgraph_v2.conversations (tenant_id, conversation_id) VALUES (%s, %s)",
            (tenant_id, conversation_id),
        )
        connection.execute(
            """
            INSERT INTO langgraph_v2.runs (tenant_id, run_id, conversation_id, status)
            VALUES (%s, %s, %s, 'completed')
            """,
            (tenant_id, run_id, conversation_id),
        )
        connection.execute(
            """
            INSERT INTO langgraph_v2.messages (
                tenant_id, message_id, conversation_id, run_id, role, content,
                idempotency_key, created_at
            ) VALUES (%s, %s, %s, %s, 'user', 'question', 'legacy:user', %s),
                     (%s, %s, %s, %s, 'assistant', 'answer', 'legacy:assistant', %s)
            """,
            (
                tenant_id,
                user_id,
                conversation_id,
                run_id,
                created_at,
                tenant_id,
                assistant_id,
                conversation_id,
                run_id,
                created_at,
            ),
        )

    command.upgrade(config, "0013_artifact_turn_provenance")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        rows = connection.execute(
            """
            SELECT role, turn_id, resume_deadline, content
            FROM langgraph_v2.messages
            ORDER BY role
            """
        ).fetchall()
        run_column = connection.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'langgraph_v2'
              AND table_name = 'messages' AND column_name = 'run_id'
            """
        ).fetchone()
    assert rows[0] == ("assistant", run_id, None, "answer")
    assert rows[1] == (
        "user",
        run_id,
        created_at,
        "question",
    )
    assert run_column is None
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT turn_id FROM langgraph_v2.runs WHERE run_id = %s", (run_id,)
        ).fetchone() == (run_id,)
        connection.execute(
            """
            UPDATE langgraph_v2.conversations
            SET owner_subject_id = 'subject-a'
            WHERE tenant_id = %s AND conversation_id = %s
            """,
            (tenant_id, conversation_id),
        )

    command.upgrade(config, "head")

    async def read_migrated_messages() -> list[tuple[str, str]]:
        async with AsyncConnectionPool(
            langgraph_v2_test_database_url, min_size=1, max_size=2
        ) as pool:
            records = await ConversationMessageRepository(pool).list_messages(
                context=TrustedRequestContext(
                    tenant_id=tenant_id, subject_id="subject-a"
                ),
                conversation_id=conversation_id,
            )
        return [(record.role, record.content) for record in records]

    assert sorted(asyncio.run(read_migrated_messages())) == [
        ("assistant", "answer"),
        ("user", "question"),
    ]

    command.downgrade(config, "0011_run_turn_association")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            """
            SELECT role, run_id, turn_id
            FROM langgraph_v2.messages
            ORDER BY role
            """
        ).fetchall() == [
            ("assistant", run_id, run_id),
            ("user", run_id, run_id),
        ]

    command.downgrade(config, "0009_conversation_authorization")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            "SELECT run_id, content FROM langgraph_v2.messages ORDER BY role"
        ).fetchall() == [(run_id, "answer"), (run_id, "question")]
    command.downgrade(config, "base")


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
        journal_tables = connection.execute(
            """SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'langgraph_v2'
              AND table_name IN (
                  'runs', 'events', 'phase_results', 'cancellation_intents'
              )
            ORDER BY table_name"""
        ).fetchall()
    assert exists_after_upgrade == (True,)
    assert journal_tables == []

    command.downgrade(config, "base")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        exists_after_downgrade = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)",
            ("langgraph_v2",),
        ).fetchone()
    assert exists_after_downgrade == (False,)


def test_upgrade_from_0013_preserves_conversation_and_checkpoint_but_drops_artifacts(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0013_artifact_turn_provenance")
    tenant_id = "tenant-a"
    conversation_id = "conversation-preserved"
    thread_id = thread_id_for(tenant_id, conversation_id)
    turn_id = uuid4()
    artifact_id = uuid4()
    legacy_run_id = uuid4()
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            """INSERT INTO langgraph_v2.conversations
            (tenant_id, conversation_id, owner_subject_id, thread_id)
            VALUES (%s, %s, 'subject-a', %s)""",
            (tenant_id, conversation_id, thread_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.messages
            (tenant_id, message_id, conversation_id, turn_id, role, content,
             idempotency_key, resume_deadline)
            VALUES (%s, %s, %s, %s, 'user', 'preserved question',
                    'preserved:user', now() + interval '1 hour')""",
            (tenant_id, uuid4(), conversation_id, turn_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.artifacts
            (tenant_id, artifact_id, artifact_type, payload,
             conversation_id, turn_id)
            VALUES (%s, %s, 'document', %s, %s, %s)""",
            (
                tenant_id,
                artifact_id,
                Jsonb({"id": "preserved-evidence"}),
                conversation_id,
                turn_id,
            ),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.runs
            (tenant_id, run_id, conversation_id, status, turn_id)
            VALUES (%s, %s, %s, 'running', %s)""",
            (tenant_id, legacy_run_id, conversation_id, turn_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.events
            (tenant_id, run_id, sequence, event_key, type, canonical_envelope)
            VALUES (%s, %s, 1, 'legacy:event', 'token', '{}')""",
            (tenant_id, legacy_run_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.phase_results
            (tenant_id, run_id, phase_name, execution_epoch,
             normalized_result, canonical_result)
            VALUES (%s, %s, 'query', 1, '{}'::jsonb, '{}')""",
            (tenant_id, legacy_run_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.cancellation_intents
            (tenant_id, run_id) VALUES (%s, %s)""",
            (tenant_id, legacy_run_id),
        )

    async def write_checkpoint() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_test_database_url,
            min_size=1,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as pool:
            saver = AsyncPostgresSaver(cast(Any, pool))
            await saver.setup()
            graph = build_linear_graph(checkpointer=saver)
            await graph.ainvoke(
                {
                    "query": "preserved checkpoint",
                    "conversation_id": conversation_id,
                    "client_request_id": None,
                },
                config=initial_checkpoint_config(
                    thread_id=thread_id,
                    checkpoint_ns="",
                ),
                durability="sync",
            )

    asyncio.run(write_checkpoint())
    command.upgrade(config, "head")

    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            """SELECT owner_subject_id, thread_id
            FROM langgraph_v2.conversations
            WHERE tenant_id = %s AND conversation_id = %s""",
            (tenant_id, conversation_id),
        ).fetchone() == ("subject-a", thread_id)
        assert connection.execute(
            """SELECT role, content, turn_id
            FROM langgraph_v2.messages
            WHERE tenant_id = %s AND conversation_id = %s""",
            (tenant_id, conversation_id),
        ).fetchone() == ("user", "preserved question", turn_id)
        assert (
            connection.execute(
                """SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'langgraph_v2'
              AND table_name IN
                  ('runs', 'events', 'phase_results', 'cancellation_intents',
                   'artifacts')"""
            ).fetchall()
            == []
        )

    async def read_checkpoint() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_test_database_url,
            min_size=1,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as pool:
            saver = AsyncPostgresSaver(cast(Any, pool))
            checkpoint = await saver.aget_tuple(
                initial_checkpoint_config(thread_id=thread_id, checkpoint_ns="")
            )
            assert checkpoint is not None
            assert checkpoint.checkpoint["channel_values"]["query"] == (
                "preserved checkpoint"
            )

    asyncio.run(read_checkpoint())


def test_downgrade_to_0013_recreates_empty_compatible_journal_tables(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "head")
    command.downgrade(config, "0013_artifact_turn_provenance")

    run_id = uuid4()
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        table_counts = connection.execute(
            """SELECT
                (SELECT count(*) FROM langgraph_v2.runs),
                (SELECT count(*) FROM langgraph_v2.events),
                (SELECT count(*) FROM langgraph_v2.phase_results),
                (SELECT count(*) FROM langgraph_v2.cancellation_intents)"""
        ).fetchall()
        assert table_counts == [(0, 0, 0, 0)]
        run_columns = {
            row[0]
            for row in connection.execute(
                """SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'langgraph_v2' AND table_name = 'runs'"""
            ).fetchall()
        }
        assert run_columns == {
            "tenant_id",
            "run_id",
            "conversation_id",
            "status",
            "next_event_sequence",
            "terminal_outcome",
            "created_at",
            "completed_at",
            "owner_instance_id",
            "execution_epoch",
            "heartbeat_at",
            "expires_at",
            "checkpoint_id",
            "checkpoint_ns",
            "turn_id",
        }
        connection.execute(
            """INSERT INTO langgraph_v2.runs
            (tenant_id, run_id, conversation_id, status)
            VALUES ('tenant-a', %s, 'conversation-1', 'running')""",
            (run_id,),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.events
            (tenant_id, run_id, sequence, event_key, type, canonical_envelope)
            VALUES ('tenant-a', %s, 1, 'event-1', 'token', '{}')""",
            (run_id,),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.phase_results
            (tenant_id, run_id, phase_name, execution_epoch,
             normalized_result, canonical_result)
            VALUES ('tenant-a', %s, 'query', 1, '{}'::jsonb, '{}')""",
            (run_id,),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.cancellation_intents
            (tenant_id, run_id) VALUES ('tenant-a', %s)""",
            (run_id,),
        )

    command.downgrade(config, "base")


def test_artifact_migration_backfills_turn_provenance_and_downgrades(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0012_message_turn_identity")
    tenant_id = "tenant-a"
    conversation_id = "conversation-artifact"
    turn_id = uuid4()
    run_id = uuid4()
    payload = {"content": "evidence", "id": "d1"}
    run_payload = {"query": "legacy", "source": "mock"}
    run_created_at = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    artifact_id = uuid5(
        NAMESPACE_URL,
        ":".join(
            (
                "langgraph-v2",
                tenant_id,
                conversation_id,
                "turn",
                str(turn_id),
                "retrieval",
                "document",
                canonical,
            )
        ),
    )
    run_artifact_id = uuid5(
        NAMESPACE_URL,
        ":".join(
            (
                "langgraph-v2",
                tenant_id,
                conversation_id,
                "run",
                str(run_id),
                "retrieval",
                "retrieval_raw",
                json.dumps(
                    run_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        ),
    )
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            """INSERT INTO langgraph_v2.conversations
            (tenant_id, conversation_id, owner_subject_id, thread_id)
            VALUES (%s, %s, 'subject-a', %s)""",
            (tenant_id, conversation_id, str(uuid4())),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.messages
            (tenant_id, message_id, conversation_id, turn_id, role, content,
             idempotency_key, resume_deadline)
            VALUES (%s, %s, %s, %s, 'user', 'question', 'artifact:user',
                    now() + interval '1 hour')""",
            (tenant_id, uuid4(), conversation_id, turn_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.runs
            (tenant_id, run_id, conversation_id, status, turn_id)
            VALUES (%s, %s, %s, 'completed', %s)""",
            (tenant_id, run_id, conversation_id, turn_id),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.artifacts
            (tenant_id, artifact_id, artifact_type, payload)
            VALUES (%s, %s, 'document', %s)""",
            (tenant_id, artifact_id, Jsonb(payload)),
        )
        connection.execute(
            """INSERT INTO langgraph_v2.artifacts
            (tenant_id, artifact_id, artifact_type, payload, created_at)
            VALUES (%s, %s, 'retrieval_raw', %s, %s)""",
            (tenant_id, run_artifact_id, Jsonb(run_payload), run_created_at),
        )

    command.upgrade(config, "0013_artifact_turn_provenance")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert connection.execute(
            """SELECT conversation_id, turn_id
            FROM langgraph_v2.artifacts
            WHERE tenant_id = %s AND artifact_id = %s""",
            (tenant_id, artifact_id),
        ).fetchone() == (conversation_id, turn_id)
        assert connection.execute(
            """SELECT conversation_id, turn_id, created_at
            FROM langgraph_v2.artifacts
            WHERE tenant_id = %s AND artifact_id = %s""",
            (tenant_id, run_artifact_id),
        ).fetchone() == (conversation_id, turn_id, run_created_at)

    command.downgrade(config, "0012_message_turn_identity")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assert (
            connection.execute(
                """SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'langgraph_v2' AND table_name = 'artifacts'
              AND column_name IN ('conversation_id', 'turn_id')"""
            ).fetchall()
            == []
        )
        assert connection.execute(
            "SELECT payload FROM langgraph_v2.artifacts WHERE artifact_id = %s",
            (artifact_id,),
        ).fetchone() == (payload,)
        assert connection.execute(
            """SELECT payload, created_at FROM langgraph_v2.artifacts
            WHERE artifact_id = %s""",
            (run_artifact_id,),
        ).fetchone() == (run_payload, run_created_at)
    command.downgrade(config, "base")


def test_artifact_migration_rejects_unmatched_legacy_provenance(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0012_message_turn_identity")
    artifact_id = uuid4()
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            """INSERT INTO langgraph_v2.artifacts
            (tenant_id, artifact_id, artifact_type, payload)
            VALUES ('tenant-a', %s, 'document', %s)""",
            (artifact_id, Jsonb({"id": "unmatched"})),
        )

    with pytest.raises(
        RuntimeError,
        match=rf"Artifact {artifact_id} has 0 provenance matches",
    ):
        command.upgrade(config, "0013_artifact_turn_provenance")
    command.downgrade(config, "base")


def test_message_turn_migration_restores_transitional_run_identity_on_downgrade(
    langgraph_v2_test_database_url: str,
) -> None:
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0011_run_turn_association")
    tenant_id = "tenant-a"
    conversation_id = "conversation-rollback"
    turn_id = uuid4()
    user_run_id = uuid4()
    assistant_run_id = uuid4()
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            """
            INSERT INTO langgraph_v2.conversations (
                tenant_id, conversation_id, owner_subject_id, thread_id
            ) VALUES (%s, %s, 'subject-a', %s)
            """,
            (tenant_id, conversation_id, str(uuid4())),
        )
        connection.execute(
            """
            INSERT INTO langgraph_v2.runs (
                tenant_id, run_id, conversation_id, status, turn_id, created_at
            ) VALUES
                (%s, %s, %s, 'interrupted', %s, %s),
                (%s, %s, %s, 'completed', %s, %s)
            """,
            (
                tenant_id,
                user_run_id,
                conversation_id,
                turn_id,
                datetime(2026, 1, 1, tzinfo=UTC),
                tenant_id,
                assistant_run_id,
                conversation_id,
                turn_id,
                datetime(2026, 1, 2, tzinfo=UTC),
            ),
        )
        connection.execute(
            """
            INSERT INTO langgraph_v2.messages (
                tenant_id, message_id, conversation_id, run_id, turn_id, role,
                content, idempotency_key, resume_deadline
            ) VALUES
                (%s, %s, %s, %s, %s, 'user', 'question', 'rollback:user', %s),
                (%s, %s, %s, %s, %s, 'assistant', 'answer',
                 'rollback:assistant', NULL)
            """,
            (
                tenant_id,
                user_message_id,
                conversation_id,
                user_run_id,
                turn_id,
                datetime(2026, 2, 1, tzinfo=UTC),
                tenant_id,
                assistant_message_id,
                conversation_id,
                assistant_run_id,
                turn_id,
            ),
        )

    command.upgrade(config, "0013_artifact_turn_provenance")
    command.downgrade(config, "0011_run_turn_association")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        rows = connection.execute(
            """
            SELECT role, run_id
            FROM langgraph_v2.messages
            ORDER BY role
            """
        ).fetchall()
    assert rows == [
        ("assistant", assistant_run_id),
        ("user", user_run_id),
    ]
    command.downgrade(config, "base")


def test_conversation_authorization_migrates_existing_conversations_forward(
    langgraph_v2_test_database_url: str,
) -> None:
    tenant_id = "租户/a,[]"
    conversation_id = '对话|:,[]"'
    expected_thread_id = thread_id_for(tenant_id, conversation_id)
    config = build_alembic_config(langgraph_v2_test_database_url)
    command.upgrade(config, "0008_cancellation_intents")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        connection.execute(
            """
            INSERT INTO langgraph_v2.conversations (tenant_id, conversation_id)
            VALUES (%s, %s)
            """,
            (tenant_id, conversation_id),
        )

    async def write_checkpoint() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_test_database_url,
            min_size=1,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as pool:
            saver = AsyncPostgresSaver(cast(Any, pool))
            await saver.setup()
            graph = build_linear_graph(checkpointer=saver)
            await graph.ainvoke(
                {
                    "query": "migration checkpoint",
                    "conversation_id": conversation_id,
                    "client_request_id": None,
                },
                config=initial_checkpoint_config(
                    thread_id=expected_thread_id,
                    checkpoint_ns="",
                ),
            )

    asyncio.run(write_checkpoint())
    command.upgrade(config, "head")

    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        migrated = connection.execute(
            """
            SELECT owner_subject_id, thread_id
            FROM langgraph_v2.conversations
            WHERE tenant_id = %s AND conversation_id = %s
            """,
            (tenant_id, conversation_id),
        ).fetchone()
        assert migrated is not None
        assert migrated == ("__unassigned__", expected_thread_id)

        with pytest.raises(psycopg.errors.UniqueViolation):
            with connection.transaction():
                connection.execute(
                    """
                    INSERT INTO langgraph_v2.conversations (
                        tenant_id, conversation_id, owner_subject_id, thread_id
                    ) VALUES (%s, 'another-conversation', 'subject-a', %s)
                    """,
                    (tenant_id, expected_thread_id),
                )

    async def read_checkpoint() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_test_database_url,
            min_size=1,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as pool:
            saver = AsyncPostgresSaver(cast(Any, pool))
            await saver.setup()
            checkpoint = await saver.aget_tuple(
                initial_checkpoint_config(
                    thread_id=migrated[1],
                    checkpoint_ns="",
                )
            )
            assert checkpoint is not None

    asyncio.run(read_checkpoint())

    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        assigned = connection.execute(
            """
            UPDATE langgraph_v2.conversations
            SET owner_subject_id = 'subject-a'
            WHERE tenant_id = %s AND conversation_id = %s
              AND owner_subject_id = '__unassigned__'
            RETURNING owner_subject_id
            """,
            (tenant_id, conversation_id),
        ).fetchone()
        repeated = connection.execute(
            """
            UPDATE langgraph_v2.conversations
            SET owner_subject_id = 'subject-a'
            WHERE tenant_id = %s AND conversation_id = %s
              AND owner_subject_id = '__unassigned__'
            RETURNING owner_subject_id
            """,
            (tenant_id, conversation_id),
        ).fetchone()
        wrong = connection.execute(
            """
            UPDATE langgraph_v2.conversations
            SET owner_subject_id = 'subject-b'
            WHERE tenant_id = %s AND conversation_id = %s
              AND owner_subject_id = '__unassigned__'
            RETURNING owner_subject_id
            """,
            (tenant_id, conversation_id),
        ).fetchone()
        assert assigned == ("subject-a",)
        assert repeated is None
        assert wrong is None

    async def verify_assignment() -> None:
        async with AsyncConnectionPool(
            langgraph_v2_test_database_url, min_size=1, max_size=2
        ) as pool:
            repository = ConversationMessageRepository(pool)
            with pytest.raises(ConversationNotFound):
                await repository.get_conversation(
                    context=TrustedRequestContext(
                        tenant_id=tenant_id, subject_id="subject-b"
                    ),
                    conversation_id=conversation_id,
                )
            assigned = await repository.get_conversation(
                context=TrustedRequestContext(
                    tenant_id=tenant_id, subject_id="subject-a"
                ),
                conversation_id=conversation_id,
            )
            assert assigned.owner_subject_id == "subject-a"

    asyncio.run(verify_assignment())

    command.downgrade(config, "0008_cancellation_intents")
    command.upgrade(config, "head")
    with psycopg.connect(langgraph_v2_test_database_url) as connection:
        restored = connection.execute(
            """
            SELECT owner_subject_id, thread_id
            FROM langgraph_v2.conversations
            WHERE tenant_id = %s AND conversation_id = %s
            """,
            (tenant_id, conversation_id),
        ).fetchone()
    assert restored == ("__unassigned__", expected_thread_id)
