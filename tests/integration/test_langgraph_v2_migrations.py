from __future__ import annotations

import asyncio

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from alembic import command
from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.checkpointing import initial_checkpoint_config, thread_id_for
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
)
from app.langgraph_v2.graph import build_tracer_graph
from app.langgraph_v2.migrations import build_alembic_config


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
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            graph = build_tracer_graph(checkpointer=saver)
            await graph.ainvoke(
                {
                    "query": "migration checkpoint",
                    "conversation_id": conversation_id,
                    "client_request_id": None,
                    "events": [],
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
            saver = AsyncPostgresSaver(pool)
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
