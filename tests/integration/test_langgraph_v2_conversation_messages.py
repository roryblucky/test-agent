from __future__ import annotations

from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    MessageInvariantConflict,
)
from app.langgraph_v2.run_events import ClaimFenced, EventInput, RunEventRepository


@pytest.mark.asyncio
async def test_session_identity_resolves_once_per_tenant(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)

        first = await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )
        repeated = await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )
        isolated = await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-b", subject_id="subject-a"),
            conversation_id="session-1",
        )

        assert first == repeated
        assert first.conversation_id == isolated.conversation_id
        assert first.tenant_id != isolated.tenant_id
        with pytest.raises(ConversationNotFound):
            await repository.get_conversation(
                context=TrustedRequestContext(
                    tenant_id="tenant-c", subject_id="subject-a"
                ),
                conversation_id=first.conversation_id,
            )


@pytest.mark.asyncio
async def test_user_message_is_idempotent_and_tenant_scoped(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        run_id = uuid4()
        await repository.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )

        first = await repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="session-1",
            run_id=run_id,
            content="hello",
            idempotency_key="request-1:user",
        )
        repeated = await repository.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="session-1",
            run_id=run_id,
            content="hello",
            idempotency_key="request-1:user",
        )

        assert repeated == first
        assert await repository.list_messages(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        ) == [first]
        with pytest.raises(MessageInvariantConflict):
            await repository.persist_user_message(
                tenant_id="tenant-a",
                conversation_id="session-1",
                run_id=run_id,
                content="changed",
                idempotency_key="request-1:user",
            )
        with pytest.raises(ConversationNotFound):
            await repository.get_message(
                context=TrustedRequestContext(
                    tenant_id="tenant-b", subject_id="subject-a"
                ),
                conversation_id="session-1",
                message_id=first.message_id,
            )


@pytest.mark.asyncio
async def test_assistant_message_requires_successful_run_completion(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        runs = RunEventRepository(pool)
        await messages.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )
        completed = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="session-1",
            owner_instance_id="instance-1",
        )
        failed = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="session-1",
            owner_instance_id="instance-1",
        )
        cancelled = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="session-1",
            owner_instance_id="instance-1",
        )
        interrupted = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="session-1",
            owner_instance_id="instance-1",
        )
        await runs.complete_run(
            tenant_id="tenant-a",
            run_id=completed.run_id,
            event=EventInput(event_key="done", type="done", data={}),
            owner_instance_id=completed.owner_instance_id,
            execution_epoch=completed.execution_epoch,
        )
        await runs.fail_run(
            tenant_id="tenant-a",
            run_id=failed.run_id,
            event=EventInput(event_key="error", type="error", data={}),
            owner_instance_id=failed.owner_instance_id,
            execution_epoch=failed.execution_epoch,
        )
        async with pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE langgraph_v2.runs
                    SET status = CASE
                        WHEN run_id = %s THEN 'cancelled'
                        ELSE 'interrupted'
                    END
                    WHERE tenant_id = %s AND run_id IN (%s, %s)
                    """,
                    (
                        cancelled.run_id,
                        "tenant-a",
                        cancelled.run_id,
                        interrupted.run_id,
                    ),
                )

        first = await messages.persist_assistant_message_after_completion(
            tenant_id="tenant-a",
            conversation_id="session-1",
            run_id=completed.run_id,
            content="safe answer",
            idempotency_key=f"run:{completed.run_id}:assistant",
        )
        repeated = await messages.persist_assistant_message_after_completion(
            tenant_id="tenant-a",
            conversation_id="session-1",
            run_id=completed.run_id,
            content="safe answer",
            idempotency_key=f"run:{completed.run_id}:assistant",
        )

        assert repeated == first
        for blocked in (failed, cancelled, interrupted):
            with pytest.raises(ClaimFenced):
                await messages.persist_assistant_message_after_completion(
                    tenant_id="tenant-a",
                    conversation_id="session-1",
                    run_id=blocked.run_id,
                    content="must not persist",
                    idempotency_key=f"run:{blocked.run_id}:assistant",
                )
        assert [
            message.content
            for message in await messages.list_messages(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-a"
                ),
                conversation_id="session-1",
            )
        ] == ["safe answer"]


@pytest.mark.asyncio
async def test_resume_retry_reuses_the_user_message(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        runs = RunEventRepository(pool)
        await messages.resolve_conversation(
            context=TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a"),
            conversation_id="session-1",
        )
        run = await runs.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="session-1",
            owner_instance_id="instance-1",
        )
        await messages.persist_user_message(
            tenant_id="tenant-a",
            conversation_id="session-1",
            run_id=run.run_id,
            content="hello",
            idempotency_key="retry:user",
        )
        await runs.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id=run.owner_instance_id,
            execution_epoch=run.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        async with pool.connection() as connection:
            async with connection.transaction():
                await connection.execute(
                    "UPDATE langgraph_v2.runs SET status = 'interrupted', owner_instance_id = '' WHERE tenant_id = %s AND run_id = %s",
                    ("tenant-a", run.run_id),
                )
        resumed = await runs.resume_run(
            tenant_id="tenant-a",
            run_id=run.run_id,
            owner_instance_id="instance-2",
        )
        repeated = await messages.persist_user_message(
            tenant_id="tenant-a",
            conversation_id=resumed.conversation_id,
            run_id=run.run_id,
            content="hello",
            idempotency_key="retry:user",
        )

        assert repeated.run_id == run.run_id
        assert (
            len(
                await messages.list_messages(
                    context=TrustedRequestContext(
                        tenant_id="tenant-a", subject_id="subject-a"
                    ),
                    conversation_id="session-1",
                )
            )
            == 1
        )
