from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.authorization import TrustedRequestContext
from app.langgraph_v2.conversation_messages import (
    ConversationMessageRepository,
    ConversationNotFound,
    ConversationRecord,
    MessageInvariantConflict,
    ResumeExpired,
    TurnNotFound,
    TurnSuperseded,
)
from app.langgraph_v2.runs import RunRecord, RunRepository


class _PausingAdmissionRepository(ConversationMessageRepository):
    """Pause after acquiring the Conversation admission lock."""

    def __init__(
        self,
        pool: AsyncConnectionPool[Any],
        *,
        lock_acquired: asyncio.Event,
        release_lock: asyncio.Event,
    ) -> None:
        super().__init__(pool)
        self._lock_acquired = lock_acquired
        self._release_lock = release_lock

    async def _resolve_conversation_in_transaction(
        self,
        connection: Any,
        *,
        context: TrustedRequestContext,
        conversation_id: str,
        for_update: bool = False,
    ) -> ConversationRecord:
        conversation = await super()._resolve_conversation_in_transaction(
            connection,
            context=context,
            conversation_id=conversation_id,
            for_update=for_update,
        )
        if for_update:
            self._lock_acquired.set()
            await self._release_lock.wait()
        return conversation


async def _seed_admission_case(
    messages: ConversationMessageRepository,
    runs: RunRepository,
    context: TrustedRequestContext,
) -> tuple[UUID, RunRecord]:
    await messages.resolve_conversation(context=context, conversation_id="session-1")
    original_turn_id = uuid4()
    await messages.create_turn(
        context=context,
        conversation_id="session-1",
        turn_id=original_turn_id,
        content="original question",
        idempotency_key=f"turn:{original_turn_id}:user",
    )
    resume_run = await runs.create_run(
        tenant_id="tenant-a",
        run_id=uuid4(),
        conversation_id="session-1",
        owner_instance_id="resume-instance",
    )
    return original_turn_id, resume_run


@pytest.mark.asyncio
async def test_query_admission_blocks_resume_then_supersedes_it(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=2, max_size=4
    ) as pool:
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        messages = ConversationMessageRepository(pool)
        original_turn_id, resume_run = await _seed_admission_case(
            messages, RunRepository(pool), context
        )
        lock_acquired = asyncio.Event()
        release_lock = asyncio.Event()
        query_messages = _PausingAdmissionRepository(
            pool,
            lock_acquired=lock_acquired,
            release_lock=release_lock,
        )
        new_turn_id = uuid4()
        query_task = asyncio.create_task(
            query_messages.create_turn(
                context=context,
                conversation_id="session-1",
                turn_id=new_turn_id,
                content="new question",
                idempotency_key=f"turn:{new_turn_id}:user",
            )
        )
        await asyncio.wait_for(lock_acquired.wait(), timeout=1)
        resume_task = asyncio.create_task(
            messages.associate_run_with_turn(
                context=context,
                conversation_id="session-1",
                run_id=resume_run.run_id,
                owner_instance_id=resume_run.owner_instance_id,
                execution_epoch=resume_run.execution_epoch,
                turn_id=original_turn_id,
            )
        )

        await asyncio.sleep(0.05)
        assert not resume_task.done()
        release_lock.set()
        await query_task
        with pytest.raises(TurnSuperseded):
            await resume_task


@pytest.mark.asyncio
async def test_resume_admission_blocks_query_until_resume_is_bound(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=2, max_size=4
    ) as pool:
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        messages = ConversationMessageRepository(pool)
        original_turn_id, resume_run = await _seed_admission_case(
            messages, RunRepository(pool), context
        )
        lock_acquired = asyncio.Event()
        release_lock = asyncio.Event()
        resume_messages = _PausingAdmissionRepository(
            pool,
            lock_acquired=lock_acquired,
            release_lock=release_lock,
        )
        resume_task = asyncio.create_task(
            resume_messages.associate_run_with_turn(
                context=context,
                conversation_id="session-1",
                run_id=resume_run.run_id,
                owner_instance_id=resume_run.owner_instance_id,
                execution_epoch=resume_run.execution_epoch,
                turn_id=original_turn_id,
            )
        )
        await asyncio.wait_for(lock_acquired.wait(), timeout=1)
        new_turn_id = uuid4()
        query_task = asyncio.create_task(
            messages.create_turn(
                context=context,
                conversation_id="session-1",
                turn_id=new_turn_id,
                content="new question",
                idempotency_key=f"turn:{new_turn_id}:user",
            )
        )

        await asyncio.sleep(0.05)
        assert not query_task.done()
        release_lock.set()
        await resume_task
        await query_task


@pytest.mark.asyncio
async def test_turn_creation_is_idempotent_and_resume_deadline_uses_user_message_time(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        repository = ConversationMessageRepository(
            pool, resume_ttl=timedelta(minutes=5)
        )
        await repository.resolve_conversation(
            context=context,
            conversation_id="session-1",
        )
        turn_id = uuid4()
        first = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key="turn-1:user",
        )
        repeated = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key="turn-1:user",
        )

        assert repeated == first
        assert first.turn_id == turn_id
        original_deadline = first.resume_deadline
        assert original_deadline - first.created_at == timedelta(minutes=5)
        message = (
            await repository.list_messages(context=context, conversation_id="session-1")
        )[0]
        assert message.turn_id == turn_id

        with pytest.raises(MessageInvariantConflict):
            await repository.create_turn(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                content="changed",
                idempotency_key="turn-1:user",
            )
        with pytest.raises(MessageInvariantConflict):
            await repository.create_turn(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                content="hello",
                idempotency_key="turn-1:different-key",
            )

        repository_with_longer_ttl = ConversationMessageRepository(
            pool, resume_ttl=timedelta(minutes=30)
        )
        retried_after_config_change = await repository_with_longer_ttl.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key="turn-1:user",
        )
        turn = await repository_with_longer_ttl.get_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
        )
        assert retried_after_config_change == first
        assert turn.resume_deadline == original_deadline
        with pytest.raises(ResumeExpired):
            await repository.get_turn_for_resume(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                now=original_deadline + timedelta(seconds=1),
            )


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
async def test_turn_messages_are_idempotent_without_a_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        await repository.resolve_conversation(
            context=context,
            conversation_id="session-1",
        )
        turn_id = uuid4()

        first_turn = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        repeated_turn = await repository.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="hello",
            idempotency_key=f"turn:{turn_id}:user",
        )
        first = await repository.persist_assistant_message(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="answer",
            idempotency_key=f"turn:{turn_id}:assistant",
        )
        repeated = await repository.persist_assistant_message(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="answer",
            idempotency_key=f"turn:{turn_id}:assistant",
        )

        assert repeated == first
        assert repeated_turn == first_turn
        records = await repository.list_messages(
            context=context, conversation_id="session-1"
        )
        assert [(record.role, record.content) for record in records] == [
            ("user", "hello"),
            ("assistant", "answer"),
        ]
        with pytest.raises(MessageInvariantConflict):
            await repository.persist_assistant_message(
                context=context,
                conversation_id="session-1",
                turn_id=turn_id,
                content="changed",
                idempotency_key=f"turn:{turn_id}:assistant",
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
async def test_assistant_message_requires_an_existing_user_turn(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        messages = ConversationMessageRepository(pool)
        context = TrustedRequestContext(tenant_id="tenant-a", subject_id="subject-a")
        await messages.resolve_conversation(
            context=context,
            conversation_id="session-1",
        )
        turn_id = uuid4()
        await messages.create_turn(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="question",
            idempotency_key=f"turn:{turn_id}:user",
        )
        assistant = await messages.persist_assistant_message(
            context=context,
            conversation_id="session-1",
            turn_id=turn_id,
            content="safe answer",
            idempotency_key=f"turn:{turn_id}:assistant",
        )
        assert assistant.turn_id == turn_id
        with pytest.raises(ConversationNotFound):
            await messages.persist_assistant_message(
                context=TrustedRequestContext(
                    tenant_id="tenant-a", subject_id="subject-b"
                ),
                conversation_id="session-1",
                turn_id=turn_id,
                content="unauthorized",
                idempotency_key="unauthorized:assistant",
            )
        with pytest.raises(TurnNotFound):
            await messages.persist_assistant_message(
                context=context,
                conversation_id="session-1",
                turn_id=uuid4(),
                content="orphan",
                idempotency_key="orphan:assistant",
            )
