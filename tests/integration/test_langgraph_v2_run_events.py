from __future__ import annotations

import asyncio
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.run_events import (
    ClaimFenced,
    EventInput,
    EventInvariantConflict,
    EventNotFound,
    ResumeConflict,
    RunEventRepository,
    RunNotFound,
    RunRecord,
)


@pytest.mark.asyncio
async def test_run_and_event_are_persisted_and_read_back(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()

        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="local",
        )
        appended = await repository.append_event(
            tenant_id="tenant-a",
            run_id=run_id,
            event=EventInput(
                event_key="phase:query:step_start:1",
                type="step_start",
                step="query",
            ),
            owner_instance_id="local",
            execution_epoch=1,
        )

        assert created.status == "running"
        assert appended.sequence == 1
        assert await repository.get_run("tenant-a", run_id) == created
        assert await repository.list_events("tenant-a", run_id) == [appended]


@pytest.mark.asyncio
async def test_resume_claims_expired_run_once_and_fences_old_owner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=4
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=created.owner_instance_id,
            execution_epoch=created.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )
        resumed = await repository.resume_run(
            tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-b"
        )
        assert resumed.execution_epoch == created.execution_epoch + 1
        assert resumed.owner_instance_id == "instance-b"
        with pytest.raises(ResumeConflict):
            await repository.resume_run(
                tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-c"
            )
        with pytest.raises(ClaimFenced):
            await repository.append_event(
                tenant_id="tenant-a",
                run_id=run_id,
                event=EventInput(event_key="old", type="step_start"),
                owner_instance_id="instance-a",
                execution_epoch=created.execution_epoch,
            )


@pytest.mark.asyncio
async def test_resume_rejects_run_without_authoritative_checkpoint(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )
        with pytest.raises(ResumeConflict):
            await repository.resume_run(
                tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-b"
            )


@pytest.mark.asyncio
async def test_resume_concurrent_requests_have_one_winner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=4
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=created.owner_instance_id,
            execution_epoch=created.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )
        results = await asyncio.gather(
            repository.resume_run(
                tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-b"
            ),
            repository.resume_run(
                tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-c"
            ),
            return_exceptions=True,
        )
        winners = [item for item in results if isinstance(item, RunRecord)]
        assert len(winners) == 1
        assert sum(isinstance(item, ResumeConflict) for item in results) == 1
        winner = winners[0]
        await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run_id,
            event=EventInput(event_key="recovery:done", type="done", data={}),
            owner_instance_id=winner.owner_instance_id,
            execution_epoch=winner.execution_epoch,
        )
        completed = await repository.get_run("tenant-a", run_id)
        assert completed.execution_epoch == 2
        assert completed.status == "completed"


@pytest.mark.asyncio
async def test_resume_concurrent_interrupted_run_has_one_winner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id=created.owner_instance_id,
            execution_epoch=created.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET status = 'interrupted', owner_instance_id = '',
                    expires_at = clock_timestamp() + interval '30 seconds'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )
        results = await asyncio.gather(
            repository.resume_run(
                tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-b"
            ),
            repository.resume_run(
                tenant_id="tenant-a", run_id=run_id, owner_instance_id="instance-c"
            ),
            return_exceptions=True,
        )
        winners = [item for item in results if isinstance(item, RunRecord)]
        assert len(winners) == 1
        assert sum(isinstance(item, ResumeConflict) for item in results) == 1
        winner = winners[0]
        await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run_id,
            event=EventInput(event_key="recovery:done", type="done", data={}),
            owner_instance_id=winner.owner_instance_id,
            execution_epoch=winner.execution_epoch,
        )
        completed = await repository.get_run("tenant-a", run_id)
        assert completed.status == "completed"
        assert completed.execution_epoch == 2


@pytest.mark.asyncio
async def test_run_and_event_access_conceals_another_tenants_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="local",
        )

        with pytest.raises(RunNotFound):
            await repository.get_run("tenant-b", run_id)
        with pytest.raises(RunNotFound):
            await repository.list_events("tenant-b", run_id)
        with pytest.raises(RunNotFound):
            await repository.append_event(
                tenant_id="tenant-b",
                run_id=run_id,
                event=EventInput(
                    event_key="phase:query:step_start:1",
                    type="step_start",
                    step="query",
                ),
                owner_instance_id="local",
                execution_epoch=1,
            )

        persisted = await repository.append_event(
            tenant_id="tenant-a",
            run_id=run_id,
            event=EventInput(
                event_key="lifecycle:started:0",
                type="step_start",
                data={"status": "running"},
            ),
            owner_instance_id="local",
            execution_epoch=1,
        )
        assert (
            await repository.get_event(
                "tenant-a",
                run_id,
                "lifecycle:started:0",
            )
            == persisted
        )
        with pytest.raises(EventNotFound):
            await repository.get_event(
                "tenant-b",
                run_id,
                "lifecycle:started:0",
            )
        with pytest.raises(RunNotFound):
            await repository.complete_run(
                tenant_id="tenant-b",
                run_id=run_id,
                event=EventInput(
                    event_key="lifecycle:completed:0",
                    type="done",
                    data={"status": "completed"},
                ),
                owner_instance_id="local",
                execution_epoch=1,
            )


@pytest.mark.asyncio
async def test_lifecycle_event_retry_is_idempotent_and_conflicts_fail_the_run(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="local",
        )
        first = await repository.append_event(
            tenant_id="tenant-a",
            run_id=run_id,
            event=EventInput(
                event_key="lifecycle:started:0",
                type="step_start",
                data={"b": 2, "a": 1},
            ),
            owner_instance_id="local",
            execution_epoch=1,
        )

        repeated = await repository.append_event(
            tenant_id="tenant-a",
            run_id=run_id,
            event=EventInput(
                event_key="lifecycle:started:0",
                type="step_start",
                data={"a": 1, "b": 2},
            ),
            owner_instance_id="local",
            execution_epoch=1,
        )

        assert repeated == first
        with pytest.raises(EventInvariantConflict):
            await repository.append_event(
                tenant_id="tenant-a",
                run_id=run_id,
                event=EventInput(
                    event_key="lifecycle:started:0",
                    type="step_start",
                    data={"a": 999, "b": 2},
                ),
                owner_instance_id="local",
                execution_epoch=1,
            )

        assert (await repository.get_run("tenant-a", run_id)).status == "failed"
        events = await repository.list_events("tenant-a", run_id)
        assert events == [first]


@pytest.mark.asyncio
async def test_terminal_event_retry_is_atomic_and_idempotent(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="local",
        )
        terminal = EventInput(
            event_key="lifecycle:completed:0",
            type="done",
            data={"answer": None, "session_id": "conversation-1"},
        )

        first = await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run_id,
            event=terminal,
            owner_instance_id="local",
            execution_epoch=1,
        )
        repeated = await repository.complete_run(
            tenant_id="tenant-a",
            run_id=run_id,
            event=terminal,
            owner_instance_id="local",
            execution_epoch=1,
        )

        completed = await repository.get_run("tenant-a", run_id)
        assert repeated == first
        assert completed.status == "completed"
        assert completed.terminal_outcome == terminal.data
        assert completed.completed_at is not None
        assert await repository.list_events("tenant-a", run_id) == [first]

        with pytest.raises(EventInvariantConflict):
            await repository.complete_run(
                tenant_id="tenant-a",
                run_id=run_id,
                event=terminal.model_copy(update={"data": {"answer": "different"}}),
                owner_instance_id="local",
                execution_epoch=1,
            )
        assert (await repository.get_run("tenant-a", run_id)).status == "failed"


@pytest.mark.asyncio
async def test_concurrent_event_appends_allocate_one_ordered_sequence(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=4,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="local",
        )

        await asyncio.gather(
            *(
                repository.append_event(
                    tenant_id="tenant-a",
                    run_id=run_id,
                    event=EventInput(
                        event_key=f"phase:query:progress:{ordinal}",
                        type="step_completed",
                        step="query",
                        data={"ordinal": ordinal},
                    ),
                    owner_instance_id="local",
                    execution_epoch=1,
                )
                for ordinal in range(1, 9)
            )
        )

        persisted = await repository.list_events("tenant-a", run_id)
        assert [event.sequence for event in persisted] == list(range(1, 9))
        assert {event.event_key for event in persisted} == {
            f"phase:query:progress:{ordinal}" for ordinal in range(1, 9)
        }


@pytest.mark.asyncio
async def test_expired_claim_rejects_heartbeat_and_writes(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        refreshed = await repository.heartbeat(
            tenant_id="tenant-a",
            run_id=run_id,
            owner_instance_id="instance-a",
            execution_epoch=created.execution_epoch,
        )
        assert refreshed.expires_at > refreshed.heartbeat_at

        with psycopg.connect(
            langgraph_v2_migrated_database_url,
            autocommit=True,
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = now() - interval '1 second'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )

        with pytest.raises(ClaimFenced):
            await repository.heartbeat(
                tenant_id="tenant-a",
                run_id=run_id,
                owner_instance_id="instance-a",
                execution_epoch=created.execution_epoch,
            )
        with pytest.raises(ClaimFenced):
            await repository.append_event(
                tenant_id="tenant-a",
                run_id=run_id,
                event=EventInput(
                    event_key="phase:query:step_start:1",
                    type="step_start",
                    step="query",
                ),
                owner_instance_id="instance-a",
                execution_epoch=created.execution_epoch,
            )
        with pytest.raises(ClaimFenced):
            await repository.complete_run(
                tenant_id="tenant-a",
                run_id=run_id,
                event=EventInput(
                    event_key="lifecycle:completed:0",
                    type="done",
                    data={"status": "completed"},
                ),
                owner_instance_id="instance-a",
                execution_epoch=created.execution_epoch,
            )


@pytest.mark.asyncio
async def test_replaced_claim_epoch_rejects_stale_owner_writes(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )

        with psycopg.connect(
            langgraph_v2_migrated_database_url,
            autocommit=True,
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET owner_instance_id = 'instance-b',
                    execution_epoch = 2,
                    heartbeat_at = now(),
                    expires_at = now() + interval '30 seconds'
                WHERE tenant_id = %s AND run_id = %s
                """,
                ("tenant-a", run_id),
            )

        stale_claim = {
            "tenant_id": "tenant-a",
            "run_id": run_id,
            "owner_instance_id": "instance-a",
            "execution_epoch": created.execution_epoch,
        }
        with pytest.raises(ClaimFenced):
            await repository.heartbeat(**stale_claim)
        with pytest.raises(ClaimFenced):
            await repository.append_event(
                **stale_claim,
                event=EventInput(
                    event_key="phase:query:step_start:1",
                    type="step_start",
                    step="query",
                ),
            )
        with pytest.raises(ClaimFenced):
            await repository.complete_run(
                **stale_claim,
                event=EventInput(
                    event_key="lifecycle:completed:0",
                    type="done",
                    data={"status": "completed"},
                ),
            )


@pytest.mark.asyncio
async def test_claim_expiry_is_rechecked_after_waiting_for_run_lock(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url,
        min_size=1,
        max_size=2,
    ) as pool:
        repository = RunEventRepository(pool)
        run_id = uuid4()
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=run_id,
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )

        with psycopg.connect(
            langgraph_v2_migrated_database_url,
            autocommit=False,
        ) as connection:
            for operation in ("heartbeat", "append", "complete"):
                connection.execute(
                    """
                    UPDATE langgraph_v2.runs
                    SET expires_at = clock_timestamp() + interval '50 milliseconds'
                    WHERE tenant_id = %s AND run_id = %s
                    """,
                    ("tenant-a", run_id),
                )
                connection.commit()
                connection.execute(
                    """
                    SELECT run_id
                    FROM langgraph_v2.runs
                    WHERE tenant_id = %s AND run_id = %s
                    FOR UPDATE
                    """,
                    ("tenant-a", run_id),
                )

                if operation == "heartbeat":
                    pending = repository.heartbeat(
                        tenant_id="tenant-a",
                        run_id=run_id,
                        owner_instance_id="instance-a",
                        execution_epoch=created.execution_epoch,
                    )
                elif operation == "append":
                    pending = repository.append_event(
                        tenant_id="tenant-a",
                        run_id=run_id,
                        event=EventInput(
                            event_key="phase:query:lock_wait:1",
                            type="step_start",
                            step="query",
                        ),
                        owner_instance_id="instance-a",
                        execution_epoch=created.execution_epoch,
                    )
                else:
                    pending = repository.complete_run(
                        tenant_id="tenant-a",
                        run_id=run_id,
                        event=EventInput(
                            event_key="lifecycle:lock_wait:0",
                            type="done",
                            data={"status": "completed"},
                        ),
                        owner_instance_id="instance-a",
                        execution_epoch=created.execution_epoch,
                    )
                task = asyncio.create_task(pending)
                await asyncio.sleep(0.1)
                connection.commit()
                with pytest.raises(ClaimFenced):
                    await task
