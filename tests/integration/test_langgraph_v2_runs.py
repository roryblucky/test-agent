from __future__ import annotations

import asyncio
from uuid import uuid4

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from app.langgraph_v2.runs import (
    ClaimFenced,
    ResumeConflict,
    RunNotFound,
    RunRecord,
    RunRepository,
)


def _expire(database_url: str, run_id: object) -> None:
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            """
            UPDATE langgraph_v2.runs
            SET expires_at = clock_timestamp() - interval '1 second'
            WHERE tenant_id = 'tenant-a' AND run_id = %s
            """,
            (run_id,),
        )


@pytest.mark.asyncio
async def test_resume_claims_expired_run_once_and_fences_old_owner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=4
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=created.run_id,
            owner_instance_id=created.owner_instance_id,
            execution_epoch=created.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        _expire(langgraph_v2_migrated_database_url, created.run_id)

        resumed = await repository.resume_run(
            tenant_id="tenant-a",
            run_id=created.run_id,
            owner_instance_id="instance-b",
        )

        assert resumed.execution_epoch == created.execution_epoch + 1
        assert resumed.owner_instance_id == "instance-b"
        with pytest.raises(ResumeConflict):
            await repository.resume_run(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-c",
            )
        with pytest.raises(ClaimFenced):
            await repository.heartbeat(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-a",
                execution_epoch=created.execution_epoch,
            )


@pytest.mark.asyncio
async def test_resume_requires_an_authoritative_checkpoint(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        _expire(langgraph_v2_migrated_database_url, created.run_id)
        with pytest.raises(ResumeConflict):
            await repository.resume_run(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-b",
            )


@pytest.mark.asyncio
async def test_concurrent_resume_has_one_winner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=4
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=created.run_id,
            owner_instance_id=created.owner_instance_id,
            execution_epoch=created.execution_epoch,
            checkpoint_id="checkpoint-1",
            checkpoint_ns="namespace-1",
        )
        _expire(langgraph_v2_migrated_database_url, created.run_id)
        results = await asyncio.gather(
            repository.resume_run(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-b",
            ),
            repository.resume_run(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-c",
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(item, RunRecord) for item in results) == 1
        assert sum(isinstance(item, ResumeConflict) for item in results) == 1


@pytest.mark.asyncio
async def test_concurrent_resume_of_interrupted_run_has_one_winner(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=4
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        await repository.update_checkpoint_pointer(
            tenant_id="tenant-a",
            run_id=created.run_id,
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
                SET status = 'interrupted', owner_instance_id = ''
                WHERE tenant_id = 'tenant-a' AND run_id = %s
                """,
                (created.run_id,),
            )

        results = await asyncio.gather(
            repository.resume_run(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-b",
            ),
            repository.resume_run(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id="instance-c",
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(item, RunRecord) for item in results) == 1
        assert sum(isinstance(item, ResumeConflict) for item in results) == 1


@pytest.mark.asyncio
async def test_run_access_conceals_another_tenant(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )

        with pytest.raises(RunNotFound):
            await repository.get_run("tenant-b", created.run_id)


@pytest.mark.asyncio
async def test_replaced_claim_epoch_rejects_stale_owner_writes(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=True
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET owner_instance_id = 'instance-b', execution_epoch = 2,
                    expires_at = clock_timestamp() + interval '30 seconds'
                WHERE tenant_id = 'tenant-a' AND run_id = %s
                """,
                (created.run_id,),
            )

        with pytest.raises(ClaimFenced):
            await repository.heartbeat(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id=created.owner_instance_id,
                execution_epoch=created.execution_epoch,
            )
        with pytest.raises(ClaimFenced):
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id=created.owner_instance_id,
                execution_epoch=created.execution_epoch,
                checkpoint_id="stale",
                checkpoint_ns="",
            )


@pytest.mark.asyncio
async def test_expired_claim_rejects_heartbeat_and_checkpoint_write(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        refreshed = await repository.heartbeat(
            tenant_id="tenant-a",
            run_id=created.run_id,
            owner_instance_id=created.owner_instance_id,
            execution_epoch=created.execution_epoch,
        )
        assert refreshed.expires_at > refreshed.heartbeat_at
        _expire(langgraph_v2_migrated_database_url, created.run_id)

        with pytest.raises(ClaimFenced):
            await repository.heartbeat(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id=created.owner_instance_id,
                execution_epoch=created.execution_epoch,
            )
        with pytest.raises(ClaimFenced):
            await repository.update_checkpoint_pointer(
                tenant_id="tenant-a",
                run_id=created.run_id,
                owner_instance_id=created.owner_instance_id,
                execution_epoch=created.execution_epoch,
                checkpoint_id="too-late",
                checkpoint_ns="",
            )


@pytest.mark.asyncio
async def test_claim_expiry_is_rechecked_after_waiting_for_run_lock(
    langgraph_v2_migrated_database_url: str,
) -> None:
    async with AsyncConnectionPool(
        langgraph_v2_migrated_database_url, min_size=1, max_size=2
    ) as pool:
        repository = RunRepository(pool)
        created = await repository.create_run(
            tenant_id="tenant-a",
            run_id=uuid4(),
            conversation_id="conversation-1",
            owner_instance_id="instance-a",
        )
        with psycopg.connect(
            langgraph_v2_migrated_database_url, autocommit=False
        ) as connection:
            connection.execute(
                """
                UPDATE langgraph_v2.runs
                SET expires_at = clock_timestamp() + interval '50 milliseconds'
                WHERE tenant_id = 'tenant-a' AND run_id = %s
                """,
                (created.run_id,),
            )
            connection.commit()
            connection.execute(
                """
                SELECT run_id FROM langgraph_v2.runs
                WHERE tenant_id = 'tenant-a' AND run_id = %s FOR UPDATE
                """,
                (created.run_id,),
            )
            pending = asyncio.create_task(
                repository.heartbeat(
                    tenant_id="tenant-a",
                    run_id=created.run_id,
                    owner_instance_id=created.owner_instance_id,
                    execution_epoch=created.execution_epoch,
                )
            )
            await asyncio.sleep(0.1)
            connection.commit()
            with pytest.raises(ClaimFenced):
                await pending
