-- LangGraph Linear Core PostgreSQL bootstrap DDL.
--
-- This file represents the schema at Alembic revision
-- 0018_drop_registry. It is intended for a new, empty PostgreSQL
-- database. Existing databases should continue to use Alembic migrations.
--
-- The public checkpoint tables match langgraph-checkpoint-postgres 3.1.2.
-- When that dependency is upgraded, use AsyncPostgresSaver.setup() to apply
-- any newer official checkpoint migrations.

BEGIN;

CREATE SCHEMA langgraph_v2;

-- Official LangGraph PostgreSQL checkpointer tables intentionally remain in
-- the public schema because AsyncPostgresSaver uses these unqualified names.

CREATE TABLE public.checkpoint_migrations (
    v INTEGER PRIMARY KEY
);

CREATE TABLE public.checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE public.checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE public.checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX checkpoints_thread_id_idx
    ON public.checkpoints (thread_id);

CREATE INDEX checkpoint_blobs_thread_id_idx
    ON public.checkpoint_blobs (thread_id);

CREATE INDEX checkpoint_writes_thread_id_idx
    ON public.checkpoint_writes (thread_id);

-- Mark every migration bundled with langgraph-checkpoint-postgres 3.1.2 as
-- applied. AsyncPostgresSaver.setup() will apply later versions when present.
INSERT INTO public.checkpoint_migrations (v)
VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9);

-- Mark the application-owned schema at the matching Alembic head so the
-- normal application lifespan can safely run `alembic upgrade head`.
CREATE TABLE public.alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

INSERT INTO public.alembic_version (version_num)
VALUES ('0018_drop_registry');

COMMIT;
