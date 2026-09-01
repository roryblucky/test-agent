-- LangGraph Linear Core PostgreSQL bootstrap DDL.
--
-- This file represents the schema at Alembic revision
-- 0016_history_redesign. It is intended for a new, empty PostgreSQL
-- database. Existing databases should continue to use Alembic migrations.
--
-- The public checkpoint tables match langgraph-checkpoint-postgres 3.1.2.
-- When that dependency is upgraded, use AsyncPostgresSaver.setup() to apply
-- any newer official checkpoint migrations.

BEGIN;

CREATE SCHEMA langgraph_v2;

CREATE TABLE langgraph_v2.conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    owner_subject_id TEXT NOT NULL,
    runtime_mode TEXT NOT NULL
        CONSTRAINT conversations_runtime_mode_check
        CHECK (runtime_mode IN ('linear', 'agent')),
    next_message_sequence BIGINT NOT NULL DEFAULT 1
        CONSTRAINT conversations_next_message_sequence_check
        CHECK (next_message_sequence > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX conversations_history_idx
    ON langgraph_v2.conversations (
        tenant_id, owner_subject_id, updated_at DESC
    );

CREATE TABLE langgraph_v2.messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL
        REFERENCES langgraph_v2.conversations (conversation_id)
        ON DELETE CASCADE,
    request_id TEXT NOT NULL,
    sequence BIGINT NOT NULL
        CONSTRAINT messages_sequence_check CHECK (sequence > 0),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT messages_role_check
        CHECK (role IN ('user', 'assistant')),
    CONSTRAINT messages_request_role_unique
        UNIQUE (conversation_id, request_id, role),
    CONSTRAINT messages_conversation_sequence_unique
        UNIQUE (conversation_id, sequence)
);

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
VALUES ('0016_history_redesign');

COMMIT;
