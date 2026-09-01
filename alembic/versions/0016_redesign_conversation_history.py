"""Replace Turn-keyed persistence with ordered Conversation history."""

from alembic import op

revision = "0016_history_redesign"
down_revision = "0015_drop_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the pre-release Query-only History schema."""
    op.execute("DROP TABLE langgraph_v2.messages")
    op.execute("DROP TABLE langgraph_v2.conversations")
    op.execute(
        """
        CREATE TABLE langgraph_v2.conversations (
            conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            owner_subject_id TEXT NOT NULL,
            runtime_mode TEXT NOT NULL
                CHECK (runtime_mode IN ('linear', 'agent')),
            next_message_sequence BIGINT NOT NULL DEFAULT 1
                CHECK (next_message_sequence > 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX conversations_history_idx
        ON langgraph_v2.conversations (
            tenant_id, owner_subject_id, updated_at DESC
        )
        """
    )
    op.execute(
        """
        CREATE TABLE langgraph_v2.messages (
            message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL
                REFERENCES langgraph_v2.conversations (conversation_id)
                ON DELETE CASCADE,
            request_id TEXT NOT NULL,
            sequence BIGINT NOT NULL CHECK (sequence > 0),
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (conversation_id, request_id, role),
            UNIQUE (conversation_id, sequence)
        )
        """
    )


def downgrade() -> None:
    """Remove the pre-release History schema.

    Revision 0016 is the first-release schema boundary. The repository has no
    deployed pre-0016 data, so downgrade intentionally restores empty legacy
    tables instead of claiming a data-preserving conversion.
    """
    op.execute("DROP TABLE langgraph_v2.messages")
    op.execute("DROP TABLE langgraph_v2.conversations")
    op.execute(
        """
        CREATE TABLE langgraph_v2.conversations (
            tenant_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            owner_subject_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, conversation_id),
            CONSTRAINT conversations_tenant_thread_unique
                UNIQUE (tenant_id, thread_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE langgraph_v2.messages (
            tenant_id TEXT NOT NULL,
            message_id UUID NOT NULL,
            conversation_id TEXT NOT NULL,
            turn_id UUID NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, message_id),
            UNIQUE (tenant_id, idempotency_key),
            CONSTRAINT messages_turn_role_unique
                UNIQUE (tenant_id, conversation_id, turn_id, role),
            FOREIGN KEY (tenant_id, conversation_id)
                REFERENCES langgraph_v2.conversations (tenant_id, conversation_id)
                ON DELETE CASCADE
        )
        """
    )
