"""Create tenant-scoped Conversations and Messages."""

from alembic import op

revision = "0007_conversation_messages"
down_revision = "0006_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create durable Conversation and exactly-once Message storage."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.conversations (
            tenant_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, conversation_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE langgraph_v2.messages (
            tenant_id TEXT NOT NULL,
            message_id UUID NOT NULL,
            conversation_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, message_id),
            UNIQUE (tenant_id, idempotency_key),
            UNIQUE (tenant_id, run_id, role),
            FOREIGN KEY (tenant_id, conversation_id)
                REFERENCES langgraph_v2.conversations (tenant_id, conversation_id)
                ON DELETE CASCADE
        )
        """
    )


def downgrade() -> None:
    """Remove Conversation and Message storage."""
    op.execute("DROP TABLE langgraph_v2.messages")
    op.execute("DROP TABLE langgraph_v2.conversations")
