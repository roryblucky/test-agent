"""Remove product Message History from the pre-release Linear Core."""

from alembic import op

revision = "0017_drop_history"
down_revision = "0016_history_redesign"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Keep only the Conversation ownership and lifecycle registry."""
    op.execute("DROP TABLE langgraph_v2.messages")
    op.execute("DROP INDEX langgraph_v2.conversations_history_idx")
    op.execute(
        "ALTER TABLE langgraph_v2.conversations DROP COLUMN next_message_sequence"
    )
    op.execute(
        """
        CREATE INDEX conversations_owner_idx
        ON langgraph_v2.conversations (tenant_id, owner_subject_id)
        """
    )


def downgrade() -> None:
    """Restore the empty pre-release Message History schema."""
    op.execute("DROP INDEX langgraph_v2.conversations_owner_idx")
    op.execute(
        """
        ALTER TABLE langgraph_v2.conversations
        ADD COLUMN next_message_sequence BIGINT NOT NULL DEFAULT 1
            CHECK (next_message_sequence > 0)
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
