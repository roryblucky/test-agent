"""Drop the superseded Conversation registry.

Revision ID: 0018_drop_registry
Revises: 0017_drop_history
"""

from alembic import op

revision = "0018_drop_registry"
down_revision = "0017_drop_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove product Conversation persistence from the Linear Core."""
    op.execute("DROP TABLE langgraph_v2.conversations")


def downgrade() -> None:
    """Restore the empty pre-release Conversation registry shape."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.conversations (
            conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            owner_subject_id TEXT NOT NULL,
            runtime_mode TEXT NOT NULL
                CONSTRAINT conversations_runtime_mode_check
                CHECK (runtime_mode IN ('linear', 'agent')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX conversations_owner_idx
        ON langgraph_v2.conversations (tenant_id, owner_subject_id)
        """
    )
