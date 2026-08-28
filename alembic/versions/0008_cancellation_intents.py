"""Add tenant-scoped cooperative cancellation intents."""

from alembic import op

revision = "0008_cancellation_intents"
down_revision = "0007_conversation_messages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create one durable, idempotent cancellation intent per Run."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.cancellation_intents (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            requested_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, run_id),
            FOREIGN KEY (tenant_id, run_id)
                REFERENCES langgraph_v2.runs (tenant_id, run_id)
                ON DELETE CASCADE
        )
        """
    )


def downgrade() -> None:
    """Remove cooperative cancellation intents."""
    op.execute("DROP TABLE langgraph_v2.cancellation_intents")
