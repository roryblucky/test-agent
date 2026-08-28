"""Create tenant-scoped Run and Event tables."""

from alembic import op

revision = "0002_run_events"
down_revision = "0001_langgraph_v2_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the minimal application-owned Run and Event schema."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.runs (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            next_event_sequence BIGINT NOT NULL DEFAULT 1,
            terminal_outcome JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ,
            PRIMARY KEY (tenant_id, run_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE langgraph_v2.events (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            sequence BIGINT NOT NULL,
            event_key TEXT NOT NULL,
            type TEXT NOT NULL,
            step TEXT,
            data JSONB,
            canonical_envelope TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, run_id, sequence),
            UNIQUE (tenant_id, run_id, event_key),
            FOREIGN KEY (tenant_id, run_id)
                REFERENCES langgraph_v2.runs (tenant_id, run_id)
                ON DELETE CASCADE
        )
        """
    )


def downgrade() -> None:
    """Remove the minimal Run and Event schema."""
    op.execute("DROP TABLE langgraph_v2.events")
    op.execute("DROP TABLE langgraph_v2.runs")
