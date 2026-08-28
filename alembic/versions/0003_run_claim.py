"""Add the fenced direct-execution claim to each Run."""

from alembic import op

revision = "0003_run_claim"
down_revision = "0002_run_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add owner, epoch, and expiring heartbeat fields."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.runs
            ADD COLUMN owner_instance_id TEXT NOT NULL DEFAULT 'legacy',
            ADD COLUMN execution_epoch BIGINT NOT NULL DEFAULT 1,
            ADD COLUMN heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            ADD COLUMN expires_at TIMESTAMPTZ NOT NULL
                DEFAULT now() + interval '30 seconds'
        """
    )


def downgrade() -> None:
    """Remove the direct-execution claim fields."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.runs
            DROP COLUMN expires_at,
            DROP COLUMN heartbeat_at,
            DROP COLUMN execution_epoch,
            DROP COLUMN owner_instance_id
        """
    )
