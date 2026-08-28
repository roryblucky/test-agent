"""Store the application-authoritative LangGraph checkpoint identity."""

from alembic import op

revision = "0004_run_checkpoint"
down_revision = "0003_run_claim"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add exact checkpoint identity fields to each application Run."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.runs
            ADD COLUMN checkpoint_id TEXT,
            ADD COLUMN checkpoint_ns TEXT
        """
    )


def downgrade() -> None:
    """Remove the application checkpoint identity fields."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.runs
            DROP COLUMN checkpoint_ns,
            DROP COLUMN checkpoint_id
        """
    )
