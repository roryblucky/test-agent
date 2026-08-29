"""Store the authoritative Turn associated with each transitional Run."""

from alembic import op

revision = "0011_run_turn_association"
down_revision = "0010_turn_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Carry the durable Turn identity on Runs during the transition."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.runs
            ADD COLUMN turn_id UUID
        """
    )
    op.execute(
        """
        UPDATE langgraph_v2.runs AS runs
        SET turn_id = messages.turn_id
        FROM langgraph_v2.messages AS messages
        WHERE messages.tenant_id = runs.tenant_id
          AND messages.run_id = runs.run_id
          AND messages.role = 'user'
          AND runs.turn_id IS NULL
        """
    )


def downgrade() -> None:
    """Remove the transitional Run-to-Turn association."""
    op.execute("ALTER TABLE langgraph_v2.runs DROP COLUMN turn_id")
