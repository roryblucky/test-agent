"""Make Turn identity authoritative for durable Messages."""

from alembic import op

revision = "0012_message_turn_identity"
down_revision = "0011_run_turn_association"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove transitional Run identity from Messages."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.messages
            DROP CONSTRAINT messages_tenant_id_run_id_role_key,
            DROP COLUMN run_id
        """
    )


def downgrade() -> None:
    """Deterministically restore prior Run-keyed Message identity."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.messages
            ADD COLUMN run_id UUID
        """
    )
    op.execute(
        """
        UPDATE langgraph_v2.messages
        SET run_id = turn_id
        """
    )
    op.execute(
        """
        ALTER TABLE langgraph_v2.messages
            ALTER COLUMN run_id SET NOT NULL,
            ADD CONSTRAINT messages_tenant_id_run_id_role_key
            UNIQUE (tenant_id, run_id, role)
        """
    )
