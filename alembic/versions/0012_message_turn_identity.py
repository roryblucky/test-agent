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
        SET run_id = COALESCE(
            (
                SELECT runs.run_id
                FROM langgraph_v2.runs AS runs
                WHERE runs.tenant_id = messages.tenant_id
                  AND runs.conversation_id = messages.conversation_id
                  AND runs.turn_id = messages.turn_id
                ORDER BY
                    CASE
                        WHEN messages.role = 'assistant'
                             AND runs.status = 'completed' THEN 0
                        WHEN messages.role = 'assistant' THEN 1
                        ELSE 0
                    END,
                    runs.created_at,
                    runs.run_id
                LIMIT 1
            ),
            turn_id
        )
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
