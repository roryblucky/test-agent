"""Establish durable Turn identity alongside the transitional Run identity."""

from alembic import op

revision = "0010_turn_identity"
down_revision = "0009_conversation_authorization"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Associate every existing Message with its deterministic Run-based Turn."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.messages
            ADD COLUMN turn_id UUID,
            ADD COLUMN resume_deadline TIMESTAMPTZ
        """
    )
    # Existing Runs are the only stable identity available before this
    # migration. Reusing their UUIDs keeps the expand step deterministic.
    op.execute(
        """
        UPDATE langgraph_v2.messages
        SET turn_id = run_id
        WHERE turn_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE langgraph_v2.messages
        SET resume_deadline = created_at + interval '1 hour'
        WHERE role = 'user' AND resume_deadline IS NULL
        """
    )
    op.execute(
        """
        ALTER TABLE langgraph_v2.messages
            ALTER COLUMN turn_id SET NOT NULL,
            ADD CONSTRAINT messages_resume_deadline_role_check
            CHECK (
                (role = 'user' AND resume_deadline IS NOT NULL)
                OR (role = 'assistant' AND resume_deadline IS NULL)
            ),
            ADD CONSTRAINT messages_turn_role_unique
            UNIQUE (tenant_id, conversation_id, turn_id, role)
        """
    )


def downgrade() -> None:
    """Remove Turn identity while preserving the prior Run-keyed schema."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.messages
            DROP CONSTRAINT messages_turn_role_unique,
            DROP CONSTRAINT messages_resume_deadline_role_check,
            DROP COLUMN resume_deadline,
            DROP COLUMN turn_id
        """
    )
