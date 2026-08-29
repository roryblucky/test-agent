"""Store Conversation ownership and its stable LangGraph thread identity."""

from alembic import op

revision = "0009_conversation_authorization"
down_revision = "0008_cancellation_intents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the durable Tenant/Subject authorization boundary."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.conversations
            ADD COLUMN owner_subject_id TEXT,
            ADD COLUMN thread_id TEXT
        """
    )
    op.execute(
        """
        UPDATE langgraph_v2.conversations
        SET owner_subject_id = '__unassigned__',
            thread_id = rtrim(
                translate(
                    replace(
                        encode(
                            convert_to(
                                '[' || to_json('thread'::text) || ',' ||
                                to_json(tenant_id) || ',' ||
                                to_json(conversation_id) || ']',
                                'UTF8'
                            ),
                            'base64'
                        ),
                        E'\\n',
                        ''
                    ),
                    '+/',
                    '-_'
                ),
                '='
            )
        WHERE owner_subject_id IS NULL OR thread_id IS NULL
        """
    )
    op.execute(
        """
        ALTER TABLE langgraph_v2.conversations
            ALTER COLUMN owner_subject_id SET NOT NULL,
            ALTER COLUMN thread_id SET NOT NULL
        """
    )
    op.execute(
        """
        ALTER TABLE langgraph_v2.conversations
            ADD CONSTRAINT conversations_tenant_thread_unique
            UNIQUE (tenant_id, thread_id)
        """
    )


def downgrade() -> None:
    """Remove Conversation ownership and thread identity."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.conversations
            DROP CONSTRAINT conversations_tenant_thread_unique,
            DROP COLUMN thread_id,
            DROP COLUMN owner_subject_id
        """
    )
