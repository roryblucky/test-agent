"""Drop persisted retrieval Artifacts."""

from alembic import op

revision = "0015_drop_artifacts"
down_revision = "0014_drop_run_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove full retrieval chunks from durable PostgreSQL storage."""
    op.execute("DROP TABLE langgraph_v2.artifacts")


def downgrade() -> None:
    """Restore the prior Turn-scoped Artifact table without historical rows."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.artifacts (
            tenant_id TEXT NOT NULL,
            artifact_id UUID NOT NULL,
            artifact_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            conversation_id TEXT NOT NULL,
            turn_id UUID NOT NULL,
            turn_role TEXT GENERATED ALWAYS AS ('user'::TEXT) STORED,
            PRIMARY KEY (tenant_id, artifact_id),
            CONSTRAINT artifacts_turn_fk
                FOREIGN KEY (tenant_id, conversation_id, turn_id, turn_role)
                REFERENCES langgraph_v2.messages
                    (tenant_id, conversation_id, turn_id, role)
                ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """CREATE INDEX artifacts_turn_scope_idx
        ON langgraph_v2.artifacts
        (tenant_id, conversation_id, turn_id, created_at, artifact_id)"""
    )
