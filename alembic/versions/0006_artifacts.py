"""Create tenant-scoped v2 Artifact storage."""

from alembic import op

revision = "0006_artifacts"
down_revision = "0005_phase_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE langgraph_v2.artifacts (
            tenant_id TEXT NOT NULL,
            artifact_id UUID NOT NULL,
            artifact_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, artifact_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE langgraph_v2.artifacts")
