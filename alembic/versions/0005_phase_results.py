"""Create the tenant-scoped PhaseResult recovery journal."""

from alembic import op

revision = "0005_phase_results"
down_revision = "0004_run_checkpoint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Store one normalized, replayable result per Run phase."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.phase_results (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            phase_name TEXT NOT NULL CHECK (phase_name IN (
                'query', 'pre_moderation', 'question_refinement', 'retrieval',
                'reranking', 'answer', 'groundedness', 'post_moderation',
                'finalization'
            )),
            execution_epoch BIGINT NOT NULL,
            normalized_result JSONB,
            artifact_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
            event_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
            canonical_result TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, run_id, phase_name),
            FOREIGN KEY (tenant_id, run_id)
                REFERENCES langgraph_v2.runs (tenant_id, run_id)
                ON DELETE CASCADE,
            CHECK (normalized_result IS NOT NULL OR artifact_refs <> '[]'::jsonb)
        )
        """
    )


def downgrade() -> None:
    """Remove the PhaseResult recovery journal."""
    op.execute("DROP TABLE langgraph_v2.phase_results")
