"""Drop the superseded application execution journal."""

from alembic import op

revision = "0014_drop_run_lifecycle"
down_revision = "0013_artifact_turn_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove application lifecycle tables after task48 code is deployed."""
    op.execute("DROP TABLE langgraph_v2.cancellation_intents")
    op.execute("DROP TABLE langgraph_v2.phase_results")
    op.execute("DROP TABLE langgraph_v2.events")
    op.execute("DROP TABLE langgraph_v2.runs")


def downgrade() -> None:
    """Recreate the 0013-compatible execution journal without historical rows."""
    op.execute(
        """
        CREATE TABLE langgraph_v2.runs (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            conversation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            next_event_sequence BIGINT NOT NULL DEFAULT 1,
            terminal_outcome JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ,
            owner_instance_id TEXT NOT NULL DEFAULT 'legacy',
            execution_epoch BIGINT NOT NULL DEFAULT 1,
            heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL
                DEFAULT now() + interval '30 seconds',
            checkpoint_id TEXT,
            checkpoint_ns TEXT,
            turn_id UUID,
            PRIMARY KEY (tenant_id, run_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE langgraph_v2.events (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            sequence BIGINT NOT NULL,
            event_key TEXT NOT NULL,
            type TEXT NOT NULL,
            step TEXT,
            data JSONB,
            canonical_envelope TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, run_id, sequence),
            UNIQUE (tenant_id, run_id, event_key),
            FOREIGN KEY (tenant_id, run_id)
                REFERENCES langgraph_v2.runs (tenant_id, run_id)
                ON DELETE CASCADE
        )
        """
    )
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
    op.execute(
        """
        CREATE TABLE langgraph_v2.cancellation_intents (
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            requested_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (tenant_id, run_id),
            FOREIGN KEY (tenant_id, run_id)
                REFERENCES langgraph_v2.runs (tenant_id, run_id)
                ON DELETE CASCADE
        )
        """
    )
