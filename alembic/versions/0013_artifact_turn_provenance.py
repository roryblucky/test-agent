"""Make Conversation and Turn authoritative Artifact provenance."""

from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import text

from alembic import op

revision = "0013_artifact_turn_provenance"
down_revision = "0012_message_turn_identity"
branch_labels = None
depends_on = None


def _legacy_artifact_id(
    *,
    tenant_id: str,
    conversation_id: str,
    scope_kind: str,
    scope_id: UUID,
    artifact_type: str,
    payload: Any,
) -> UUID:
    canonical_payload = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return uuid5(
        NAMESPACE_URL,
        ":".join(
            (
                "langgraph-v2",
                tenant_id,
                conversation_id,
                scope_kind,
                str(scope_id),
                "retrieval",
                artifact_type,
                canonical_payload,
            )
        ),
    )


def upgrade() -> None:
    """Backfill every Artifact to one uniquely identifiable Turn."""
    op.execute(
        """
        ALTER TABLE langgraph_v2.artifacts
            ADD COLUMN conversation_id TEXT,
            ADD COLUMN turn_id UUID,
            ADD COLUMN turn_role TEXT
                GENERATED ALWAYS AS ('user'::text) STORED
        """
    )
    connection = op.get_bind()
    artifacts = connection.execute(
        text(
            """SELECT tenant_id, artifact_id, artifact_type, payload
            FROM langgraph_v2.artifacts"""
        )
    ).mappings()
    candidates = list(
        connection.execute(
            text(
                """SELECT messages.tenant_id, messages.conversation_id,
                          messages.turn_id, runs.run_id
                FROM langgraph_v2.messages AS messages
                LEFT JOIN langgraph_v2.runs AS runs
                  ON runs.tenant_id = messages.tenant_id
                 AND runs.conversation_id = messages.conversation_id
                 AND runs.turn_id = messages.turn_id
                WHERE messages.role = 'user'"""
            )
        ).mappings()
    )
    for artifact in artifacts:
        matches: set[tuple[str, UUID]] = set()
        for candidate in candidates:
            if candidate["tenant_id"] != artifact["tenant_id"]:
                continue
            generated_ids = {
                _legacy_artifact_id(
                    tenant_id=candidate["tenant_id"],
                    conversation_id=candidate["conversation_id"],
                    scope_kind="turn",
                    scope_id=candidate["turn_id"],
                    artifact_type=artifact["artifact_type"],
                    payload=artifact["payload"],
                )
            }
            if candidate["run_id"] is not None:
                generated_ids.add(
                    _legacy_artifact_id(
                        tenant_id=candidate["tenant_id"],
                        conversation_id=candidate["conversation_id"],
                        scope_kind="run",
                        scope_id=candidate["run_id"],
                        artifact_type=artifact["artifact_type"],
                        payload=artifact["payload"],
                    )
                )
            if artifact["artifact_id"] in generated_ids:
                matches.add((candidate["conversation_id"], candidate["turn_id"]))
        if len(matches) != 1:
            raise RuntimeError(
                f"Artifact {artifact['artifact_id']} has {len(matches)} provenance matches"
            )
        conversation_id, turn_id = matches.pop()
        connection.execute(
            text(
                """UPDATE langgraph_v2.artifacts
                SET conversation_id = :conversation_id, turn_id = :turn_id
                WHERE tenant_id = :tenant_id AND artifact_id = :artifact_id"""
            ),
            {
                "tenant_id": artifact["tenant_id"],
                "artifact_id": artifact["artifact_id"],
                "conversation_id": conversation_id,
                "turn_id": turn_id,
            },
        )
    op.execute(
        """
        ALTER TABLE langgraph_v2.artifacts
            ALTER COLUMN conversation_id SET NOT NULL,
            ALTER COLUMN turn_id SET NOT NULL,
            ADD CONSTRAINT artifacts_turn_fk
            FOREIGN KEY (tenant_id, conversation_id, turn_id, turn_role)
            REFERENCES langgraph_v2.messages
                (tenant_id, conversation_id, turn_id, role)
            ON DELETE CASCADE
        """
    )
    op.execute(
        """CREATE INDEX artifacts_turn_scope_idx
        ON langgraph_v2.artifacts
        (tenant_id, conversation_id, turn_id, created_at, artifact_id)"""
    )


def downgrade() -> None:
    """Remove Turn provenance while preserving Artifact data."""
    op.execute("DROP INDEX langgraph_v2.artifacts_turn_scope_idx")
    op.execute(
        """
        ALTER TABLE langgraph_v2.artifacts
            DROP CONSTRAINT artifacts_turn_fk,
            DROP COLUMN turn_role,
            DROP COLUMN turn_id,
            DROP COLUMN conversation_id
        """
    )
