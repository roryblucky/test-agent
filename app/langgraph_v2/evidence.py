"""Request-local retrieval evidence that is never persisted."""

from __future__ import annotations

from pydantic import BaseModel

from app.models.domain import Document


class Evidence(BaseModel):
    """One retrieved document with a stable citation identity."""

    evidence_id: str
    document: Document
