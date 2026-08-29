"""Trusted request identity at the API Gateway boundary."""

from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field, field_validator

UNASSIGNED_SUBJECT_ID = "__unassigned__"


class TrustedRequestContext(BaseModel):
    """Tenant and Subject authenticated and overwritten by the API Gateway.

    The deployment invariant is that the API Gateway authenticates the caller
    and overwrites both headers before forwarding a request to this service.
    """

    tenant_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)

    @field_validator("subject_id")
    @classmethod
    def reject_unassigned_subject(cls, value: str) -> str:
        """Keep migrated, unassigned Conversations outside normal auth."""
        if value == UNASSIGNED_SUBJECT_ID:
            raise ValueError("reserved subject identity")
        return value


async def get_trusted_request_context(
    x_application_id: Annotated[str, Header(alias="X-Application-Id", min_length=1)],
    x_subject_id: Annotated[str, Header(alias="X-Subject-Id", min_length=1)],
) -> TrustedRequestContext:
    """Build trusted identity from headers installed by the API Gateway."""
    if x_subject_id == UNASSIGNED_SUBJECT_ID:
        raise HTTPException(status_code=422, detail="reserved subject identity")
    return TrustedRequestContext(
        tenant_id=x_application_id,
        subject_id=x_subject_id,
    )
