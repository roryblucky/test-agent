"""Application-specific exceptions."""

from __future__ import annotations

from app.models.domain import ModerationResult


class ContentFlaggedError(Exception):
    """Raised when content moderation detects a policy violation."""

    def __init__(self, result: ModerationResult) -> None:
        self.result = result
        super().__init__(
            f"Content flagged by moderation: {result.reason or result.categories}"
        )


class TenantNotFoundError(Exception):
    """Raised when the requested tenant / application ID is not found."""

    def __init__(self, application_id: str) -> None:
        self.application_id = application_id
        super().__init__(f"Tenant not found: {application_id}")


class AccessDeniedError(Exception):
    """Raised when the user's AD groups do not match the tenant's."""

    def __init__(self, application_id: str) -> None:
        self.application_id = application_id
        super().__init__(f"Access denied for tenant: {application_id}")
