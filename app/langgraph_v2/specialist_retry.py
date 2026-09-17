"""Closed retry policy for one PydanticAI V1 Specialist Task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import httpx
import openai
import pydantic_ai
from pydantic_ai.exceptions import (
    ContentFilterError,
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.usage import UsageLimits

PINNED_PYDANTIC_AI_VERSION = "1.93.0"
SPECIALIST_MAX_ATTEMPTS = 3
SPECIALIST_MODEL_REQUEST_LIMIT = 12
SPECIALIST_TOOL_CALL_LIMIT = 8


class PinnedPydanticAIVersionError(RuntimeError):
    """Reject a runtime whose SDK semantics differ from the reviewed mapping."""


def require_pinned_pydantic_ai_version(*, version: str | None = None) -> None:
    """Require the PydanticAI baseline that this closed table was reviewed for."""
    resolved = pydantic_ai.__version__ if version is None else version
    if resolved != PINNED_PYDANTIC_AI_VERSION:
        raise PinnedPydanticAIVersionError(
            "Specialist retry mapping requires "
            f"pydantic-ai=={PINNED_PYDANTIC_AI_VERSION}, got {resolved}"
        )


class SpecialistModelBoundary(StrEnum):
    """The only configured model-call provider boundaries in this POC."""

    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    UNKNOWN = "unknown"


class RetryDisposition(StrEnum):
    """The only non-fatal classifications owned by this adapter."""

    RETRY = "retry"
    TASK_FAILED = "task_failed"


class SpecialistModelRequestTimeout(TimeoutError):
    """Proves the Specialist adapter's own per-model-request deadline fired."""


@dataclass(frozen=True)
class SpecialistFailureFacts:
    """Typed invocation facts permitted by the closed V1 classifier."""

    boundary: SpecialistModelBoundary
    at_model_request_boundary: bool
    terminal_output_tool_rejected: bool
    count_limit_exhausted: bool
    unreturned_model_requests: int
    usage_limits: UsageLimits


class SpecialistInvocationFailure(Exception):
    """Preserve typed SDK-boundary facts without converting an unknown error."""

    def __init__(
        self,
        error: Exception,
        *,
        facts: SpecialistFailureFacts,
        messages: tuple[object, ...],
    ) -> None:
        self.error = error
        self.facts = facts
        self.messages = messages
        super().__init__(str(error))


@dataclass(frozen=True)
class SpecialistFailedAttemptDiagnostic:
    """Request-local failed-attempt message capture, never a Task Outcome field."""

    attempt: int
    messages: tuple[object, ...]


@dataclass
class SpecialistExecutionDiagnostics:
    """Ephemeral execution diagnostics owned by the invocation caller."""

    failed_attempts: list[SpecialistFailedAttemptDiagnostic]

    def __init__(self) -> None:
        self.failed_attempts = []

    def record_failed_attempt(
        self, *, attempt: int, messages: tuple[object, ...]
    ) -> None:
        """Retain one failed SDK invocation outside accepted graph state."""
        self.failed_attempts.append(
            SpecialistFailedAttemptDiagnostic(attempt=attempt, messages=messages)
        )


def specialist_usage_limits() -> UsageLimits:
    """Return the Task-cumulative count limits with every token limit disabled."""
    return UsageLimits(
        request_limit=SPECIALIST_MODEL_REQUEST_LIMIT,
        tool_calls_limit=SPECIALIST_TOOL_CALL_LIMIT,
        input_tokens_limit=None,
        output_tokens_limit=None,
        total_tokens_limit=None,
    )


def _count_limits_only(limits: UsageLimits) -> bool:
    return (
        limits.request_limit is not None
        and limits.tool_calls_limit is not None
        and not limits.has_token_limits()
    )


def classify_specialist_failure(
    error: Exception,
    *,
    facts: SpecialistFailureFacts,
) -> RetryDisposition | None:
    """Classify only the exact reviewed PydanticAI V1 failure allowlist."""
    if isinstance(error, ContentFilterError):
        return None
    if isinstance(error, SpecialistModelRequestTimeout):
        return (
            RetryDisposition.RETRY if facts.at_model_request_boundary else None
        )
    if isinstance(error, ModelHTTPError):
        if (
            facts.boundary is not SpecialistModelBoundary.UNKNOWN
            and facts.at_model_request_boundary
            and (error.status_code == 429 or 500 <= error.status_code <= 599)
        ):
            return RetryDisposition.RETRY
        return None
    if isinstance(error, ModelAPIError):
        if (
            facts.boundary is SpecialistModelBoundary.AZURE_OPENAI
            and facts.at_model_request_boundary
            and isinstance(error.__cause__, openai.APIConnectionError)
        ):
            return RetryDisposition.RETRY
        return None
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
        if (
            facts.boundary is SpecialistModelBoundary.GOOGLE
            and facts.at_model_request_boundary
        ):
            return RetryDisposition.RETRY
        return None
    if isinstance(error, IncompleteToolCall):
        return (
            RetryDisposition.TASK_FAILED
            if facts.terminal_output_tool_rejected
            else None
        )
    if isinstance(error, UnexpectedModelBehavior):
        return (
            RetryDisposition.TASK_FAILED
            if facts.terminal_output_tool_rejected
            else None
        )
    if isinstance(error, UsageLimitExceeded):
        return (
            RetryDisposition.TASK_FAILED
            if _count_limits_only(facts.usage_limits) and facts.count_limit_exhausted
            else None
        )
    return None
