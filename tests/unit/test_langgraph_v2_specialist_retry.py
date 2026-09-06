"""Closed PydanticAI V1 retry-policy coverage for Specialists."""

from __future__ import annotations

import httpx
import openai
import pytest
from pydantic_ai.exceptions import (
    ContentFilterError,
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.usage import RunUsage, UsageLimits

from app.langgraph_v2.specialist_retry import (
    PinnedPydanticAIVersionError,
    RetryDisposition,
    SpecialistFailureFacts,
    SpecialistModelBoundary,
    SpecialistModelRequestTimeout,
    classify_specialist_failure,
    require_pinned_pydantic_ai_version,
    specialist_usage_limits,
)


def _facts(
    *,
    boundary: SpecialistModelBoundary = SpecialistModelBoundary.AZURE_OPENAI,
    at_model_request_boundary: bool = True,
    terminal_output_tool_rejected: bool = False,
    count_limit_exhausted: bool = True,
    usage_limits: UsageLimits | None = None,
) -> SpecialistFailureFacts:
    return SpecialistFailureFacts(
        boundary=boundary,
        at_model_request_boundary=at_model_request_boundary,
        terminal_output_tool_rejected=terminal_output_tool_rejected,
        count_limit_exhausted=count_limit_exhausted,
        unreturned_model_requests=0,
        usage_limits=usage_limits
        or UsageLimits(
            request_limit=12,
            tool_calls_limit=8,
            input_tokens_limit=None,
            output_tokens_limit=None,
            total_tokens_limit=None,
        ),
    )


@pytest.mark.parametrize("status", [429, 500, 599])
def test_closed_v1_classifier_retries_only_allowlisted_http_statuses(
    status: int,
) -> None:
    assert (
        classify_specialist_failure(
            ModelHTTPError(status, "specialist"), facts=_facts()
        )
        is RetryDisposition.RETRY
    )


@pytest.mark.parametrize("status", [401, 408, 600])
def test_closed_v1_classifier_rejects_other_http_statuses(status: int) -> None:
    assert (
        classify_specialist_failure(
            ModelHTTPError(status, "specialist"), facts=_facts()
        )
        is None
    )


def test_closed_v1_classifier_proves_provider_and_phase_before_retrying() -> None:
    request = httpx.Request("GET", "https://example.test")
    azure_connection = ModelAPIError("specialist", "connection")
    azure_connection.__cause__ = openai.APIConnectionError(request=request)

    assert (
        classify_specialist_failure(azure_connection, facts=_facts())
        is RetryDisposition.RETRY
    )
    assert (
        classify_specialist_failure(
            azure_connection,
            facts=_facts(boundary=SpecialistModelBoundary.GOOGLE),
        )
        is None
    )
    assert (
        classify_specialist_failure(
            httpx.ConnectError("down", request=request),
            facts=_facts(boundary=SpecialistModelBoundary.GOOGLE),
        )
        is RetryDisposition.RETRY
    )
    assert (
        classify_specialist_failure(
            httpx.TimeoutException("down", request=request),
            facts=_facts(boundary=SpecialistModelBoundary.GOOGLE),
        )
        is RetryDisposition.RETRY
    )
    assert (
        classify_specialist_failure(
            httpx.TimeoutException("down", request=request),
            facts=_facts(boundary=SpecialistModelBoundary.AZURE_OPENAI),
        )
        is None
    )
    assert (
        classify_specialist_failure(
            httpx.ConnectError("down", request=request),
            facts=_facts(at_model_request_boundary=False),
        )
        is None
    )
    timeout_connection = ModelAPIError("specialist", "timeout")
    timeout_connection.__cause__ = openai.APITimeoutError(request=request)
    assert (
        classify_specialist_failure(timeout_connection, facts=_facts())
        is RetryDisposition.RETRY
    )
    assert (
        classify_specialist_failure(
            SpecialistModelRequestTimeout(),
            facts=_facts(at_model_request_boundary=False),
        )
        is None
    )


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.TimeoutException])
@pytest.mark.parametrize(
    "boundary",
    [SpecialistModelBoundary.AZURE_OPENAI, SpecialistModelBoundary.UNKNOWN],
)
def test_raw_google_transport_errors_are_fatal_outside_the_google_model_boundary(
    error_type: type[httpx.TransportError],
    boundary: SpecialistModelBoundary,
) -> None:
    error = error_type("business transport failure", request=httpx.Request("GET", "https://example.test"))

    assert classify_specialist_failure(error, facts=_facts(boundary=boundary)) is None
    assert (
        classify_specialist_failure(
            error,
            facts=_facts(
                boundary=SpecialistModelBoundary.GOOGLE,
                at_model_request_boundary=False,
            ),
        )
        is None
    )


@pytest.mark.parametrize(
    ("error", "facts", "expected"),
    [
        (
            IncompleteToolCall("truncated"),
            _facts(terminal_output_tool_rejected=True),
            RetryDisposition.RETRY,
        ),
        (
            IncompleteToolCall("business tool"),
            _facts(),
            None,
        ),
        (
            UnexpectedModelBehavior("output rejected"),
            _facts(terminal_output_tool_rejected=True),
            RetryDisposition.RETRY,
        ),
        (
            ContentFilterError("blocked"),
            _facts(terminal_output_tool_rejected=True),
            None,
        ),
        (UsageLimitExceeded("requests"), _facts(), RetryDisposition.TASK_FAILED),
        (
            UsageLimitExceeded("tokens"),
            _facts(
                usage_limits=UsageLimits(
                    request_limit=12,
                    tool_calls_limit=8,
                    input_tokens_limit=1,
                )
            ),
            None,
        ),
        (
            UsageLimitExceeded("business tool"),
            _facts(count_limit_exhausted=False),
            None,
        ),
        (ModelAPIError("specialist", "untyped"), _facts(), None),
        (RuntimeError("unknown"), _facts(), None),
    ],
)
def test_closed_v1_classifier_has_no_superclass_or_message_fallback(
    error: Exception,
    facts: SpecialistFailureFacts,
    expected: RetryDisposition | None,
) -> None:
    assert classify_specialist_failure(error, facts=facts) is expected


def test_pinned_pydantic_ai_version_is_required() -> None:
    require_pinned_pydantic_ai_version()

    with pytest.raises(PinnedPydanticAIVersionError):
        require_pinned_pydantic_ai_version(version="1.93.1")


def test_specialist_count_limits_allow_exact_bound_and_reject_one_more() -> None:
    limits = specialist_usage_limits()

    assert limits.request_limit == 12
    assert limits.tool_calls_limit == 8
    assert not limits.has_token_limits()
    limits.check_before_request(RunUsage(requests=11))
    with pytest.raises(UsageLimitExceeded):
        limits.check_before_request(RunUsage(requests=12))
    limits.check_before_tool_call(RunUsage(tool_calls=8))
    with pytest.raises(UsageLimitExceeded):
        limits.check_before_tool_call(RunUsage(tool_calls=9))
