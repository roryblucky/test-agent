"""Resilience module — Retry policies and circuit breakers.

Provides standard retry configurations (exponential backoff) and safe execution
wrappers for external API calls (LLMs, databases, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.services.exceptions import ContentFlaggedError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# Standard retry policy for external dependencies
# Wait 1s, 2s, 4s... up to 10s. Stop after 3 attempts.
RETRY_POLICY = AsyncRetrying(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((OSError, asyncio.TimeoutError)),
    reraise=True,
)


async def safe_execute(
    func: Callable[P, Awaitable[R]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Execute an async function with standard retry policy.

    Retries on network/timeout errors.
    Does NOT retry on logical errors (e.g., ContentFlaggedError, ValueError).
    """
    try:
        async for attempt in RETRY_POLICY:
            with attempt:
                return await func(*args, **kwargs)
    except RetryError as e:
        logger.error(f"Operation failed after retries: {func.__name__}")
        raise e.last_attempt.result() if e.last_attempt else e
    except ContentFlaggedError:
        # Don't retry moderation failures
        raise
    except Exception as e:
        logger.error(f"Operation failed: {func.__name__} - {e}")
        raise
