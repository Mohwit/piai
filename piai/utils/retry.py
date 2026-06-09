"""
piai/utils/retry.py — RetryPolicy execution with exponential backoff.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, TypeVar

from piai.types import RetryPolicy

T = TypeVar("T")

# Status codes that are retryable by default
_RETRYABLE_STATUS = frozenset([429, 500, 502, 503, 504])
# Status codes that are never retried
_PERMANENT_STATUS = frozenset([400, 401, 403, 404])


def default_is_retryable(exc: Exception) -> bool:
    """
    Default predicate: retry on common transient HTTP errors.
    Checks for status_code or status attributes (httpx, requests, openai, anthropic).
    """
    for attr in ("status_code", "status"):
        code = getattr(exc, attr, None)
        if code is not None:
            return int(code) in _RETRYABLE_STATUS
    # Retry on generic network errors
    msg = str(exc).lower()
    if any(kw in msg for kw in ("timeout", "connection", "network", "reset")):
        return True
    return False


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: Optional[RetryPolicy] = None,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
) -> T:
    """
    Execute fn() with retry according to policy.

    Args:
        fn:           Async callable to retry.
        policy:       RetryPolicy; if None, no retry is performed.
        is_retryable: Predicate — returns True if the exception should trigger retry.
                      Defaults to default_is_retryable.
    """
    if policy is None:
        return await fn()

    predicate = is_retryable or default_is_retryable
    delay = policy.delay_ms / 1000.0
    last_exc: Optional[Exception] = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt == policy.max_retries or not predicate(exc):
                raise
            await asyncio.sleep(delay)
            delay *= policy.backoff_multiplier

    raise last_exc  # type: ignore[misc]
