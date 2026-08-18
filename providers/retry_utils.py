"""One retry policy for all three vendors.

The split that matters is not "did it fail" but "will waiting help":

  * 429, 5xx, and transport errors are the provider's problem and are usually
    transient -- back off and try again.
  * 400 and 401 are our problem. A malformed request or a wrong key returns the
    identical error three times, three times slower, and buries the real cause
    under a retry trace. These fail immediately.

Three attempts with exponential backoff, capped at 20s: past that the run is
better off surfacing the failure to the Controller, which has a budget to
protect and a user it can ask.
"""

from __future__ import annotations

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

MAX_ATTEMPTS = 3
RETRYABLE_STATUS = {408, 409, 429}


class ProviderError(RuntimeError):
    """A provider failure carrying the status code the retry policy needs.

    ``status`` stays ``None`` for local failures (a missing key, an unparseable
    body) -- which is what makes them non-retryable by construction rather than
    by an explicit rule.
    """

    def __init__(self, provider: str, message: str, status: int | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status
        self.message = message


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code in RETRYABLE_STATUS or code >= 500
    if isinstance(exc, ProviderError):
        return exc.status is not None and (
            exc.status in RETRYABLE_STATUS or exc.status >= 500
        )
    # Timeouts, connection resets, DNS failures: the request may never have
    # reached the provider, so there is nothing to be idempotent about.
    return isinstance(exc, (httpx.TransportError,))


def _reraise(state: RetryCallState):
    if state.outcome is not None:
        state.outcome.result()


with_retry = retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry_error_callback=_reraise,
    reraise=True,
)


def raise_for_status(provider: str, response: httpx.Response) -> None:
    """Convert a non-2xx response into a ``ProviderError``.

    The body is truncated but never dropped: vendor error messages are where
    the actual cause lives ("schema is invalid", "credit balance too low"), and
    a bare status code turns a two-minute fix into an afternoon.
    """
    if response.is_success:
        return
    body = response.text[:800]
    raise ProviderError(
        provider,
        f"HTTP {response.status_code}: {body}",
        status=response.status_code,
    )
