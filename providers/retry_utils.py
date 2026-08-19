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

import re

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

MAX_ATTEMPTS = 4
RETRYABLE_STATUS = {408, 409, 429}

# Never wait longer than this on a provider's say-so. A quota that needs more
# than a minute is a capacity problem, not a blip, and the caller has a budget
# and a user it can report to.
MAX_RETRY_WAIT = 75.0


class ProviderError(RuntimeError):
    """A provider failure carrying the status code the retry policy needs.

    ``status`` stays ``None`` for local failures (a missing key, an unparseable
    body) -- which is what makes them non-retryable by construction rather than
    by an explicit rule.

    ``retry_after`` carries the delay the provider itself asked for. Guessing
    with exponential backoff when the server has already said "30 seconds" is
    how a run burns its attempts inside a window that was never going to open.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        status: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status
        self.message = message
        self.retry_after = retry_after


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


_BACKOFF = wait_exponential(multiplier=2, min=1, max=30)


def wait_policy(state: RetryCallState) -> float:
    """Obey the provider's own retry delay; fall back to exponential backoff."""
    exc = state.outcome.exception() if state.outcome else None
    suggested = getattr(exc, "retry_after", None)
    if suggested:
        # A second of slack: waking up exactly on the boundary tends to land
        # on the wrong side of the provider's clock.
        return min(float(suggested) + 1.0, MAX_RETRY_WAIT)
    return _BACKOFF(state)


with_retry = retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_policy,
    retry_error_callback=_reraise,
    reraise=True,
)


# "Please retry in 30.6s", "retryDelay": "31s", and the like.
_DELAY_PATTERN = re.compile(
    r"retry(?:_?delay|\s+in)?[\"\s:]*([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE
)


def parse_retry_after(response: httpx.Response) -> float | None:
    """Read the delay a provider asked for, from the header or the body.

    The header is the standard place; Gemini puts it in the JSON body instead,
    and OpenAI-compatible endpoints vary. Reading both is cheaper than being
    surprised by whichever one this vendor uses today.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass  # HTTP-date form; fall through to the body
    match = _DELAY_PATTERN.search(response.text[:2000])
    return float(match.group(1)) if match else None


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
        retry_after=parse_retry_after(response),
    )
