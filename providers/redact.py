"""Strip credentials out of text before anyone can see it.

Two mechanisms, deliberately not one:

*Exact removal* is the guarantee. When we hold the key we can look for that
exact string and take it out, and no vendor's error format can defeat it.

*Pattern removal* is the backstop for the case exact removal cannot cover: a
key we were never given. A request built by hand in a test, a value pasted into
a prompt by mistake, a second credential inside a vendor's echo of our request.
The patterns match the shapes the three vendors issue.

Neither is applied on its own. Exact removal without the patterns misses keys
we do not hold; patterns without exact removal miss any key whose format
changes, and formats change without warning.

The output says ``[REDACTED]`` rather than dropping the text, because an error
message with a visible hole is debuggable and one that silently lost a span is
not.
"""

from __future__ import annotations

import re

PLACEHOLDER = "[REDACTED]"

# The issued shapes, as of the vendors we call. A prefix plus a run of key
# characters: long enough that ordinary prose cannot trip it, loose enough that
# a rotated format still matches.
_KEY_SHAPES = re.compile(
    r"""(
          sk-ant-[A-Za-z0-9\-_]{8,}      # Anthropic
        | xai-[A-Za-z0-9\-_]{8,}         # xAI
        | AIza[A-Za-z0-9\-_]{8,}         # Google, classic
        | AQ\.[A-Za-z0-9\-_]{8,}         # Google, newer project keys
        | sk-[A-Za-z0-9]{16,}            # OpenAI-compatible, incl. our canary shape
    )""",
    re.VERBOSE,
)

# Header names whose value is always a credential, whatever it looks like.
# Matched on a single line so a JSON or header dump cannot smuggle one past the
# shape patterns.
_AUTH_LINES = re.compile(
    r"""(?im)^(\s*["']?(?:authorization|x-api-key|x-goog-api-key|api[-_]?key)["']?\s*[:=]\s*)
        (["']?)([^"'\r\n,}]+)""",
    re.VERBOSE,
)

# The shortest string worth removing exactly. Below this, a "key" is more
# likely to be a placeholder like "" or "x" whose removal would blank out
# unrelated text -- redacting every "x" in a message helps nobody.
MIN_EXACT_LEN = 8


def redact(text: str, *secrets: str) -> str:
    """Return ``text`` with credentials replaced by ``[REDACTED]``.

    ``secrets`` are exact values to remove -- pass every key in play, even the
    ones this provider does not use, since a vendor echoing our request body
    can quote a credential we did not send it.
    """
    if not text:
        return text
    out = text
    for secret in secrets:
        secret = (secret or "").strip()
        if len(secret) >= MIN_EXACT_LEN:
            out = out.replace(secret, PLACEHOLDER)
    out = _AUTH_LINES.sub(lambda m: f"{m.group(1)}{m.group(2)}{PLACEHOLDER}", out)
    return _KEY_SHAPES.sub(PLACEHOLDER, out)


def redact_exc(exc: BaseException, *secrets: str) -> str:
    """A redacted one-line rendering of an exception.

    Exceptions are the leak path that is easiest to forget: ``str(exc)`` on an
    httpx error can carry the request line, and a bare ``repr`` of a wrapper
    object can carry whatever it holds. Anything that puts an exception in
    front of a human or into a log goes through here.
    """
    return redact(f"{type(exc).__name__}: {exc}", *secrets)
