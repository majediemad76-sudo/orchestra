"""API credentials as an explicit argument, never an ambient one.

Reading ``os.environ`` inside a provider is convenient exactly once: while the
process serves a single user whose keys live in ``.env``. The moment a second
caller appears -- an HTTP request carrying its own credentials -- ambient
lookup stops being a shortcut and becomes a correctness bug, because there is
no way for two concurrent tasks to want different keys.

So the environment is read at the edges (``controller.main``, ``app.py``, the
scripts under ``scripts/``) and the value is threaded down as a parameter. A
provider that cannot reach the environment cannot accidentally use the wrong
caller's key.

The second job of this type is to be hostile to leaks. ``repr`` is overridden
rather than left to the dataclass default, because the default would print
every key the first time one of these lands in a traceback frame, a log line,
or a debugger. See ``providers.redact`` for the other half.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

# The provider names used as attribute names here and as ``ApiKeys`` field
# names. Kept in one place so a typo is an AttributeError at import time
# rather than a silent "" that turns into a 401 an hour later.
PROVIDERS = ("xai", "anthropic", "google")

ENV_VARS = {
    "xai": "XAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class MissingKey(RuntimeError):
    """A provider was called without the credential it needs.

    Raised before any network call, and phrased in terms of the variable the
    user has to set. It carries no value -- only the name.
    """

    def __init__(self, provider: str):
        self.provider = provider
        env_var = ENV_VARS.get(provider, provider.upper())
        super().__init__(f"no API key for {provider} (set {env_var}, see .env.example)")


@dataclass
class ApiKeys:
    """The three credentials, passed explicitly to whoever needs them.

    Not frozen: ``clear()`` needs to drop the values when a task finishes. That
    is a best-effort scrub -- Python strings are immutable, so what this really
    does is drop the last reference this object holds and let the garbage
    collector reclaim it. It stops the keys living as long as the task record
    does, which is the actual risk; it is not a guarantee about process memory,
    and nothing here should be described as one.
    """

    xai: str = ""
    anthropic: str = ""
    google: str = ""

    @classmethod
    def from_env(cls) -> ApiKeys:
        """Read the three variables. Only entry points may call this."""
        return cls(**{name: os.environ.get(var, "").strip() for name, var in ENV_VARS.items()})

    def require(self, provider: str) -> str:
        """The key for ``provider``, or ``MissingKey`` before anything is sent."""
        value = getattr(self, provider, "").strip()
        if not value:
            raise MissingKey(provider)
        return value

    def present(self) -> dict[str, bool]:
        """Which keys are set. Booleans only -- never the values, never a mask.

        A masked key is still a leak: the prefix identifies the vendor and the
        account, and the length narrows the rest. The UI has never needed more
        than yes/no.
        """
        return {name: bool(getattr(self, name, "").strip()) for name in PROVIDERS}

    def secrets(self) -> tuple[str, ...]:
        """Every non-empty value, for scrubbing text that may quote one back."""
        return tuple(v for v in (getattr(self, n, "") for n in PROVIDERS) if v)

    def clear(self) -> None:
        """Drop the values. Called when the task that supplied them is done."""
        for f in fields(self):
            setattr(self, f.name, "")

    def __repr__(self) -> str:
        state = ", ".join(f"{n}={'set' if v else 'missing'}" for n, v in self.present().items())
        return f"ApiKeys({state})"

    __str__ = __repr__


# A convenience for call sites that build one key at a time.
def only(provider: str, value: str) -> ApiKeys:
    """An ``ApiKeys`` carrying a single credential."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    return ApiKeys(**{provider: value})
