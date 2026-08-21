"""Claude Code headless -- the code Worker.

A subprocess rather than an HTTP call, because this Worker's value is precisely
what an API call cannot do: read and write files, run commands, and iterate in
a real working directory.

The governing decision here is that *no failure of this worker is fatal*. A
timeout, a missing CLI, a non-zero exit, unparseable stdout -- each returns a
``CodeResult`` with ``ok=False`` and a reason. The Controller translates that
into a Critic-style rejection and the loop revises and retries. A crash would
throw away the run's accumulated context and its budget along with it; a
rejection is information the Manager can use.

The ``--max-turns`` cap is the second half of that bargain: the timeout bounds
wall-clock, the turn cap bounds spend, and neither is left to the model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT = 600
DEFAULT_MAX_TURNS = 15
CLI = "claude"

# Everything the child is allowed to inherit, by name. An allowlist rather than
# a denylist because the failure modes are not symmetric: forgetting to allow a
# variable makes the CLI misbehave visibly and gets fixed, while forgetting to
# deny one hands a credential to a subprocess and nobody finds out.
#
# The three provider key variables are absent and must stay absent. Two reasons,
# and the second is the one that is easy to miss:
#
#   1. This subprocess has no use for them. It authenticates through the CLI's
#      own login under HOME.
#   2. ANTHROPIC_API_KEY in particular would silently switch the CLI from the
#      user's subscription to API-key billing -- changing what the run costs and
#      which cost_basis the log should have claimed.
INHERITED_ENV = (
    "PATH",       # find the CLI and whatever it shells out to
    "HOME",       # ~/.claude: login, settings, MCP config
    "USER",
    "LOGNAME",
    "SHELL",      # the CLI runs commands through it
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TERM",
    "SSL_CERT_FILE",   # custom CA bundles, where a proxy demands one
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
    "__CF_USER_TEXT_ENCODING",  # macOS; its absence makes Python/node warn
)

# Named so the check in self_check and the reader agree on what must never be
# forwarded, without either restating the list.
FORBIDDEN_ENV = ("XAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")


def child_env() -> dict[str, str]:
    """The environment the subprocess gets: an allowlist, not the parent's.

    ``subprocess.run`` with no ``env`` hands the child everything this process
    holds, which now includes credentials belonging to whichever HTTP caller
    happens to be running. Those must not reach a program that can write files
    and run commands.
    """
    env = {name: os.environ[name] for name in INHERITED_ENV if name in os.environ}
    # Belt and braces. If a key name is ever added to INHERITED_ENV by mistake,
    # this drops it rather than trusting the review that let it through.
    for name in FORBIDDEN_ENV:
        env.pop(name, None)
    return env


@dataclass
class CodeResult:
    """Outcome of one headless run -- success and failure share this shape.

    ``timed_out`` is kept separate from ``error`` because it is the one failure
    the Manager can act on directly: the task was too big for one session, so
    the next round should decompose it further.
    """

    ok: bool
    result: str
    cost_usd: float = 0.0
    num_turns: int = 0
    error: str = ""
    timed_out: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def available() -> bool:
    """Whether the CLI is installed. Checked before every run, not at import:
    the orchestrator must stay usable for text-only tasks on a machine that has
    no Claude Code."""
    return shutil.which(CLI) is not None


def run(
    prompt: str,
    cwd: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout: int = DEFAULT_TIMEOUT,
    allowed_tools: list[str] | None = None,
) -> CodeResult:
    """Run one headless session and parse its JSON envelope.

    Never raises. Every failure path returns a ``CodeResult`` the loop can
    treat as a rejection.
    """
    if not available():
        return CodeResult(
            ok=False,
            result="",
            error=f"'{CLI}' CLI not found on PATH; cannot run a code worker.",
        )

    cmd = [CLI, "-p", prompt, "--output-format", "json", "--max-turns", str(max_turns)]
    if allowed_tools:
        cmd += ["--allowedTools", ",".join(allowed_tools)]

    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env(),
        )
    except subprocess.TimeoutExpired:
        # The expensive failure, and the one worth naming precisely: work was
        # done and paid for, we just cannot see the result. The cost is
        # unknowable here (the envelope never arrived), so it goes uncharged --
        # the round cap is what bounds a pathological repeat.
        return CodeResult(
            ok=False,
            result="",
            error=f"claude code headless run exceeded {timeout}s",
            timed_out=True,
        )
    except OSError as exc:
        return CodeResult(ok=False, result="", error=f"could not start {CLI}: {exc}")

    if completed.returncode != 0:
        return CodeResult(
            ok=False,
            result=completed.stdout.strip(),
            error=(completed.stderr or "").strip()[:2000]
            or f"{CLI} exited with code {completed.returncode}",
        )

    try:
        body = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        # The CLI's output contract changed, or a wrapper polluted stdout.
        # The text is still probably the answer -- let the Critic judge it
        # rather than discard a completed run over its packaging.
        return CodeResult(
            ok=True,
            result=(completed.stdout or "").strip(),
            error="stdout was not the expected JSON envelope",
        )

    # What `total_cost_usd` actually is, because the name invites a wrong read:
    # it is what this session WOULD have cost at API rates. On a personal Claude
    # subscription (Pro/Max) the CLI runs against the plan's included usage --
    # nothing here is drawn from API credit and no invoice line appears for it.
    # So it is a real measure of work done and a real input to the budget
    # ceiling, but it is not money billed. Runs that mix this worker with the
    # API roles therefore add two different kinds of spend, which is why every
    # charge is tagged with a cost_basis in the run log.
    # (On an API-key-authenticated CLI the same figure IS billed usage.)
    return CodeResult(
        ok=not body.get("is_error", False),
        result=str(body.get("result", "")),
        cost_usd=float(body.get("total_cost_usd", 0.0) or 0.0),
        num_turns=int(body.get("num_turns", 0) or 0),
        error="" if not body.get("is_error") else str(body.get("result", ""))[:2000],
        raw=body,
    )
