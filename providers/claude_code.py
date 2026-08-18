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
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT = 600
DEFAULT_MAX_TURNS = 15
CLI = "claude"


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
    raw: Dict[str, Any] = field(default_factory=dict)


def available() -> bool:
    """Whether the CLI is installed. Checked before every run, not at import:
    the orchestrator must stay usable for text-only tasks on a machine that has
    no Claude Code."""
    return shutil.which(CLI) is not None


def run(
    prompt: str,
    cwd: Optional[str] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout: int = DEFAULT_TIMEOUT,
    allowed_tools: Optional[List[str]] = None,
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

    return CodeResult(
        ok=not body.get("is_error", False),
        result=str(body.get("result", "")),
        cost_usd=float(body.get("total_cost_usd", 0.0) or 0.0),
        num_turns=int(body.get("num_turns", 0) or 0),
        error="" if not body.get("is_error") else str(body.get("result", ""))[:2000],
        raw=body,
    )
