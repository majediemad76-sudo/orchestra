"""Worker -- the only role that touches the actual task.

Two backends, chosen by the Manager's ``worker_type``:

  * ``text`` -> claude-sonnet-5 over the Anthropic API
  * ``code`` -> Claude Code headless in a subprocess

They are wildly different underneath -- one HTTP call versus a multi-turn agent
with filesystem access -- and converge on one ``WorkerRun`` so that everything
downstream stays backend-agnostic. Cost is the one place the difference leaks
through, hence the either/or below.
"""

from __future__ import annotations

from dataclasses import dataclass

from keys import ApiKeys
from providers import anthropic, claude_code
from schemas import ManagerPlan, WorkerOutput

TEXT_MODEL = anthropic.DEFAULT_MODEL
CODE_MODEL = "claude-code-headless"

SYSTEM = (
    "تو عامل مجری هستی. دستور زیر خودبسنده است؛ تاریخچه‌ی گفتگو را نداری. "
    "دقیقاً همان چیزی را تولید کن که خواسته شده، بدون توضیح اضافه درباره‌ی کارت."
)


@dataclass
class WorkerRun:
    """What the Worker produced and what it cost.

    ``cost_usd`` set means the backend reported dollars directly and token
    counts are meaningless; ``None`` means charge by tokens. Exactly one of the
    two paths applies, which is why the field is Optional rather than 0.0 --
    zero is a plausible cost and would be charged silently.
    """

    output: WorkerOutput
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # dollars when the backend reports them
    failure_reason: str = ""


def execute(
    plan: ManagerPlan,
    cwd: str | None = None,
    *,
    keys: ApiKeys,
    allow_code_worker: bool = False,
) -> WorkerRun:
    """Run the plan on whichever backend it asked for, if that backend is allowed.

    ``keys`` is required even for the code path, which does not use it: the
    Manager picks the backend at runtime, so a caller cannot know in advance
    which one will need a credential. Making it conditional would move that
    failure from the call site to the middle of a paid round.

    ``allow_code_worker`` defaults to *off*, and the default is the point. The
    code backend writes files and runs commands on the host; the Manager, not
    the caller, decides when to reach for it. A caller that has not said yes --
    an HTTP request from someone who only asked for text -- must not be able to
    get there by phrasing a goal that sounds like a coding task.

    Refusal is a rejection, not an exception: ``ok=False`` with a reason, which
    ``critic.failed_worker_verdict`` turns into a normal rejected round. The
    Manager sees why and can re-plan the same task as text, which is a better
    outcome than a traceback and a lost run.
    """
    if plan.worker_type == "code":
        if not allow_code_worker:
            return WorkerRun(
                output=WorkerOutput(result="", ok=False),
                model=CODE_MODEL,
                failure_reason=(
                    "the code worker is disabled for this caller; it can read and write "
                    "files on the host. Re-plan this task for the text worker: produce "
                    "the answer, or the file contents, directly in the response instead "
                    "of editing anything."
                ),
            )
        return _execute_code(plan, cwd=cwd)
    return _execute_text(plan, keys=keys)


def _execute_text(plan: ManagerPlan, *, keys: ApiKeys) -> WorkerRun:
    result = anthropic.call_text(
        system=SYSTEM,
        user=plan.worker_prompt,
        api_key=keys.require("anthropic"),
        model=TEXT_MODEL,
    )
    text = result.data.get("text", "").strip()
    return WorkerRun(
        output=WorkerOutput(result=text, ok=bool(text)),
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        failure_reason="" if text else "empty response from the text worker",
    )


def _execute_code(plan: ManagerPlan, cwd: str | None = None) -> WorkerRun:
    run = claude_code.run(plan.worker_prompt, cwd=cwd)
    reason = run.error if not run.ok else ""
    return WorkerRun(
        output=WorkerOutput(
            result=run.result,
            notes=f"turns={run.num_turns}" + (f"; error={run.error}" if run.error else ""),
            ok=run.ok and bool(run.result),
        ),
        model=CODE_MODEL,
        cost_usd=run.cost_usd,
        failure_reason=reason or ("" if run.result else "code worker returned nothing"),
    )
