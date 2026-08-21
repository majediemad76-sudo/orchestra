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


def execute(plan: ManagerPlan, cwd: str | None = None, *, keys: ApiKeys) -> WorkerRun:
    """Run the plan on whichever backend it asked for.

    ``keys`` is required even for the code path, which does not use it: the
    Manager picks the backend at runtime, so a caller cannot know in advance
    which one will need a credential. Making it conditional would move that
    failure from the call site to the middle of a paid round.
    """
    if plan.worker_type == "code":
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
