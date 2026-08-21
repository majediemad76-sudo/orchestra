"""Manager -- grok-4.6 over the xAI API.

Decomposes the task and writes the Worker's instruction. Two things it never
does: execute the task, or decide how the loop proceeds. It may *report* that
it lacks information; whether that reaches the user is the Controller's call.

The Manager is the only role that sees history -- prior plan, prior output,
Critic feedback, the user's answer -- because it is the only role that needs
to. The Worker gets a self-contained prompt precisely so that the quality of
its output cannot depend on conversational context that a fresh session, or a
subprocess, would not have.
"""

from __future__ import annotations

import json

from keys import ApiKeys
from providers import ProviderResult, xai
from providers.schema_utils import json_hint
from schemas import CriticVerdict, ManagerPlan, Task

from . import load_prompt

MODEL = xai.DEFAULT_MODEL


def _system() -> str:
    # Appending the field list is redundant against strict mode, and worth it
    # anyway: the schema dictates shape, this dictates intent.
    return (
        load_prompt("manager.md")
        + "\n\nManagerPlan fields:\n"
        + json_hint(ManagerPlan)
    )


def build_user_message(
    task: Task,
    previous_plan: ManagerPlan | None = None,
    verdict: CriticVerdict | None = None,
    worker_result: str = "",
    user_answer: str = "",
) -> str:
    parts = [f"هدف کاربر:\n{task.goal}"]
    if task.context:
        parts.append(f"زمینه:\n{task.context}")
    if user_answer:
        parts.append(f"پاسخ کاربر به پرسش قبلی:\n{user_answer}")
    if previous_plan is not None:
        parts.append(f"دستور دور قبل به عامل مجری:\n{previous_plan.worker_prompt}")
        parts.append(
            "معیارهای پذیرش دور قبل:\n"
            + "\n".join(
                f"- [{'critical' if c.critical else 'optional'}] {c.text} ({c.check_method})"
                for c in previous_plan.acceptance_criteria
            )
        )
    if worker_result:
        parts.append(f"خروجی دور قبل عامل مجری:\n{worker_result[:4000]}")
    if verdict is not None:
        parts.append(
            "بازخورد Critic:\n"
            + json.dumps(verdict.model_dump(), ensure_ascii=False, indent=2)
        )
    return "\n\n---\n\n".join(parts)


def plan(
    task: Task,
    previous_plan: ManagerPlan | None = None,
    verdict: CriticVerdict | None = None,
    worker_result: str = "",
    user_answer: str = "",
    *,
    keys: ApiKeys,
) -> tuple[ManagerPlan, ProviderResult]:
    """Produce a plan for this round.

    Returns the validated plan alongside the raw call, because the Controller
    needs the token counts to charge the budget and the plan to act on -- and
    should not have to ask twice.
    """
    result = xai.call_structured(
        ManagerPlan,
        system=_system(),
        user=build_user_message(task, previous_plan, verdict, worker_result, user_answer),
        api_key=keys.require("xai"),
        model=MODEL,
    )
    parsed = ManagerPlan.model_validate(result.data)
    if parsed.needs_user_input and parsed.question is None:
        # The schema permits needs_user_input without a question; the
        # escalation rule does not. Rather than raise -- the plan is otherwise
        # usable and already paid for -- downgrade to "proceed", which costs a
        # round at worst and matches the instruction to guess where guessing is
        # safe.
        parsed.needs_user_input = False
    return parsed, result
