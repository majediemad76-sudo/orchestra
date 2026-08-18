"""Critic -- gemini-3.1-flash-lite over the Google API.

A different vendor from the Worker, and this is load-bearing rather than
incidental. A model grading its own output is the worst available judge: the
same weights that produced the mistake rate it as fine. A sibling model from
the same family is barely better -- shared training data means shared blind
spots. Independence is what makes the score worth reading.

The Critic never edits the output. Letting it rewrite would collapse the roles:
the grade would then describe the Critic's own work, and the isolation above
would be lost. It diagnoses and prescribes; the Worker applies.

If the model here is ever changed, change it to a third vendor -- not to
whatever is cheapest that week.
"""

from __future__ import annotations

from typing import Tuple

from providers import ProviderResult, google
from providers.schema_utils import json_hint
from schemas import CriticVerdict, ManagerPlan, WorkerOutput

from . import load_prompt

MODEL = google.DEFAULT_MODEL


def _system() -> str:
    return (
        load_prompt("critic.md")
        + "\n\nCriticVerdict fields:\n"
        + json_hint(CriticVerdict)
    )


def build_user_message(plan: ManagerPlan, output: WorkerOutput) -> str:
    criteria = "\n".join(f"- {c}" for c in plan.acceptance_criteria)
    return (
        f"دستور داده‌شده به عامل مجری:\n{plan.worker_prompt}"
        f"\n\n---\n\nمعیارهای پذیرش:\n{criteria}"
        f"\n\n---\n\nخروجی عامل مجری:\n{output.result[:12000]}"
        + (f"\n\nیادداشت عامل مجری:\n{output.notes}" if output.notes else "")
    )


def review(plan: ManagerPlan, output: WorkerOutput) -> Tuple[CriticVerdict, ProviderResult]:
    """Grade one Worker output against the plan's acceptance criteria."""
    result = google.call_structured(
        CriticVerdict,
        system=_system(),
        user=build_user_message(plan, output),
        model=MODEL,
    )
    # Gemini expresses "this field is optional" as nullable, and then actually
    # sends null -- an empty fix_instruction arrives as None rather than "".
    # Dropping nulls hands the field back to its Pydantic default, which is
    # what nullable was standing in for. The schema stays as-is: the vendor is
    # within its rights here, and narrowing the schema would be the wrong fix.
    data = {key: value for key, value in result.data.items() if value is not None}
    return CriticVerdict.model_validate(data), result


def failed_worker_verdict(reason: str) -> CriticVerdict:
    """Synthesise a rejection for a Worker that produced nothing.

    A timeout or a dead CLI is a fact about the round, not an error condition:
    scoring it 0 lets it flow through the same path as a bad answer, so the
    two-rejection rule counts it and the Manager gets told why.

    No Critic call is made -- there is nothing to grade, and the run should not
    pay to be told so.
    """
    return CriticVerdict(
        score=0,
        met_criteria=[],
        failed_criteria=["worker produced no usable output"],
        fix_instruction=(
            "کار را به گام‌های کوچک‌تر بشکن و دستور را کوتاه‌تر و مشخص‌تر کن. "
            f"علت شکست دور قبل: {reason}"
        ),
        verdict="revise",
    )
