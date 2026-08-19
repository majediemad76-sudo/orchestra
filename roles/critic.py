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

from providers import ProviderResult, google
from providers.schema_utils import json_hint
from schemas import Criterion, CriterionCheck, CriticVerdict, ManagerPlan, WorkerOutput

from . import load_prompt

MODEL = google.DEFAULT_MODEL


def _system() -> str:
    return (
        load_prompt("critic.md")
        + "\n\nCriticVerdict fields:\n"
        + json_hint(CriticVerdict)
    )


def build_user_message(plan: ManagerPlan, output: WorkerOutput) -> str:
    criteria = "\n".join(
        f"{index}. {c.text}\n   critical: {'yes' if c.critical else 'no'}"
        f"\n   check_method: {c.check_method}"
        for index, c in enumerate(plan.acceptance_criteria, start=1)
    )
    return (
        f"دستور داده‌شده به عامل مجری:\n{plan.worker_prompt}"
        f"\n\n---\n\nمعیارهای پذیرش:\n{criteria}"
        f"\n\n---\n\nخروجی عامل مجری:\n{output.result[:12000]}"
        + (f"\n\nیادداشت عامل مجری:\n{output.notes}" if output.notes else "")
    )


def review(plan: ManagerPlan, output: WorkerOutput) -> tuple[CriticVerdict, ProviderResult]:
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
    verdict = CriticVerdict.model_validate(data)
    return _reassert_criticality(verdict, plan), result


def _reassert_criticality(verdict: CriticVerdict, plan: ManagerPlan) -> CriticVerdict:
    """Restore each check's ``critical`` flag from the Manager's plan.

    The Critic is asked to echo the flag so that it judges with the stakes in
    view, but its echo is never trusted: criticality decides whether a round is
    accepted, and a judge that can downgrade a criterion it just failed is a
    judge marking its own homework.

    Matching is by exact criterion text, then by position -- the Critic is told
    to copy the text verbatim and to keep the order, so the two agree in
    practice; the positional fallback covers a model that paraphrased. A check
    that matches neither keeps whatever it was given, which is the
    conservative outcome only when it was already non-critical, so it is
    forced to critical when the plan has no room for an extra criterion.
    """
    by_text = {criterion.text.strip(): criterion.critical for criterion in plan.acceptance_criteria}
    for index, check in enumerate(verdict.checks):
        text = check.criterion.strip()
        if text in by_text:
            check.critical = by_text[text]
        elif index < len(plan.acceptance_criteria):
            check.critical = plan.acceptance_criteria[index].critical
        else:
            check.critical = True
    return verdict


def failed_worker_verdict(
    reason: str, criteria: list[Criterion] | None = None
) -> CriticVerdict:
    """Synthesise a rejection for a Worker that produced nothing.

    A timeout or a dead CLI is a fact about the round, not an error condition.
    Marking every criterion failed lets it flow through the same path as a bad
    answer, so the two-rejection rule counts it and the Manager gets told why.

    The evidence field is empty here and that is honest: there is no output to
    quote. No Critic call is made either -- there is nothing to grade, and the
    run should not pay to be told so.
    """
    criteria = criteria or [
        Criterion(
            text="worker produced usable output",
            critical=True,
            check_method="the run log records an output",
        )
    ]
    return CriticVerdict(
        checks=[
            CriterionCheck(
                criterion=criterion.text,
                passed=False,
                # Every criterion is blocking here: there is no output at all,
                # so this can never be waved through as a cosmetic miss.
                critical=True,
                evidence="",
                reason=f"The worker produced no output to check. {reason}".strip(),
            )
            for criterion in criteria
        ],
        fix_instruction=(
            "Break the task into smaller steps and make the instruction shorter and "
            f"more specific. The previous round failed because: {reason}"
        ),
        verdict="revise",
    )
