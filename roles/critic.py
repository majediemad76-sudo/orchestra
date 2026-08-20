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

from typing import Any

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
    data = _repair(drop_nulls(result.data))
    verdict = CriticVerdict.model_validate(data)
    return _reassert_criticality(verdict, plan), result


def _repair(data: Any) -> Any:
    """Make a stripped payload validatable again, without inventing judgements.

    Dropping nulls is only half the job. ``critical`` and ``fix_instruction``
    have defaults, so losing them is harmless -- but ``reason`` and
    ``evidence`` do not, and stripping a null there turns one vendor quirk into
    a different validation error. The whole verdict, already paid for, is lost
    either way.

    So: descriptive fields are backfilled with an empty string, because their
    absence changes nothing about the verdict. The two fields that carry the
    actual judgement, ``criterion`` and ``passed``, are never invented -- a
    check missing either of them cannot be graded, so it is dropped, and the
    empty-checks rule in CriticVerdict.accepted then fails closed on its own.
    """
    if not isinstance(data, dict):
        return data
    checks = data.get("checks")
    if not isinstance(checks, list):
        return data
    repaired = []
    for check in checks:
        if not isinstance(check, dict) or "criterion" not in check or "passed" not in check:
            continue
        repaired.append({"evidence": "", "reason": "", **check})
    return {**data, "checks": repaired}


def drop_nulls(value: Any) -> Any:
    """Strip nulls at every depth, so a Pydantic default takes over instead.

    Gemini expresses "this field is optional" as nullable, and then actually
    sends null -- an empty fix_instruction arrives as None rather than "".
    Handing the field back to its default is what nullable was standing in for.

    Stripping only the top level was not enough, and the gap cost a fixture in
    a live run: one element of ``checks`` arrived with ``critical: null``, which
    is nested two levels down, and the whole verdict failed validation. The
    vendor is within its rights; narrowing the schema would be the wrong fix,
    and so is patching one field at a time as each one bites.
    """
    if isinstance(value, dict):
        return {k: drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nulls(item) for item in value if item is not None]
    return value


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
