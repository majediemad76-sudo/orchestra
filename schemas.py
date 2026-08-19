"""The contracts between roles.

These models do double duty: they validate what comes back from a model, and
they *are* the schemas sent to the three vendors (via
``providers/schema_utils.py``). One definition, so a field cannot drift between
what is asked for and what is accepted.

Constraints belong here rather than in prose. ``Question.options`` is bounded
2..4 in the type system, which is why "always give the user a closed choice"
cannot be quietly forgotten at a call site.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Evidence quotes are for auditing, not for reproducing the output.
EVIDENCE_MAX_CHARS = 200


class Question(BaseModel):
    """A closed question for the user.

    Open questions are banned by design. An escalation interrupts a human, so
    it has to be answerable in one keystroke; "what should I do?" hands the
    problem back untouched. Fewer than two options is not a question, and more
    than four is a menu nobody reads.
    """

    text: str = Field(description="The question to ask the user.")
    options: List[str] = Field(
        min_length=2,
        max_length=4,
        description="Between 2 and 4 concrete options the user can pick from.",
    )


class Task(BaseModel):
    """The goal, plus the four limits the Controller enforces.

    The limits live on the task rather than in module constants so a caller can
    say "this one is worth a dollar and six rounds" without touching code.
    """

    goal: str = Field(description="What the user wants.")
    context: str = Field(default="", description="Any extra context supplied by the user.")
    max_rounds: int = Field(default=4, ge=1, description="Hard cap on Manager/Worker/Critic rounds.")
    budget_usd: float = Field(default=0.50, gt=0, description="Dollar ceiling for the whole run.")


class ManagerPlan(BaseModel):
    """The Manager's decomposition and the Worker's marching orders.

    ``needs_user_input`` is the Manager's only channel to the user, and it is a
    request rather than an action: the Controller decides whether the question
    is actually asked.
    """

    plan: str = Field(description="Short decomposition of the task.")
    worker_prompt: str = Field(
        description="Self-contained instruction for the worker. The worker sees no history."
    )
    acceptance_criteria: List[Criterion] = Field(
        min_length=1,
        description=(
            "Objectively checkable criteria, not matters of taste. At least one must be "
            "critical, or nothing is actually being required of the output."
        ),
    )
    worker_type: Literal["text", "code"] = Field(
        description="'text' routes to the Anthropic API, 'code' routes to Claude Code headless."
    )
    needs_user_input: bool = Field(
        default=False,
        description="True only when required information is missing and guessing is risky.",
    )
    question: Optional[Question] = Field(
        default=None, description="Required when needs_user_input is true."
    )


class WorkerOutput(BaseModel):
    """What the Worker produced, from either backend.

    ``ok=False`` means no usable output -- a timeout, an empty response, a dead
    CLI. It is a rejection, never an exception; see ``critic.failed_worker_verdict``.
    """

    result: str = Field(description="The actual work product.")
    notes: str = Field(default="", description="Anything the worker wants the critic to know.")
    ok: bool = Field(default=True, description="False when the worker failed to produce a result.")


class Criterion(BaseModel):
    """One acceptance criterion, written to be checkable rather than admired.

    The Critic can only be as objective as the criteria it is handed. "The
    tone should be good" cannot be judged twice the same way; "fewer than 150
    words" can. Forcing the Manager to name a ``check_method`` is what makes
    that difference explicit at authoring time -- a criterion whose method the
    Manager cannot state is a criterion nobody can check.

    ``critical`` records how much a violation actually matters, which is a
    property of the task rather than of the output, so the Manager is the role
    that knows it.
    """

    text: str = Field(
        description=(
            "The criterion itself. Must be binary, objective and unambiguous: "
            "two readers checking the same output must reach the same answer."
        )
    )
    critical: bool = Field(
        description="Whether violating this criterion blocks acceptance of the output."
    )
    check_method: str = Field(
        description=(
            "How to verify it objectively -- e.g. 'count the words', "
            "'search for the substring', 'run the test'. Not a restatement of the text."
        )
    )


class CriterionCheck(BaseModel):
    """One acceptance criterion, judged on its own.

    A holistic 0-100 score asks the model to compress many judgements into one
    number, and models anchor those numbers high. A single criterion answered
    yes/no, with the quote that settles it, is a far more stable thing to ask
    for -- and it is auditable afterwards, which a bare score never is.

    ``evidence`` must be lifted from the output verbatim. Requiring a quote is
    what stops the judgement from drifting into taste: a criterion the Critic
    cannot point at is a criterion it did not really check.
    """

    criterion: str = Field(description="The criterion text, copied exactly as the Manager wrote it.")
    passed: bool = Field(description="Whether the output satisfies this criterion.")
    critical: bool = Field(
        default=False,
        description="Whether this criterion blocks acceptance. Copied from the Manager's plan.",
    )
    evidence: str = Field(
        description=(
            "A direct quote from the worker's output that settles this criterion, "
            "at most 200 characters. Quote the absence-revealing part when it failed."
        )
    )
    reason: str = Field(
        description="Why that evidence shows the criterion was met or violated."
    )

    @field_validator("evidence")
    @classmethod
    def _cap_evidence(cls, value: str) -> str:
        """Truncate rather than reject.

        The 200-character cap keeps logs and the UI readable; it is not worth
        failing a paid round over a quote that ran to 214 characters, which a
        raising validator would do.
        """
        value = value.strip()
        return value if len(value) <= EVIDENCE_MAX_CHARS else value[: EVIDENCE_MAX_CHARS - 1] + "…"


class CriticVerdict(BaseModel):
    """The Critic's judgement: one binary check per acceptance criterion.

    The model is never asked for a score. It answers a series of yes/no
    questions and quotes its evidence; the number is arithmetic done here, in
    code. That is the same rule the Controller follows -- judgement from the
    model, decisions from Python -- applied one level down.

    ``fix_instruction`` is the field that actually moves the loop forward -- it
    is fed back to the Manager, so a diagnosis without a remedy costs a round.

    ``verdict`` is advice, not control flow. Acceptance is decided by
    ``all_passed``, not by the model saying "accept". The value that still
    carries information is ``escalate``: the problem is with the task itself
    rather than the output. It counts as a rejection and returns to the
    Manager, who may then ask the user.
    """

    checks: List[CriterionCheck] = Field(
        default_factory=list,
        description="One entry per acceptance criterion, in the order they were given.",
    )
    fix_instruction: str = Field(
        default="", description="Actionable fix, not a description of the problem."
    )
    verdict: Literal["accept", "revise", "escalate"] = Field(description="What should happen next.")

    @property
    def all_passed(self) -> bool:
        """Every criterion held, critical or not. Reporting only -- see ``accepted``."""
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def blocking_failures(self) -> List[str]:
        """Critical criteria that failed. These are what stop a round."""
        return [check.criterion for check in self.checks if check.critical and not check.passed]

    @property
    def accepted(self) -> bool:
        """The acceptance test: no critical criterion failed.

        Non-critical failures are recorded, shown and fed back to the Manager,
        but they do not buy another round. A criterion nobody was willing to
        call critical should not be able to spend the budget twice.

        Two ways to fail closed. An empty check list judged nothing, and
        "nothing was judged" must never read as "everything passed". A plan
        where *no* criterion was marked critical is treated as though all of
        them were -- otherwise a Manager could make any output acceptable by
        marking nothing important.
        """
        if not self.checks:
            return False
        critical = [check for check in self.checks if check.critical]
        if not critical:
            return all(check.passed for check in self.checks)
        return all(check.passed for check in critical)

    @property
    def score(self) -> int:
        """Share of criteria met, 0-100. Derived, never supplied by the model.

        It exists for logs, dashboards and eval thresholds -- not for the
        accept decision, which is binary.
        """
        if not self.checks:
            return 0
        return round(100 * sum(check.passed for check in self.checks) / len(self.checks))

    @property
    def met_criteria(self) -> List[str]:
        return [check.criterion for check in self.checks if check.passed]

    @property
    def failed_criteria(self) -> List[str]:
        return [check.criterion for check in self.checks if not check.passed]

    def as_record(self) -> Dict[str, Any]:
        """Serialisation for the run log: the raw checks plus what they imply.

        The derived values are written out rather than left to be recomputed,
        so a log line stands on its own years later without this class.
        """
        data = self.model_dump()
        data.update(
            score=self.score,
            accepted=self.accepted,
            all_passed=self.all_passed,
            blocking_failures=self.blocking_failures,
            met_criteria=self.met_criteria,
            failed_criteria=self.failed_criteria,
        )
        return data


class Escalation(BaseModel):
    """A hand-off to the user.

    ``trigger`` is a closed set on purpose: it is the type-level statement of
    the rule that there are exactly three reasons to interrupt someone. Adding
    a fourth means editing this Literal, which is a design decision and should
    read like one in review.
    """

    trigger: Literal["two_rejections", "manager_needs_input", "budget_exceeded"]
    reason: str
    question: Question
