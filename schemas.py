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

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


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
    accept_score: int = Field(default=80, ge=0, le=100, description="Critic score needed to accept.")


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
    acceptance_criteria: List[str] = Field(
        min_length=1, description="Objectively checkable criteria, not matters of taste."
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


class CriticVerdict(BaseModel):
    """The Critic's grade against the acceptance criteria.

    ``fix_instruction`` is the field that actually moves the loop forward -- it
    is fed back to the Manager, so a diagnosis without a remedy costs a round.

    ``verdict`` is advice, not control flow: ``escalate`` does not reach the
    user by itself (see the three triggers in ``controller.py``); it counts as
    a rejection and returns to the Manager, who may then ask for input.
    """

    score: int = Field(ge=0, le=100, description="0-100 score against the acceptance criteria.")
    met_criteria: List[str] = Field(default_factory=list, description="Criteria that passed.")
    failed_criteria: List[str] = Field(default_factory=list, description="Criteria that failed.")
    fix_instruction: str = Field(
        default="", description="Actionable fix, not a description of the problem."
    )
    verdict: Literal["accept", "revise", "escalate"] = Field(description="What should happen next.")


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
