"""The Controller: a state machine in Python, never a model.

Every consequential decision -- how many retries, when to stop, when to
interrupt a human, when the money has run out -- is made here, in code you can
read end to end. Models are asked for judgement (is this good? what should the
Worker do?); they are never asked what happens next. A model that decides its
own retry budget will always find one more thing worth trying.

The user is interrupted for exactly three reasons:

  1. ``two_rejections``     -- the Critic rejected twice in a row
  2. ``manager_needs_input``-- the Manager reports missing information
  3. ``budget_exceeded``    -- the dollar ceiling was crossed

Everything else resolves inside the loop. A Critic ``escalate`` verdict counts
as a rejection and returns to the Manager, who may then trigger (2). A Worker
timeout counts as a rejection. Hitting the round cap ends the run with a
report. This is a deliberate narrowing: an orchestrator that asks whenever it
is unsure is an orchestrator nobody leaves running.

A round is accepted when every acceptance criterion passes its own binary
check -- see ``schemas.CriticVerdict.all_passed``. There is no score threshold
to tune, and the model's own "accept" does not decide anything.

Every escalation is one question with 2-4 concrete options. See
``schemas.Question`` for why.

``run_task`` runs headless by default and drives the terminal. It also accepts
three optional hooks -- ``on_progress``, ``on_escalation``, ``stop_flag`` -- so
a UI or another process can observe the run, answer its questions, and cancel
it. The hooks are strictly additive: none of them can change a decision, and
with all three omitted the loop behaves exactly as it did before they existed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from budget import API_BILLED, SUBSCRIPTION_EQUIVALENT, BudgetGuard
from roles import critic as critic_role
from roles import manager as manager_role
from roles import worker as worker_role
from schemas import CriticVerdict, Escalation, ManagerPlan, Question, Task, WorkerOutput

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
console = Console()

# What an answered escalation does. The user picks a label; the Controller
# reads the action bound to it. Keeping the two separate means option wording
# can be rewritten -- or translated -- without touching control flow.
CONTINUE = "continue"
STOP = "stop"
ACCEPT_BEST = "accept_best"
RAISE_BUDGET = "raise_budget"


class RunLog:
    """Append-only JSONL, one file per run.

    JSONL rather than a single JSON document because a run that dies partway
    through still leaves a readable file, and ``tail -f`` works while it is in
    flight. Every stage is recorded with its cost, so "why did this run cost
    $2" is answerable after the fact rather than reconstructible at best.
    """

    def __init__(self, run_id: str, directory: Optional[Path] = None):
        directory = directory or RUNS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{run_id}.jsonl"
        self.run_id = run_id

    def write(self, event: str, **data: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@dataclass
class RunState:
    """Everything that survives from one round to the next.

    ``best_result`` tracks the highest-scoring output separately from the
    latest one: revision is not monotonic, and a round can make things worse.
    Without it, "accept the best you have" would mean "accept the most recent",
    which is not the same offer.
    """

    task: Task
    budget: BudgetGuard
    round: int = 0
    consecutive_rejections: int = 0
    plan: Optional[ManagerPlan] = None
    output: Optional[WorkerOutput] = None
    verdict: Optional[CriticVerdict] = None
    user_answer: str = ""
    best_result: str = ""
    best_score: int = -1
    escalations: List[Dict[str, Any]] = field(default_factory=list)


# --- talking to whoever is driving -----------------------------------------

# The loop's own asker: an option index, or None meaning "no answer, stop".
AnswerFn = Callable[[Question], Optional[int]]

# Embedder-facing hooks. Deliberately narrower than the internals they wrap:
# progress is one-way, and an escalation answer is a label the caller saw --
# neither can express an action the Controller did not already offer.
ProgressFn = Callable[[str, Dict[str, Any]], None]
EscalationFn = Callable[[Question], str]


def ask_on_console(question: Question) -> Optional[int]:
    """Ask on the terminal. ``None`` means no answer, which always means stop.

    Injected rather than called directly so the loop can be driven by tests, or
    later by a web UI, without the state machine knowing the difference.

    A non-tty is treated as "no answer" instead of guessing a default: the
    three triggers exist because the run genuinely needs a human, and a run
    that silently picks option 1 in CI is worse than one that halts and says
    what it was about to ask.
    """
    console.print(Panel(question.text, title="Escalation", border_style="yellow"))
    for index, option in enumerate(question.options, start=1):
        console.print(f"  [bold]{index}[/bold]. {option}")
    if not sys.stdin.isatty():
        console.print("[yellow]stdin is not interactive; stopping with the question open.[/yellow]")
        return None
    while True:
        raw = input("Your choice (number, or blank to stop): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(question.options):
            return int(raw) - 1
        console.print("[red]Not a valid option number.[/red]")


def _asker_from_escalation(on_escalation: EscalationFn) -> AnswerFn:
    """Adapt an embedder's label-returning hook to the loop's index-returning one.

    The hook returns what the user picked, as text, because that is what a UI
    naturally has. Matching it back to an option index happens here so the
    state machine keeps its single notion of an answer.

    An unrecognised answer becomes ``None`` -- the same as no answer, i.e.
    stop. Coercing it to a default would let a UI bug spend money or abandon a
    run without anyone having chosen either.
    """

    def ask(question: Question) -> Optional[int]:
        answer = on_escalation(question)
        if answer is None:
            return None
        answer = str(answer).strip()
        if not answer:
            return None
        options = question.options
        if answer in options:
            return options.index(answer)
        folded = [option.strip().casefold() for option in options]
        if answer.casefold() in folded:
            return folded.index(answer.casefold())
        # A bare "1".."4" is what a CLI-ish front end tends to send back.
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        console.print(f"[yellow]unrecognised escalation answer: {answer!r} -- stopping[/yellow]")
        return None

    return ask


def _progress_emitter(on_progress: Optional[ProgressFn]) -> ProgressFn:
    """Wrap the progress hook so a broken observer cannot kill a paid run.

    Reporting is not part of the work. A UI callback that raises -- a closed
    websocket, a dead queue -- gets its exception surfaced on the console and
    the loop carries on, because the alternative is discarding a run mid-flight
    over a rendering problem.
    """
    if on_progress is None:
        return lambda event, data: None

    def emit(event: str, data: Dict[str, Any]) -> None:
        try:
            on_progress(event, data)
        except Exception as exc:  # noqa: BLE001 -- observers must not be fatal
            console.print(f"[yellow]on_progress({event}) raised {exc!r}; continuing[/yellow]")

    return emit


def _escalate(
    state: RunState,
    log: RunLog,
    trigger: str,
    reason: str,
    options: List[Tuple[str, str]],
    ask: AnswerFn,
) -> Tuple[str, str]:
    """Raise one escalation and return ``(action, chosen label)``.

    The 2..4 bound is asserted here as well as in ``Question`` -- this is the
    one function that constructs escalations, so it is the last place a
    malformed one can be caught before it reaches a human.
    """
    if not 2 <= len(options) <= 4:
        raise ValueError("an escalation must offer between 2 and 4 options")
    question = Question(text=reason, options=[label for label, _ in options])
    escalation = Escalation(trigger=trigger, reason=reason, question=question)
    log.write("escalation", **escalation.model_dump())

    choice = ask(question)
    if choice is None:
        log.write("escalation_answer", trigger=trigger, answer=None, action=STOP)
        state.escalations.append({"trigger": trigger, "answer": None})
        return STOP, ""
    label, action = options[choice]
    log.write("escalation_answer", trigger=trigger, answer=label, action=action)
    state.escalations.append({"trigger": trigger, "answer": label})
    return action, label


# --- the loop --------------------------------------------------------------


def run_task(
    task: Task,
    cwd: Optional[str] = None,
    ask: AnswerFn = ask_on_console,
    run_id: Optional[str] = None,
    on_progress: Optional[ProgressFn] = None,
    on_escalation: Optional[EscalationFn] = None,
    stop_flag: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run the Manager/Worker/Critic loop to acceptance, exhaustion or halt.

    The three optional hooks let another process drive this loop without
    changing it:

    ``on_progress(event, data)``
        Called after each stage -- ``round_start``, ``manager_plan``,
        ``worker_output``, ``critic_verdict``, ``run_end``. Observation only;
        the return value is ignored and an exception is not fatal. The JSONL
        log remains the record of truth, unchanged.

    ``on_escalation(question) -> str``
        Replaces the terminal prompt at all three triggers. Returns the label
        the user picked; anything unrecognised is treated as no answer, which
        means stop.

    ``stop_flag``
        Checked at the top of each round. When set, the run ends with status
        ``stopped_by_flag`` and returns the state it had reached.

    With all three omitted this is the original headless behaviour, down to the
    console prompts.
    """
    run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log = RunLog(run_id)
    state = RunState(task=task, budget=BudgetGuard(limit_usd=task.budget_usd))
    emit = _progress_emitter(on_progress)
    if on_escalation is not None:
        ask = _asker_from_escalation(on_escalation)
    log.write("run_start", task=task.model_dump())
    started = time.time()

    status = "max_rounds"
    while state.round < task.max_rounds:
        # Cancellation is checked before the round is counted and before any
        # money is spent, so a stopped run is never charged for work nobody
        # will read.
        if stop_flag is not None and stop_flag.is_set():
            status = "stopped_by_flag"
            break

        state.round += 1
        console.rule(f"Round {state.round} / {task.max_rounds}")
        emit(
            "round_start",
            {
                "round": state.round,
                "max_rounds": task.max_rounds,
                "spent_usd": state.budget.spent_usd,
                "limit_usd": state.budget.limit_usd,
            },
        )

        # Trigger 3, checked first and before any call: the previous round
        # may have crossed the line, and the cheapest possible reaction is to
        # not start a new one.
        if state.budget.exceeded:
            action, _ = _escalate(
                state,
                log,
                "budget_exceeded",
                f"The budget ceiling (${state.budget.limit_usd:.2f}) was crossed; "
                f"${state.budget.spent_usd:.4f} spent so far. How should I proceed?",
                [
                    ("Double the ceiling and keep going", RAISE_BUDGET),
                    ("Accept the best output so far and finish", ACCEPT_BEST),
                    ("Stop here", STOP),
                ],
                ask,
            )
            if action == RAISE_BUDGET:
                state.budget.limit_usd *= 2
                log.write("budget_raised", limit_usd=state.budget.limit_usd)
            elif action == ACCEPT_BEST:
                status = "accepted_by_user"
                break
            else:
                status = "stopped_by_user"
                break

        # --- Manager -------------------------------------------------------
        plan, plan_call = manager_role.plan(
            task,
            previous_plan=state.plan,
            verdict=state.verdict,
            worker_result=state.output.result if state.output else "",
            user_answer=state.user_answer,
        )
        entry = state.budget.charge(
            f"round{state.round}.manager", plan_call.model, plan_call.input_tokens, plan_call.output_tokens
        )
        state.plan = plan
        state.user_answer = ""
        log.write(
            "manager_plan",
            round=state.round,
            plan=plan.model_dump(),
            cost_usd=entry.cost_usd,
            cost_basis=API_BILLED,
            model=entry.model,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
        )
        console.print(Panel(plan.plan, title=f"Manager plan ({plan.worker_type})", border_style="cyan"))
        emit(
            "manager_plan",
            {
                "round": state.round,
                "plan": plan.plan,
                "worker_prompt": plan.worker_prompt,
                "acceptance_criteria": list(plan.acceptance_criteria),
                "worker_type": plan.worker_type,
                "needs_user_input": plan.needs_user_input,
                "cost_usd": entry.cost_usd,
                "model": entry.model,
            },
        )

        # Trigger 2. The Manager reports; the Controller decides -- and the
        # question comes from the Manager because it is the role that knows
        # what is missing.
        if plan.needs_user_input and plan.question is not None:
            options = list(plan.question.options)[:4]
            while len(options) < 2:
                options.append("Make your best assumption and continue")
            action, label = _escalate(
                state,
                log,
                "manager_needs_input",
                plan.question.text,
                [(option, CONTINUE) for option in options],
                ask,
            )
            if action == STOP:
                status = "escalated_unanswered"
                break
            state.user_answer = label
            # Answering a question is not an attempt at the task. Charging it
            # a round would let a talkative Manager exhaust the cap without
            # the Worker ever running.
            state.round -= 1
            continue

        # --- Worker --------------------------------------------------------
        run = worker_role.execute(plan, cwd=cwd)
        if run.cost_usd is not None:
            # Only Claude Code headless reports dollars directly, and on a
            # personal subscription that figure is API-equivalent rather than
            # billed. See providers/claude_code.py.
            entry = state.budget.charge_usd(f"round{state.round}.worker", run.model, run.cost_usd)
            cost_basis = SUBSCRIPTION_EQUIVALENT
        else:
            entry = state.budget.charge(
                f"round{state.round}.worker", run.model, run.input_tokens, run.output_tokens
            )
            cost_basis = API_BILLED
        state.output = run.output
        log.write(
            "worker_output",
            round=state.round,
            worker_type=plan.worker_type,
            ok=run.output.ok,
            result=run.output.result,
            notes=run.output.notes,
            failure_reason=run.failure_reason,
            cost_usd=entry.cost_usd,
            cost_basis=cost_basis,
            model=entry.model,
        )
        emit(
            "worker_output",
            {
                "round": state.round,
                "worker_type": plan.worker_type,
                "ok": run.output.ok,
                "result": run.output.result,
                "notes": run.output.notes,
                "failure_reason": run.failure_reason,
                "cost_usd": entry.cost_usd,
                "cost_basis": cost_basis,
                "model": entry.model,
            },
        )

        # --- Critic --------------------------------------------------------
        if not run.output.ok:
            # No output to grade: synthesise the rejection instead of paying
            # the Critic to state the obvious.
            verdict = critic_role.failed_worker_verdict(
                run.failure_reason or "worker failed", plan.acceptance_criteria
            )
            log.write(
                "critic_verdict",
                round=state.round,
                verdict=verdict.as_record(),
                cost_usd=0.0,
                cost_basis=API_BILLED,
                synthetic=True,
            )
        else:
            verdict, critic_call = critic_role.review(plan, run.output)
            entry = state.budget.charge(
                f"round{state.round}.critic",
                critic_call.model,
                critic_call.input_tokens,
                critic_call.output_tokens,
            )
            log.write(
                "critic_verdict",
                round=state.round,
                verdict=verdict.as_record(),
                cost_usd=entry.cost_usd,
                cost_basis=API_BILLED,
                model=entry.model,
            )
        state.verdict = verdict
        lines = [
            f"{'✓' if check.passed else '✗'} {check.criterion}" for check in verdict.checks
        ] or ["(the Critic returned no checks)"]
        if verdict.fix_instruction:
            lines += ["", verdict.fix_instruction]
        console.print(
            Panel(
                "\n".join(lines),
                title=f"Critic verdict — {verdict.score}% ({len(verdict.met_criteria)}"
                f"/{len(verdict.checks)} criteria)",
                border_style="green" if verdict.all_passed else "magenta",
            )
        )
        emit(
            "critic_verdict",
            {
                "round": state.round,
                "score": verdict.score,
                "all_passed": verdict.all_passed,
                "verdict": verdict.verdict,
                "checks": [check.model_dump() for check in verdict.checks],
                "met_criteria": list(verdict.met_criteria),
                "failed_criteria": list(verdict.failed_criteria),
                "fix_instruction": verdict.fix_instruction,
                "spent_usd": state.budget.spent_usd,
            },
        )

        if verdict.score > state.best_score:
            state.best_score, state.best_result = verdict.score, run.output.result

        # The one line that decides a round. Not the model's "accept" -- every
        # criterion the Manager wrote has to pass on its own. A criterion the
        # Critic could not judge (an empty check list) fails closed.
        accepted = verdict.all_passed
        if accepted:
            state.consecutive_rejections = 0
            status = "accepted"
            break

        state.consecutive_rejections += 1

        # Trigger 1. Two, not one: a single rejection is the loop working as
        # intended, and interrupting on it would make the Critic pointless.
        # Two in a row means revision is not converging, and a third round is
        # more likely to burn budget than to fix it.
        if state.consecutive_rejections >= 2:
            action, label = _escalate(
                state,
                log,
                "two_rejections",
                "Two rounds were rejected in a row. Latest problem: "
                + (verdict.fix_instruction or ", ".join(verdict.failed_criteria) or "unspecified"),
                [
                    ("Keep this approach and apply the Critic's fix", CONTINUE),
                    ("Loosen the acceptance criteria and retry", CONTINUE),
                    ("Accept the best output so far and finish", ACCEPT_BEST),
                    ("Stop here", STOP),
                ],
                ask,
            )
            if action == ACCEPT_BEST:
                status = "accepted_by_user"
                break
            if action == STOP:
                status = "stopped_by_user"
                break
            state.user_answer = label
            state.consecutive_rejections = 0

    summary = {
        "run_id": run_id,
        "status": status,
        "rounds": state.round,
        "score": state.verdict.score if state.verdict else None,
        "result": state.output.result if state.output else state.best_result,
        "best_result": state.best_result,
        "budget": state.budget.summary(),
        "escalations": state.escalations,
        "log_path": str(log.path),
        "duration_s": round(time.time() - started, 2),
    }
    log.write("run_end", **{k: v for k, v in summary.items() if k != "result"})
    emit("run_end", dict(summary))
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    table = Table(title="Run summary", show_header=False)
    table.add_row("status", str(summary["status"]))
    table.add_row("rounds", str(summary["rounds"]))
    table.add_row("score", str(summary["score"]))
    table.add_row("spent", f"{summary['budget']['spent_usd']:.4f}$ / {summary['budget']['limit_usd']:.2f}$")
    table.add_row("log", summary["log_path"])
    console.print(table)
    if summary["result"]:
        console.print(Panel(summary["result"][:4000], title="Output", border_style="green"))


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Multi-model orchestrator")
    parser.add_argument("goal", help="what you want done")
    parser.add_argument("--context", default="", help="extra context for the manager")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1.0, help="dollar ceiling for the run")
    parser.add_argument("--cwd", default=None, help="working directory for the code worker")
    args = parser.parse_args(argv)

    task = Task(
        goal=args.goal,
        context=args.context,
        max_rounds=args.max_rounds,
        budget_usd=args.budget,
    )
    summary = run_task(task, cwd=args.cwd)
    print_summary(summary)
    return 0 if summary["status"].startswith("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
