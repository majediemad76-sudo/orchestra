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

Every escalation is one question with 2-4 concrete options. See
``schemas.Question`` for why.
"""

from __future__ import annotations

import argparse
import json
import sys
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

from budget import BudgetGuard
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


# --- asking the user -------------------------------------------------------

AnswerFn = Callable[[Question], Optional[int]]


def ask_on_console(question: Question) -> Optional[int]:
    """Ask on the terminal. ``None`` means no answer, which always means stop.

    Injected rather than called directly so the loop can be driven by tests, or
    later by a web UI, without the state machine knowing the difference.

    A non-tty is treated as "no answer" instead of guessing a default: the
    three triggers exist because the run genuinely needs a human, and a run
    that silently picks option 1 in CI is worse than one that halts and says
    what it was about to ask.
    """
    console.print(Panel(question.text, title="ارجاع به کاربر", border_style="yellow"))
    for index, option in enumerate(question.options, start=1):
        console.print(f"  [bold]{index}[/bold]. {option}")
    if not sys.stdin.isatty():
        console.print("[yellow]stdin is not interactive; stopping with the question open.[/yellow]")
        return None
    while True:
        raw = input("انتخاب شما (شماره، یا خالی برای توقف): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(question.options):
            return int(raw) - 1
        console.print("[red]شماره‌ی نامعتبر.[/red]")


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
) -> Dict[str, Any]:
    run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    log = RunLog(run_id)
    state = RunState(task=task, budget=BudgetGuard(limit_usd=task.budget_usd))
    log.write("run_start", task=task.model_dump())
    started = time.time()

    status = "max_rounds"
    while state.round < task.max_rounds:
        state.round += 1
        console.rule(f"دور {state.round} / {task.max_rounds}")

        # Trigger 3, checked first and before any call: the previous round
        # may have crossed the line, and the cheapest possible reaction is to
        # not start a new one.
        if state.budget.exceeded:
            action, _ = _escalate(
                state,
                log,
                "budget_exceeded",
                f"سقف بودجه ({state.budget.limit_usd:.2f}$) رد شد؛ "
                f"تاکنون {state.budget.spent_usd:.4f}$ خرج شده. چه کار کنم؟",
                [
                    ("سقف بودجه را دو برابر کن و ادامه بده", RAISE_BUDGET),
                    ("بهترین خروجی فعلی را بپذیر و تمام کن", ACCEPT_BEST),
                    ("همین‌جا متوقف شو", STOP),
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
            model=entry.model,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
        )
        console.print(Panel(plan.plan, title=f"نقشه‌ی Manager ({plan.worker_type})", border_style="cyan"))

        # Trigger 2. The Manager reports; the Controller decides -- and the
        # question comes from the Manager because it is the role that knows
        # what is missing.
        if plan.needs_user_input and plan.question is not None:
            options = list(plan.question.options)[:4]
            while len(options) < 2:
                options.append("خودت بهترین حدس را بزن و ادامه بده")
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
            entry = state.budget.charge_usd(f"round{state.round}.worker", run.model, run.cost_usd)
        else:
            entry = state.budget.charge(
                f"round{state.round}.worker", run.model, run.input_tokens, run.output_tokens
            )
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
            model=entry.model,
        )

        # --- Critic --------------------------------------------------------
        if not run.output.ok:
            # No output to grade: synthesise the rejection instead of paying
            # the Critic to state the obvious.
            verdict = critic_role.failed_worker_verdict(run.failure_reason or "worker failed")
            log.write("critic_verdict", round=state.round, verdict=verdict.model_dump(), cost_usd=0.0, synthetic=True)
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
                verdict=verdict.model_dump(),
                cost_usd=entry.cost_usd,
                model=entry.model,
            )
        state.verdict = verdict
        console.print(
            Panel(
                f"score={verdict.score}  verdict={verdict.verdict}\n{verdict.fix_instruction}",
                title="نظر Critic",
                border_style="magenta" if verdict.verdict != "accept" else "green",
            )
        )

        if verdict.score > state.best_score:
            state.best_score, state.best_result = verdict.score, run.output.result

        accepted = verdict.verdict == "accept" and verdict.score >= task.accept_score
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
                "دو دور پیاپی رد شد. آخرین ایراد: "
                + (verdict.fix_instruction or ", ".join(verdict.failed_criteria) or "نامشخص"),
                [
                    ("همان مسیر را با اصلاح پیشنهادی Critic ادامه بده", CONTINUE),
                    ("معیارهای پذیرش را ساده‌تر کن و دوباره تلاش کن", CONTINUE),
                    ("بهترین خروجی فعلی را بپذیر و تمام کن", ACCEPT_BEST),
                    ("همین‌جا متوقف شو", STOP),
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
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    table = Table(title="خلاصه‌ی اجرا", show_header=False)
    table.add_row("status", str(summary["status"]))
    table.add_row("rounds", str(summary["rounds"]))
    table.add_row("score", str(summary["score"]))
    table.add_row("spent", f"{summary['budget']['spent_usd']:.4f}$ / {summary['budget']['limit_usd']:.2f}$")
    table.add_row("log", summary["log_path"])
    console.print(table)
    if summary["result"]:
        console.print(Panel(summary["result"][:4000], title="خروجی", border_style="green"))


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Multi-model orchestrator")
    parser.add_argument("goal", help="what you want done")
    parser.add_argument("--context", default="", help="extra context for the manager")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1.0, help="dollar ceiling for the run")
    parser.add_argument("--accept-score", type=int, default=80)
    parser.add_argument("--cwd", default=None, help="working directory for the code worker")
    args = parser.parse_args(argv)

    task = Task(
        goal=args.goal,
        context=args.context,
        max_rounds=args.max_rounds,
        budget_usd=args.budget,
        accept_score=args.accept_score,
    )
    summary = run_task(task, cwd=args.cwd)
    print_summary(summary)
    return 0 if summary["status"].startswith("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
