"""Run the task suite through the real loop and print a scorecard.

This is the measurement harness, and it is deliberately *not* self_check.
``self_check`` proves the machinery is correct without spending a cent;
this proves the machinery is any good, which cannot be done without calling
the models. Everything here therefore costs money, and the design follows
from that one fact:

  * a hard suite-wide ceiling, checked between tasks, on top of the per-task
    ceiling the Controller already enforces;
  * a worst-case estimate printed and confirmed *before* the first call;
  * one task's failure never aborts the suite -- a crash is a result, and the
    tasks already paid for should still be reported.

Escalations are answered with silence on purpose. A benchmark that waits for a
human is not a benchmark; an escalation is itself the measurement ("this task
could not be finished unattended"), so the run stops there and the scorecard
records it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from controller import run_task
from providers.retry_utils import QuotaExhausted
from schemas import Question, Task

DEFAULT_TASKS = ROOT / "evals" / "tasks.jsonl"
DEFAULT_RESULTS = ROOT / "evals" / "results"
REQUIRED_KEYS = ("XAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")

console = Console()


def _no_escalation(question: Question) -> int | None:
    """The suite's answer to every escalation: none.

    Returning ``None`` is the Controller's documented "no answer" path, which
    stops that task cleanly and still returns a summary. Picking an option
    would make the score depend on a choice no human made.
    """
    return None


@dataclass
class TaskResult:
    id: str
    status: str = "not_run"
    score: int | None = None
    rounds: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    escalations: list[str] = field(default_factory=list)
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    log_path: str = ""
    result_preview: str = ""


def load_tasks(path: Path, only: list[str], grading: str = "auto") -> list[dict[str, Any]]:
    """Read the task file, keeping only the ones this run is meant to grade.

    A task carries ``grading: "manual"`` when no mechanical check can settle
    it -- "does this email avoid blaming the customer?" has no substring or
    word count that decides it. Those tasks are filtered out here, in the
    loader, rather than inside ``grade()``: the mechanical pass rate has to
    stay a count of mechanically verified things, and mixing in tasks that a
    person still has to read would quietly inflate it.

    Filtering here also leaves ``grade()`` untouched, which matters -- it is
    the one function whose meaning the whole benchmark rests on.
    """
    tasks = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            spec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: not valid JSON -- {exc}") from exc
        if "id" not in spec or "goal" not in spec:
            raise SystemExit(f"{path}:{line_no}: every task needs an 'id' and a 'goal'")
        if only and spec["id"] not in only:
            continue
        task_grading = spec.get("grading", "auto")
        if grading != "all" and task_grading != grading:
            continue
        tasks.append(spec)
    if not tasks:
        raise SystemExit("no tasks selected")
    return tasks


# Every status run_task can end on, plus the wildcard. A task expecting
# anything outside this set is a typo, and a typo that happens to prefix a real
# status would pass silently for as long as nobody read the file.
KNOWN_STATUSES = frozenset(
    {
        "accepted",
        "accepted_by_user",
        "stopped_by_user",
        "stopped_by_flag",
        "escalated_unanswered",
        "max_rounds",
        "crashed",
    }
)


def grade(spec: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    """Check a finished run against the task's expectations.

    The checks are deliberately external to the Critic. A suite that graded
    itself on the Critic's score alone would report a perfect run whenever the
    Critic was simply wrong -- the same self-verification trap the Critic is
    a separate vendor to avoid.

    A task that asserts nothing fails. This used to be the quiet hole in the
    benchmark: ``expect_status: "any"`` skips the status check, ``min_score:
    0`` is falsy and skips the score check, and with no substrings left the
    function returned an empty failure list -- a pass, from zero checks
    performed. Counting that as a pass makes the headline number mean less than
    it appears to. Manual tasks are exempt: a person reads those.
    """
    failures: list[str] = []
    checks_performed = 0

    expected_status = spec.get("expect_status", "accepted")
    if expected_status not in KNOWN_STATUSES and expected_status != "any":
        failures.append(
            f"expect_status {expected_status!r} is not a status run_task can produce"
        )
    elif expected_status != "any":
        checks_performed += 1
        # Exact, not startswith: "accept" silently matching "accepted" and
        # "accepted_by_user" hides the difference between the Critic accepting
        # and a human waving it through.
        if summary["status"] != expected_status:
            failures.append(f"status {summary['status']} != {expected_status}")

    # Acceptance is binary now, so min_score is no longer a duplicate of the
    # accept rule -- it measures *how much* of the criteria set held up, which
    # is the interesting number when a task fails.
    min_score = spec.get("min_score")
    if min_score is not None:
        if min_score <= 0:
            failures.append("min_score 0 checks nothing -- drop the key or raise it")
        else:
            checks_performed += 1
            score = summary.get("score")
            if score is None:
                failures.append("no score produced")
            elif score < min_score:
                failures.append(f"score {score} < {min_score}")

    text = (summary.get("result") or "").lower()
    for needle in spec.get("expect_substrings", []):
        checks_performed += 1
        if needle.lower() not in text:
            failures.append(f"missing {needle!r}")

    max_cost = spec.get("max_cost_usd")
    if max_cost:
        checks_performed += 1
        if summary["budget"]["spent_usd"] > max_cost:
            failures.append(f"cost ${summary['budget']['spent_usd']:.4f} > ${max_cost:.2f}")

    if not checks_performed and spec.get("grading", "auto") != "manual":
        failures.append("this task asserts nothing -- a pass here would be meaningless")

    return failures


def run_one(spec: dict[str, Any], suite_id: str) -> TaskResult:
    outcome = TaskResult(id=spec["id"])
    task = Task(
        goal=spec["goal"],
        context=spec.get("context", ""),
        max_rounds=spec.get("max_rounds", 3),
        budget_usd=spec.get("budget_usd", 0.15),
    )
    started = time.time()
    try:
        summary = run_task(
            task,
            ask=_no_escalation,
            run_id=f"eval-{suite_id}-{spec['id']}",
        )
    except Exception as exc:
        outcome.status = "crashed"
        outcome.failures = [f"{type(exc).__name__}: {exc}"]
        outcome.duration_s = round(time.time() - started, 2)
        return outcome

    outcome.status = summary["status"]
    outcome.score = summary["score"]
    outcome.rounds = summary["rounds"]
    outcome.cost_usd = summary["budget"]["spent_usd"]
    outcome.duration_s = summary["duration_s"]
    outcome.escalations = [item["trigger"] for item in summary["escalations"]]
    outcome.log_path = summary["log_path"]
    outcome.result_preview = (summary.get("result") or "")[:280]
    outcome.failures = grade(spec, summary)
    outcome.passed = not outcome.failures
    return outcome


def print_scorecard(results: list[TaskResult], ceiling: float, elapsed: float) -> None:
    table = Table(title="Eval scorecard", header_style="bold")
    table.add_column("Task")
    table.add_column("Result")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Rounds", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Notes")

    for item in results:
        if item.status == "skipped":
            mark = "[yellow]skip[/yellow]"
        elif item.passed:
            mark = "[green]pass[/green]"
        else:
            mark = "[red]fail[/red]"
        table.add_row(
            item.id,
            mark,
            item.status,
            "—" if item.score is None else str(item.score),
            str(item.rounds),
            f"${item.cost_usd:.4f}",
            "; ".join(item.failures + [f"escalated:{e}" for e in item.escalations])[:60],
        )

    console.print(table)

    ran = [r for r in results if r.status != "skipped"]
    passed = [r for r in ran if r.passed]
    total_cost = sum(r.cost_usd for r in results)
    rounds = [r.rounds for r in ran if r.rounds]

    summary = Table(show_header=False, title="Totals")
    summary.add_row("pass rate", f"{len(passed)}/{len(ran)}" if ran else "0/0")
    summary.add_row("total cost", f"${total_cost:.4f} of ${ceiling:.2f} ceiling")
    summary.add_row("cost / task", f"${total_cost / len(ran):.4f}" if ran else "—")
    summary.add_row("mean rounds", f"{sum(rounds) / len(rounds):.2f}" if rounds else "—")
    summary.add_row("escalations", str(sum(len(r.escalations) for r in results)))
    summary.add_row("wall clock", f"{elapsed:.1f}s")
    console.print(summary)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run the eval suite against the live APIs")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--budget", type=float, default=1.00, help="ceiling for the whole suite")
    parser.add_argument(
        "--only", action="append", default=[], help="run just this task id (repeatable)"
    )
    parser.add_argument(
        "--grading", choices=["auto", "manual", "all"], default="auto",
        help=(
            "which tasks to run. 'auto' (the default) is the mechanical benchmark; "
            "'manual' runs the tasks whose criteria only a person can settle."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args = parser.parse_args(argv)

    missing = [key for key in REQUIRED_KEYS if not os.environ.get(key, "").strip()]
    if missing:
        console.print(f"[red]missing API keys: {', '.join(missing)}[/red]")
        console.print("this harness calls the real APIs; copy .env.example to .env first")
        return 2

    specs = load_tasks(args.tasks, args.only, args.grading)
    worst_case = sum(spec.get("budget_usd", 0.15) for spec in specs)
    if args.grading != "auto":
        console.print(
            f"[yellow]grading={args.grading}: manual tasks are reported, not scored. "
            "Their pass/fail column is only the mechanical checks they happen to "
            "carry -- read the Critic's evidence before believing it.[/yellow]"
        )
    console.print(
        f"[bold]{len(specs)} tasks[/bold] · worst case ${worst_case:.2f} "
        f"if every task burns its ceiling · suite ceiling ${args.budget:.2f}"
    )
    # Real money, so the default is to ask. An unattended run must opt in
    # explicitly rather than inherit consent from a pipe.
    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]not a tty; re-run with --yes to authorise the spend[/red]")
            return 2
        if input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            console.print("aborted")
            return 1

    suite_id = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    results: list[TaskResult] = []
    spent = 0.0
    started = time.time()

    for spec in specs:
        if spent >= args.budget:
            console.print(f"[yellow]suite ceiling reached; skipping {spec['id']}[/yellow]")
            results.append(TaskResult(id=spec["id"], status="skipped",
                                      failures=["suite budget exhausted"]))
            continue
        console.rule(f"[bold]{spec['id']}[/bold]")
        try:
            outcome = run_one(spec, suite_id)
        except QuotaExhausted as exc:
            console.print(f"[red]{exc}[/red]")
            console.print("[red]A daily allowance, not a rate limit. Re-run after it resets.[/red]")
            break
        except KeyboardInterrupt:
            # Interrupting a paid run should still report what it bought.
            console.print("[yellow]interrupted; reporting what finished[/yellow]")
            break
        spent += outcome.cost_usd
        results.append(outcome)

    elapsed = time.time() - started
    print_scorecard(results, args.budget, elapsed)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "suite_id": suite_id,
        "tasks_file": str(args.tasks),
        "ceiling_usd": args.budget,
        "total_cost_usd": round(spent, 6),
        "elapsed_s": round(elapsed, 2),
        "results": [asdict(item) for item in results],
    }
    out_path = args.out / f"{suite_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"report: {out_path}")

    ran = [r for r in results if r.status != "skipped"]
    return 0 if ran and all(r.passed for r in ran) else 1


if __name__ == "__main__":
    raise SystemExit(main())
