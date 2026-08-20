"""Measure the Critic against reviewed fixtures.

Two questions, deliberately kept apart:

  * On outputs that genuinely satisfy their criteria, how often does the Critic
    accept? (a low rate here means it rejects good work -- expensive, because
    every false rejection buys another round)
  * On outputs broken in exactly one named way, how often does it reject, per
    mutation type? (a low rate here means it waves through broken work --
    worse, because nothing downstream catches it)

A single "percent correct" would average these into a number that hides both.
A judge that accepts everything and a judge that rejects everything can land on
the same overall score; they are not the same judge, and neither failure is
fixed by the same change.

Only reviewed fixtures count. An unreviewed row is a hypothesis about ground
truth, and grading a judge against a hypothesis measures nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from budget import BudgetGuard
from providers.retry_utils import ProviderError
from roles import critic as critic_role
from schemas import Criterion, CriticVerdict, ManagerPlan, WorkerOutput

DEFAULT_FIXTURES = ROOT / "evals" / "critic_fixtures.jsonl"
DEFAULT_RESULTS = ROOT / "evals" / "results"

console = Console()


@dataclass
class Outcome:
    fixture_id: str
    expected: str
    mutation_type: str | None
    severity: str | None = None
    broken_criterion: str | None = None
    accepted: bool | None = None
    correct: bool = False
    caught_named_criterion: bool | None = None
    score: int | None = None
    cost_usd: float = 0.0
    error: str = ""


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_fixtures(path: Path, include_unreviewed: bool) -> list[dict[str, Any]]:
    rows, skipped = [], 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            console.print(f"[yellow]{path.name}:{line_no}: unreadable, skipped[/yellow]")
            continue
        if not row.get("reviewed") and not include_unreviewed:
            skipped += 1
            continue
        rows.append(row)
    if skipped:
        console.print(f"[yellow]{skipped} unreviewed fixture(s) skipped[/yellow]")
    return rows


def as_plan(row: dict[str, Any]) -> ManagerPlan:
    """Rebuild the Manager's plan so the Critic sees exactly what it saw live."""
    criteria = [
        Criterion(
            text=c["text"],
            critical=bool(c.get("critical", True)),
            check_method=c.get("check_method", ""),
        )
        for c in row["acceptance_criteria"]
    ]
    return ManagerPlan(
        plan="(replayed from a fixture)",
        worker_prompt=row.get("worker_prompt") or row.get("goal", ""),
        acceptance_criteria=criteria,
        worker_type="text",
    )


def is_correct(expected: str, accepted: bool) -> bool:
    """Did the Critic agree with the fixture?

    Pulled out of the loop so it can be tested without spending anything: the
    offline negative control in self_check drives this exact rule.
    """
    return accepted if expected == "accept" else not accepted


def grade_outcome(row: dict[str, Any], verdict: CriticVerdict) -> Outcome:
    """Score one verdict against one fixture. Pure: no network, no budget.

    Kept separate from ``judge`` so the scoring rule can be exercised with a
    hand-built ``CriticVerdict`` -- which is how the negative control in
    self_check runs on every gate without an API key or a cent of spend.
    """
    outcome = Outcome(
        fixture_id=row["fixture_id"],
        expected=row["expected_verdict"],
        mutation_type=row.get("mutation_type"),
        severity=row.get("severity"),
        broken_criterion=row.get("broken_criterion"),
    )
    outcome.accepted = verdict.accepted
    outcome.score = verdict.score
    outcome.correct = is_correct(row["expected_verdict"], verdict.accepted)

    named = row.get("broken_criterion")
    if named:
        # Rejecting for the wrong reason is not the same as being right. The
        # fix_instruction that goes back to the Manager comes from whichever
        # criterion the Critic thinks failed, so pointing at the wrong one
        # sends the next round chasing the wrong problem.
        failed = {c.criterion.strip() for c in verdict.checks if not c.passed}
        outcome.caught_named_criterion = named.strip() in failed
    return outcome


def judge(row: dict[str, Any], budget: BudgetGuard) -> Outcome:
    """Ask the real Critic about one fixture, then grade what came back."""
    try:
        verdict, call = critic_role.review(as_plan(row), WorkerOutput(result=row["output"]))
    except (ProviderError, ValueError) as exc:
        return Outcome(
            fixture_id=row["fixture_id"],
            expected=row["expected_verdict"],
            mutation_type=row.get("mutation_type"),
            error=f"{type(exc).__name__}: {exc}",
        )

    entry = budget.charge(row["fixture_id"][:40], call.model, call.input_tokens, call.output_tokens)
    outcome = grade_outcome(row, verdict)
    outcome.cost_usd = entry.cost_usd
    return outcome


def print_report(report: Report, budget: BudgetGuard, elapsed: float) -> dict[str, Any]:
    ran = [o for o in report.outcomes if not o.error]
    accepts = [o for o in ran if o.expected == "accept"]
    revises = [o for o in ran if o.expected == "revise"]

    accept_rate = sum(o.correct for o in accepts) / len(accepts) if accepts else 0.0
    reject_rate = sum(o.correct for o in revises) / len(revises) if revises else 0.0

    headline = Table(title="Critic scorecard", show_header=False)
    headline.add_row(
        "acceptance rate on accept cases",
        f"{sum(o.correct for o in accepts)}/{len(accepts)}  ({accept_rate:.0%})",
    )
    headline.add_row(
        "rejection rate on revise cases",
        f"{sum(o.correct for o in revises)}/{len(revises)}  ({reject_rate:.0%})",
    )
    console.print(headline)

    by_type: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in revises:
        by_type[outcome.mutation_type or "?"].append(outcome)

    table = Table(title="Rejection rate by mutation type", header_style="bold")
    table.add_column("Mutation type")
    table.add_column("Caught", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Right criterion", justify="right")
    for mutation_type in sorted(by_type):
        items = by_type[mutation_type]
        caught = sum(o.correct for o in items)
        named = [o for o in items if o.caught_named_criterion is not None]
        right = sum(bool(o.caught_named_criterion) for o in named)
        table.add_row(
            mutation_type,
            f"{caught}/{len(items)}",
            f"{caught / len(items):.0%}",
            f"{right}/{len(named)}" if named else "—",
        )
    console.print(table)

    # Severity is the axis that separates a judge from a keyword matcher.
    # Catching "walking cures diabetes" says almost nothing; catching "walking
    # can help with blood sugar control" is the whole question. Averaging the
    # two hides exactly the number worth knowing.
    labelled = [o for o in revises if o.severity in {"blatant", "borderline"}]
    if labelled:
        severity_table = Table(title="Rejection rate by severity", header_style="bold")
        severity_table.add_column("Severity")
        severity_table.add_column("Caught", justify="right")
        severity_table.add_column("Rate", justify="right")
        severity_table.add_column("Right criterion", justify="right")
        for level in ("blatant", "borderline"):
            items = [o for o in labelled if o.severity == level]
            if not items:
                continue
            caught = sum(o.correct for o in items)
            named = [o for o in items if o.caught_named_criterion is not None]
            right = sum(bool(o.caught_named_criterion) for o in named)
            severity_table.add_row(
                level,
                f"{caught}/{len(items)}",
                f"{caught / len(items):.0%}",
                f"{right}/{len(named)}" if named else "—",
            )
        unlabelled = len(revises) - len(labelled)
        if unlabelled:
            severity_table.add_row("(unlabelled)", f"—/{unlabelled}", "—", "—")
        console.print(severity_table)

    # Per criterion, because an aggregate hides which *kind* of rule the judge
    # is soft on. A Critic can be perfect on word counts and blind to implied
    # health claims, and the headline number reads the same either way.
    by_criterion: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in revises:
        if outcome.broken_criterion:
            by_criterion[outcome.broken_criterion].append(outcome)
    if by_criterion:
        criterion_table = Table(title="Rejection rate by criterion", header_style="bold")
        criterion_table.add_column("Criterion", max_width=54)
        criterion_table.add_column("Caught", justify="right")
        criterion_table.add_column("Blatant", justify="right")
        criterion_table.add_column("Borderline", justify="right")
        for criterion, items in sorted(by_criterion.items(), key=lambda kv: -len(kv[1])):
            blatant = [o for o in items if o.severity == "blatant"]
            borderline = [o for o in items if o.severity == "borderline"]
            criterion_table.add_row(
                criterion,
                f"{sum(o.correct for o in items)}/{len(items)}",
                f"{sum(o.correct for o in blatant)}/{len(blatant)}" if blatant else "—",
                f"{sum(o.correct for o in borderline)}/{len(borderline)}" if borderline else "—",
            )
        console.print(criterion_table)

    misses = [o for o in ran if not o.correct]
    if misses:
        detail = Table(title="Misses", header_style="bold")
        detail.add_column("Fixture")
        detail.add_column("Expected")
        detail.add_column("Critic said")
        detail.add_column("Score", justify="right")
        for outcome in misses:
            detail.add_row(
                outcome.fixture_id[-42:],
                outcome.expected,
                "accept" if outcome.accepted else "revise",
                str(outcome.score),
            )
        console.print(detail)

    for error in report.errors:
        console.print(f"[red]{error}[/red]")

    console.print(
        f"cost ${budget.spent_usd:.4f} of ${budget.limit_usd:.2f} · {elapsed:.1f}s · "
        f"{len(ran)}/{len(report.outcomes)} fixtures judged"
    )

    return {
        "accept_cases": len(accepts),
        "accept_rate": round(accept_rate, 4),
        "revise_cases": len(revises),
        "reject_rate": round(reject_rate, 4),
        "by_mutation_type": {
            k: {
                "cases": len(v),
                "caught": sum(o.correct for o in v),
                "rate": round(sum(o.correct for o in v) / len(v), 4),
                "right_criterion": sum(bool(o.caught_named_criterion) for o in v),
            }
            for k, v in by_type.items()
        },
        "by_severity": {
            level: {
                "cases": len([o for o in revises if o.severity == level]),
                "caught": sum(o.correct for o in revises if o.severity == level),
            }
            for level in ("blatant", "borderline")
            if any(o.severity == level for o in revises)
        },
        "by_criterion": {
            criterion: {
                "cases": len(items),
                "caught": sum(o.correct for o in items),
                "borderline_cases": len([o for o in items if o.severity == "borderline"]),
                "borderline_caught": sum(
                    o.correct for o in items if o.severity == "borderline"
                ),
            }
            for criterion, items in by_criterion.items()
        },
        "errors": len(report.errors),
        "cost_usd": round(budget.spent_usd, 6),
        "elapsed_s": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Grade the Critic against reviewed fixtures")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="judge only the first N fixtures")
    parser.add_argument(
        "--delay", type=float, default=4.0,
        help="seconds between calls. The default paces below the Gemini free tier's "
             "15 requests per minute; pass 0 on a paid tier.",
    )
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    if not args.fixtures.exists():
        console.print(f"[red]no fixtures at {args.fixtures}[/red]")
        console.print("build them first: make fixture, then review the draft")
        return 2
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        console.print("[red]GOOGLE_API_KEY is not set; the Critic runs on Gemini[/red]")
        return 2

    rows = load_fixtures(args.fixtures, args.include_unreviewed)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        console.print("[red]no reviewed fixtures to run[/red]")
        return 1

    accepts = sum(1 for r in rows if r["expected_verdict"] == "accept")
    console.print(
        f"{len(rows)} fixtures: {accepts} accept, {len(rows) - accepts} revise · "
        f"ceiling ${args.budget:.2f}"
    )
    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]not a tty; re-run with --yes to authorise the spend[/red]")
            return 2
        if input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            return 1

    budget = BudgetGuard(limit_usd=args.budget)
    report = Report()
    started = time.time()

    for index, row in enumerate(rows, start=1):
        if budget.exceeded:
            console.print("[yellow]budget ceiling reached; stopping[/yellow]")
            break
        if args.delay and index > 1:
            # Pacing beats retrying: a 429 costs the request and then a wait
            # long enough to clear the window anyway.
            time.sleep(args.delay)
        outcome = judge(row, budget)
        report.outcomes.append(outcome)
        if outcome.error:
            report.errors.append(f"{outcome.fixture_id}: {outcome.error}")
            mark = "[red]ERR [/red]"
        else:
            mark = "[green]ok  [/green]" if outcome.correct else "[red]miss[/red]"
        console.print(
            f"{mark} {index:>2}/{len(rows)} {outcome.expected:7} "
            f"{(outcome.mutation_type or '-'):13} {outcome.fixture_id[-38:]}"
        )

    elapsed = time.time() - started
    summary = print_report(report, budget, elapsed)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_path = args.out / f"critic-{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "fixtures_file": str(args.fixtures),
                **summary,
                "outcomes": [o.__dict__ for o in report.outcomes],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"report: {out_path}")

    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
