"""Seed critic fixtures from real accepted runs, then break them on purpose.

Measuring a judge needs cases where the right answer is already known. This
script builds them from runs the Critic itself accepted:

  * the accepted output, recorded with ``expected_verdict: accept``;
  * several mutated copies, each violating exactly one acceptance criterion,
    recorded with ``expected_verdict: revise`` plus the criterion it breaks and
    how it was broken.

The mutations are written by **Claude, never Gemini**, and that is the whole
point. Gemini is the Critic. A test suite authored by the model under test
inherits its blind spots: whatever Gemini cannot see, it also cannot corrupt in
a way it would later notice. The same cross-vendor rule that puts the Critic on
a different vendor from the Worker puts the fixture author on a different
vendor from the Critic.

Everything written here is a *draft*: ``reviewed: false`` on every row. A
generated mutation is a hypothesis about ground truth, not ground truth. The
usual failure is a mutation that breaks a second criterion by accident -- the
suite would then blame the Critic for a fixture's mistake. A human ticks
``reviewed`` after checking that exactly one criterion moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.table import Table

from budget import BudgetGuard
from providers import anthropic
from providers.retry_utils import ProviderError

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "evals" / "critic_fixtures_draft.jsonl"

# Every mutation type the generator understands. Not all of them break the
# output: `benign` is a control, and it is the most informative row in the file
# -- it is the only one that can catch a Critic rejecting work for looking
# different rather than for failing anything.
MUTATION_TYPES = ("quantitative", "language", "omission", "tone", "factual", "benign")

# Mutations that leave every criterion satisfied. Their fixtures expect accept,
# not revise.
BENIGN_TYPES = ("benign",)

# What the automatic rotation actually cycles through. `tone` is deliberately
# absent: these criteria almost never constrain register, so a tone mutation
# either breaks nothing (a fixture that would mark a correct Critic wrong) or
# the model reaches for a subject change and breaks two criteria at once. Both
# outcomes were observed on the first real batch and both were thrown away in
# review. It stays available via --types for a task whose criteria do talk
# about voice.
ROTATION_TYPES = ("quantitative", "language", "omission", "factual", "benign")

# Factual is the one that must always be present. It is the hardest for a judge
# -- nothing about the shape of the text gives it away, only the content does --
# so a suite without it flatters the Critic.
REQUIRED_TYPES = ("factual",)

# One mutation per run, rotating the type, rather than five mutations from one
# run. Same row count, far better spread: five mutations of one output share
# its subject, its length and its register, so a Critic that happens to handle
# that one text well scores five easy passes. Spreading the types across
# different outputs makes each type's pass rate mean something on its own.

MUTATION_GUIDE = """\
quantitative -- change a number, count or length so a measurable limit is
    breached (e.g. push a 35-word answer past a 40-word cap).
language  -- switch the language, or mix a second language in, so a
    language criterion fails.
omission  -- delete a required element (a term, a section, a mandatory
    mention) while leaving the rest intact.
tone      -- shift register or add the jargon/marketing voice a criterion
    forbids, without changing the facts.
factual   -- keep the length, language, structure and tone identical and
    change a claim so it is wrong. Nothing about the shape of the text
    should betray it; only the content is false.
benign    -- rewrite the surface and break NOTHING. Reorder clauses, swap
    words for synonyms, resplit or rejoin phrasing -- but only where no
    criterion constrains it, and re-check every criterion afterwards. If a
    criterion fixes the sentence count, keep that exact count; if it requires
    a word, keep that word; if it caps the length, stay under it. The result
    must still satisfy every single criterion. This is a control case: it
    tests whether the judge rejects work merely for looking different."""

console = Console()


class Mutation(BaseModel):
    """One deliberately broken copy of an accepted output."""

    mutation_type: str = Field(
        description="One of: quantitative, language, omission, tone, factual."
    )
    broken_criterion_index: int = Field(
        description="The 1-based number of the single criterion this violates, from the list given."
    )
    broken_criterion: str = Field(
        description="That criterion's text, copied verbatim. Used only as a cross-check."
    )
    mutated_output: str = Field(
        description="The full rewritten output. Complete text, not a diff or a description."
    )
    explanation: str = Field(
        description="Why this violates that criterion and why every other criterion still holds."
    )


class MutationSet(BaseModel):
    mutations: list[Mutation] = Field(default_factory=list)


@dataclass
class RunSource:
    """An accepted run, reduced to what a fixture needs."""

    run_id: str
    path: str
    goal: str
    worker_prompt: str
    criteria: list[dict[str, Any]]
    output: str
    criteria_format: str
    bad_lines: int = 0


@dataclass
class Stats:
    runs_seen: int = 0
    runs_used: int = 0
    accept_rows: int = 0
    mutation_rows: int = 0
    rejected: list[str] = field(default_factory=list)
    by_type: dict[str, int] = field(default_factory=dict)


# --- reading the run logs --------------------------------------------------


def _normalise_criteria(raw: Any) -> tuple[list[dict[str, Any]], str]:
    """Accept both criteria shapes this project has used.

    Runs recorded before criteria became objects hold plain strings. They are
    still perfectly good fixture material -- the text is what the Critic judges
    -- so they are promoted rather than discarded. ``critical`` defaults to
    True because a legacy run had no notion of an optional criterion, and
    guessing "optional" would quietly weaken any fixture built from it.
    """
    if not isinstance(raw, list) or not raw:
        return [], "none"
    if isinstance(raw[0], dict):
        return [c for c in raw if c.get("text")], "structured"
    return (
        [{"text": str(c), "critical": True, "check_method": ""} for c in raw if str(c).strip()],
        "legacy",
    )


def read_accepted_runs(directory: Path, include_evals: bool) -> list[RunSource]:
    """Every run the Critic accepted, newest first.

    Only ``accepted`` counts -- not ``accepted_by_user``. A run the user waved
    through is precisely a case where the criteria were *not* all met, so it
    cannot serve as an example of an output that should be accepted.
    """
    sources: list[RunSource] = []
    for path in sorted(directory.glob("*.jsonl"), reverse=True):
        if not include_evals and path.stem.startswith("eval-"):
            continue
        events, bad = [], 0
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(event, dict):
                events.append(event)
            else:
                bad += 1

        end = next((e for e in events if e.get("event") == "run_end"), None)
        if not end or end.get("status") != "accepted":
            continue
        plans = [e for e in events if e.get("event") == "manager_plan"]
        outputs = [e for e in events if e.get("event") == "worker_output" and e.get("ok")]
        if not plans or not outputs:
            continue

        plan = plans[-1].get("plan") or {}
        criteria, fmt = _normalise_criteria(plan.get("acceptance_criteria"))
        output = (outputs[-1].get("result") or "").strip()
        if not criteria or not output:
            continue

        start = next((e for e in events if e.get("event") == "run_start"), {})
        sources.append(
            RunSource(
                run_id=path.stem,
                path=str(path),
                goal=(start.get("task") or {}).get("goal", ""),
                worker_prompt=plan.get("worker_prompt", ""),
                criteria=criteria,
                output=output,
                criteria_format=fmt,
                bad_lines=bad,
            )
        )
    return sources


# --- asking Claude to break things -----------------------------------------


SYSTEM = """\
You build test cases for an automated judge by breaking correct outputs.

You are given an output that satisfies every acceptance criterion, and the list
of criteria. Produce mutated copies. Each mutated copy must:

1. Violate EXACTLY ONE criterion -- identified by its number in
   broken_criterion_index, with its text echoed in broken_criterion.
2. Keep every other criterion satisfied. This is the hard part and the part
   that matters: a mutation that breaks two criteria is useless as a test case.
3. Be a complete, natural-looking output. Never annotate the text, never mark
   the change, never explain inside the output itself. A judge must not be able
   to spot the mutation by its formatting.
4. Stay in the same language as the original, unless the mutation type is
   'language' -- that one changes the language deliberately.

Mutation types:
{guide}

Return one mutation per requested type, in the order requested."""


def request_mutations(
    source: RunSource,
    types: Iterable[str],
    model: str,
    max_tokens: int,
) -> tuple[list[Mutation], list[str], int, int]:
    """One structured call to Claude.

    Returns (mutations, per-item problems, input tokens, output tokens). Items
    are validated one at a time on purpose: a single malformed entry -- an
    empty index, a missing field -- would otherwise invalidate the whole batch
    and throw away four good mutations we already paid for.
    """
    criteria_block = "\n".join(
        f"{i}. {c['text']}"
        + (f"\n   (how it is checked: {c['check_method']})" if c.get("check_method") else "")
        for i, c in enumerate(source.criteria, start=1)
    )
    wanted = list(types)
    user = (
        f"TASK GIVEN TO THE WORKER:\n{source.worker_prompt or source.goal}\n\n"
        f"ACCEPTANCE CRITERIA (all currently satisfied):\n{criteria_block}\n\n"
        f"ACCEPTED OUTPUT:\n{source.output}\n\n"
        f"Produce exactly {len(wanted)} mutations, one of each of these types, "
        f"in this order: {', '.join(wanted)}."
    )
    result = anthropic.call_structured(
        MutationSet,
        system=SYSTEM.format(guide=MUTATION_GUIDE),
        user=user,
        model=model,
        max_tokens=max_tokens,
    )
    payload = _unwrap(result.data)
    raw_items = payload.get("mutations")
    if not isinstance(raw_items, list):
        raise ValueError(f"no mutations list in the reply ({type(raw_items).__name__})")

    mutations: list[Mutation] = []
    problems: list[str] = []
    for position, raw in enumerate(raw_items, start=1):
        try:
            mutations.append(Mutation.model_validate(raw))
        except ValidationError as exc:
            fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors())
            problems.append(f"mutation {position} unusable ({fields})")
    return mutations, problems, result.input_tokens, result.output_tokens


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """Recover a payload whose list arrived as a JSON string.

    Even under a forced tool call, a model occasionally serialises a nested
    structure into the string field instead of building it. The call is already
    paid for, so parse what came back rather than throw it away.
    """
    mutations = data.get("mutations")
    if not isinstance(mutations, str):
        return data
    try:
        decoded = json.loads(mutations)
    except json.JSONDecodeError:
        return data
    if isinstance(decoded, dict) and isinstance(decoded.get("mutations"), list):
        return {**data, "mutations": decoded["mutations"]}
    if isinstance(decoded, list):
        return {**data, "mutations": decoded}
    return data


# --- machine-checkable criteria --------------------------------------------
#
# A benign mutation is only useful if it really breaks nothing, and "the model
# said it kept every criterion" is not evidence. These patterns cover the
# criteria this system actually produces -- counts, required words, forbidden
# phrases, language -- and are checked in code before a benign fixture is
# allowed to claim expected_verdict: accept.
#
# Deliberately partial: a criterion this cannot parse is left alone rather than
# guessed at. It fails closed in the direction that matters -- a violation it
# can prove discards the fixture; a criterion it cannot read simply is not
# vouched for.

_MAX_WORDS = re.compile(
    r"(?:fewer than|less than|under|at most|no more than|maximum of|up to)\s+(\d+)\s+words",
    re.IGNORECASE,
)
# Counts are written both ways in practice -- "exactly 3 lines" and "exactly
# three lines" -- and a pattern that reads only digits silently vouches for a
# criterion it never checked.
_COUNT = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_EXACT_WORDS = re.compile(rf"exactly\s+{_COUNT}\s+words", re.IGNORECASE)
_EXACT_LINES = re.compile(rf"exactly\s+{_COUNT}\s+(?:non-empty\s+)?lines", re.IGNORECASE)
_EXACT_SENTENCES = re.compile(rf"exactly\s+{_COUNT}\s+sentences?", re.IGNORECASE)
_CONTAINS_WORD = re.compile(
    r"contains?\s+(?:the\s+)?(?:word|term|phrase)\s+[\"'“‘]?([\w' -]+?)[\"'”’]?\s*(?:\(|,|\.|$)",
    re.IGNORECASE,
)
_ABSENT_PHRASE = re.compile(
    r"(?:does not contain|must not contain|contains no|without)\s+(?:the\s+)?"
    r"(?:word|term|phrase)\s+[\"'“‘]?([\w' -]+?)[\"'”’]?\s*(?:\(|,|\.|$)",
    re.IGNORECASE,
)
_ENGLISH = re.compile(r"\b(?:in|written in)\s+English\b", re.IGNORECASE)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_count(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)
# Latin-script languages give themselves away by function words far more
# reliably than by diacritics, which English borrows freely.
_NON_ENGLISH_MARKERS = re.compile(
    r"\b(?:le|la|les|des|une|dans|avec|pour|ensuite|il|elle|est|sont"
    r"|el|los|las|una|con|para|pero|puede|cuando|que|del"
    r"|der|die|das|und|nicht|mit|auch)\b",
    re.IGNORECASE,
)


def _count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+(?:\s|$)", text.strip()) if part.strip()])


def _looks_non_english(text: str) -> bool:
    """True only when the evidence is unambiguous.

    Non-Latin script is decisive. For Latin-script languages, one stray
    borrowed word is not enough -- require several function words that English
    does not use, so an English sentence containing "que" or "la" in a quote
    does not trip it.
    """
    if re.search(r"[\u0600-\u06ff\u0400-\u04ff\u4e00-\u9fff\u3040-\u30ff]", text):
        return True
    return len(_NON_ENGLISH_MARKERS.findall(text)) >= 3


def machine_violations(output: str, criteria: list[dict[str, Any]]) -> list[str]:
    """Criteria this can check mechanically and finds violated.

    Returns a human-readable reason per violation, so a discarded fixture says
    why rather than just disappearing.
    """
    problems: list[str] = []
    words = len(output.split())
    lines = len([line for line in output.splitlines() if line.strip()])

    for criterion in criteria:
        text = criterion.get("text", "")
        if not text:
            continue

        match = _MAX_WORDS.search(text)
        if match and words >= int(match.group(1)):
            problems.append(f"{words} words breaches '{text[:50]}'")

        for pattern, actual, unit in (
            (_EXACT_WORDS, words, "words"),
            (_EXACT_LINES, lines, "lines"),
            (_EXACT_SENTENCES, _count_sentences(output), "sentences"),
        ):
            match = pattern.search(text)
            if not match:
                continue
            expected = _parse_count(match.group(1))
            if expected is not None and actual != expected:
                problems.append(f"{actual} {unit} is not exactly {expected}")

        # Absence patterns first: "does not contain the word X" also matches
        # the contains-pattern, and reading it as a requirement would invert it.
        absent = _ABSENT_PHRASE.search(text)
        if absent:
            needle = absent.group(1).strip()
            if needle and re.search(rf"\b{re.escape(needle)}\b", output, re.IGNORECASE):
                problems.append(f"contains the forbidden {needle!r}")
        else:
            required = _CONTAINS_WORD.search(text)
            if required:
                needle = required.group(1).strip()
                if needle and not re.search(rf"\b{re.escape(needle)}\b", output, re.IGNORECASE):
                    problems.append(f"missing the required {needle!r}")

        if _ENGLISH.search(text) and _looks_non_english(output):
            problems.append("does not read as English")

    return problems


def _normalise_text(value: str) -> str:
    """Whitespace and trailing punctuation only -- never a fuzzy match.

    Recovering 'fewer than 80 words.' from 'fewer than 80 words' is safe.
    Guessing which criterion 'keep it short' meant is not, so it is not tried.
    """
    return " ".join(value.split()).strip().rstrip(".").casefold()


def validate(mutation: Mutation, source: RunSource, stats: Stats) -> Mutation | None:
    """Drop anything that cannot serve as ground truth.

    Fail closed: a fixture that is wrong is worse than a fixture that is
    missing, because the suite will blame the Critic for it.
    """
    if mutation.mutation_type not in MUTATION_TYPES:
        stats.rejected.append(f"{source.run_id}: unknown type {mutation.mutation_type!r}")
        return None

    if mutation.mutation_type in BENIGN_TYPES:
        # A benign rewrite names no criterion: it is defined by breaking none.
        # Anything the model echoed in broken_criterion is noise here.
        mutation.broken_criterion = ""
        body = mutation.mutated_output.strip()
        if not body:
            stats.rejected.append(f"{source.run_id}: empty benign output")
            return None
        if body == source.output.strip():
            stats.rejected.append(f"{source.run_id}: benign output was unchanged")
            return None

        # The claim "this breaks nothing" is checked, not taken on trust. A
        # benign fixture that quietly violates a criterion would teach the
        # suite to punish a Critic for being right -- the most damaging kind of
        # wrong row in the file, and the hardest to spot later.
        violations = machine_violations(body, source.criteria)
        if violations:
            stats.rejected.append(
                f"{source.run_id}: benign rewrite is not benign -- {'; '.join(violations)}"
            )
            return None

        mutation.mutated_output = body
        return mutation

    # The index is authoritative and the echoed text is a cross-check, not the
    # other way round. Asking a model to reproduce a sentence character for
    # character invites paraphrase; asking it for a number does not. Fixtures
    # whose criterion we cannot pin down exactly are dropped rather than
    # guessed at -- this file is meant to be ground truth.
    index = mutation.broken_criterion_index
    if 1 <= index <= len(source.criteria):
        mutation.broken_criterion = source.criteria[index - 1]["text"]
    else:
        texts = {_normalise_text(c["text"]): c["text"] for c in source.criteria}
        named = _normalise_text(mutation.broken_criterion)
        if named not in texts:
            stats.rejected.append(
                f"{source.run_id}: criterion index {index} out of range and text unmatched "
                f"({mutation.broken_criterion[:60]!r})"
            )
            return None
        mutation.broken_criterion = texts[named]

    body = mutation.mutated_output.strip()
    if not body:
        stats.rejected.append(f"{source.run_id}: empty mutated output")
        return None
    if body == source.output.strip():
        stats.rejected.append(f"{source.run_id}: {mutation.mutation_type} output was unchanged")
        return None
    mutation.mutated_output = body
    return mutation


# --- fixture rows ----------------------------------------------------------


def accept_row(source: RunSource, generator: str) -> dict[str, Any]:
    return {
        "fixture_id": f"{source.run_id}::accept",
        "source_run": source.run_id,
        "source_log": source.path,
        "goal": source.goal,
        "worker_prompt": source.worker_prompt,
        "acceptance_criteria": source.criteria,
        "criteria_format": source.criteria_format,
        "output": source.output,
        "expected_verdict": "accept",
        "broken_criterion": None,
        "mutation_type": None,
        "mutation_explanation": None,
        "generator_model": None,  # this row is real, not generated
        "generated_at": datetime.now(UTC).isoformat(),
        "reviewed": False,
    }


def mutation_row(
    source: RunSource, mutation: Mutation, index: int, generator: str
) -> dict[str, Any]:
    """A fixture row for one mutation.

    The expected verdict follows from the type, not from the model's opinion:
    a benign rewrite still satisfies every criterion, so the Critic is supposed
    to accept it. Getting this backwards would train the suite to reward
    exactly the behaviour it exists to catch.
    """
    # The digest is what keeps ids unique across --append runs. Without it a
    # second batch reproduces "<run>::<type>::1" and the two rows become
    # indistinguishable downstream -- which silently cost a good fixture the
    # first time this file was reviewed.
    digest = hashlib.sha1(mutation.mutated_output.encode("utf-8")).hexdigest()[:8]
    return {
        "fixture_id": f"{source.run_id}::{mutation.mutation_type}::{index}::{digest}",
        "source_run": source.run_id,
        "source_log": source.path,
        "goal": source.goal,
        "worker_prompt": source.worker_prompt,
        "acceptance_criteria": source.criteria,
        "criteria_format": source.criteria_format,
        "output": mutation.mutated_output,
        "expected_verdict": "accept" if mutation.mutation_type in BENIGN_TYPES else "revise",
        "broken_criterion": mutation.broken_criterion or None,
        "mutation_type": mutation.mutation_type,
        "mutation_explanation": mutation.explanation,
        "generator_model": generator,
        "generated_at": datetime.now(UTC).isoformat(),
        "reviewed": False,
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Build draft critic fixtures from accepted runs")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=10, help="how many accepted runs to mine")
    parser.add_argument("--per-run", type=int, default=1, help="mutations to request per run")
    parser.add_argument(
        "--budget", type=float, default=1.00, help="dollar ceiling for the whole job"
    )
    parser.add_argument("--model", default=anthropic.DEFAULT_MODEL)
    parser.add_argument(
        "--types", default=None,
        help=(
            f"comma-separated subset of: {', '.join(MUTATION_TYPES)}. "
            f"Defaults to the rotation ({', '.join(ROTATION_TYPES)}); 'tone' is "
            "available but not rotated -- see ROTATION_TYPES."
        ),
    )
    parser.add_argument("--exclude-evals", action="store_true", help="skip eval-suite runs")
    parser.add_argument(
        "--append", action="store_true", help="add to the draft instead of replacing it"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be mined; no API calls"
    )
    parser.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args = parser.parse_args(argv)

    # An explicit --types is an instruction, not a suggestion: the required-type
    # guarantee below applies to the default rotation, not to a targeted top-up
    # someone asked for by name.
    explicit_types = args.types is not None
    types = [t.strip() for t in (args.types or ",".join(ROTATION_TYPES)).split(",") if t.strip()]
    unknown = [t for t in types if t not in MUTATION_TYPES]
    if unknown:
        console.print(f"[red]unknown mutation type(s): {', '.join(unknown)}[/red]")
        return 2
    # The factual case is the one worth guaranteeing; see REQUIRED_TYPES.
    if not explicit_types:
        for required in REQUIRED_TYPES:
            if required not in types:
                types.append(required)
                console.print(f"[yellow]added the required '{required}' mutation type[/yellow]")

    sources = read_accepted_runs(args.runs, include_evals=not args.exclude_evals)
    stats = Stats(runs_seen=len(sources))
    if not sources:
        console.print(f"[red]no accepted runs in {args.runs}[/red]")
        console.print("run `make run` or `make eval` first -- fixtures are mined from real runs")
        return 1
    selected = sources[: args.limit]

    table = Table(title="Runs to mine", header_style="bold")
    table.add_column("Run")
    table.add_column("Criteria", justify="right")
    table.add_column("Format")
    table.add_column("Output")
    for source in selected:
        table.add_row(
            source.run_id[:34], str(len(source.criteria)), source.criteria_format,
            source.output[:40].replace("\n", " ") + "…",
        )
    console.print(table)
    # Round-robin over the type list. With the defaults (10 runs, 1 each, five
    # types) every type lands exactly twice.
    assignments = {
        source.run_id: [
            types[(index * args.per_run + offset) % len(types)]
            for offset in range(max(1, args.per_run))
        ]
        for index, source in enumerate(selected)
    }
    spread: dict[str, int] = {}
    for wanted in assignments.values():
        for mutation_type in wanted:
            spread[mutation_type] = spread.get(mutation_type, 0) + 1
    benign_planned = sum(
        1 for wanted in assignments.values() for t in wanted if t in BENIGN_TYPES
    )
    breaking_planned = len(selected) * args.per_run - benign_planned
    console.print(
        f"{len(selected)} run(s) × {args.per_run} mutation(s) = {len(selected)} Claude call(s), "
        f"{len(selected) + benign_planned} accept + {breaking_planned} revise rows"
    )
    console.print("planned spread: " + ", ".join(f"{k}×{v}" for k, v in spread.items()))

    if args.dry_run:
        console.print("[yellow]dry run: no API calls, nothing written[/yellow]")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        console.print("[red]ANTHROPIC_API_KEY is not set; mutations are written by Claude[/red]")
        return 2
    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]not a tty; re-run with --yes to authorise the spend[/red]")
            return 2
        answer = input(f"proceed, ceiling ${args.budget:.2f}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            console.print("aborted")
            return 1

    budget = BudgetGuard(limit_usd=args.budget)
    rows: list[dict[str, Any]] = []

    for source in selected:
        if budget.exceeded:
            console.print(
                f"[yellow]budget ceiling reached; stopping before {source.run_id}[/yellow]"
            )
            break
        console.rule(source.run_id[:60])
        rows.append(accept_row(source, args.model))
        stats.accept_rows += 1

        wanted = assignments[source.run_id]
        try:
            mutations, problems, tin, tout = request_mutations(
                source, wanted, args.model, max_tokens=8000
            )
        except (ProviderError, ValueError) as exc:
            # One unusable run must not cost the ones already generated.
            stats.rejected.append(f"{source.run_id}: {exc}")
            console.print(f"[red]{source.run_id}: {exc}[/red]")
            continue
        entry = budget.charge(f"{source.run_id}.mutate", args.model, tin, tout)
        stats.rejected.extend(f"{source.run_id}: {note}" for note in problems)

        kept: list[Mutation] = []
        seen_outputs = {source.output.strip()}
        for mutation in mutations:
            checked = validate(mutation, source, stats)
            if checked is None:
                continue
            if checked.mutated_output in seen_outputs:
                stats.rejected.append(f"{source.run_id}: duplicate {checked.mutation_type} output")
                continue
            seen_outputs.add(checked.mutated_output)
            kept.append(checked)

        for index, mutation in enumerate(kept, start=1):
            rows.append(mutation_row(source, mutation, index, args.model))
            stats.mutation_rows += 1
            stats.by_type[mutation.mutation_type] = stats.by_type.get(mutation.mutation_type, 0) + 1

        stats.runs_used += 1
        console.print(
            f"{len(kept)} mutation(s): {', '.join(m.mutation_type for m in kept) or '—'} "
            f"· ${entry.cost_usd:.4f}"
        )

    # The required types are guaranteed here rather than per run, because with
    # one mutation per run most runs are not even asked for them. Enforced in
    # code and not left to the prompt: a suite missing the hardest mutation
    # type reports a Critic that looks better than it is.
    for required in REQUIRED_TYPES if not explicit_types else ():
        if stats.by_type.get(required) or budget.exceeded:
            continue
        for source in selected:
            console.print(
                f"[yellow]no {required} mutation yet; asking {source.run_id[:40]}[/yellow]"
            )
            try:
                retry, problems, tin, tout = request_mutations(
                    source, [required], args.model, max_tokens=4000
                )
            except (ProviderError, ValueError) as exc:
                stats.rejected.append(f"{source.run_id}: {required} retry failed -- {exc}")
                continue
            budget.charge(f"{source.run_id}.retry", args.model, tin, tout)
            stats.rejected.extend(f"{source.run_id}: retry {note}" for note in problems)
            recovered = [m for m in (validate(m, source, stats) for m in retry) if m]
            if recovered:
                mutation = recovered[0]
                rows.append(mutation_row(source, mutation, 99, args.model))
                stats.mutation_rows += 1
                stats.by_type[required] = stats.by_type.get(required, 0) + 1
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.out.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    benign_rows = sum(stats.by_type.get(t, 0) for t in BENIGN_TYPES)
    summary = Table(show_header=False, title="Draft fixtures")
    summary.add_row("accept rows (original)", str(stats.accept_rows))
    summary.add_row("accept rows (benign)", str(benign_rows))
    summary.add_row("revise rows", str(stats.mutation_rows - benign_rows))
    for mutation_type in MUTATION_TYPES:
        summary.add_row(f"  {mutation_type}", str(stats.by_type.get(mutation_type, 0)))
    summary.add_row("rejected", str(len(stats.rejected)))
    summary.add_row("cost", f"${budget.spent_usd:.4f} of ${args.budget:.2f}")
    summary.add_row("written to", str(args.out))
    console.print(summary)

    for note in stats.rejected[:10]:
        console.print(f"[yellow]dropped: {note}[/yellow]")

    console.print(
        "\n[bold]Every row is reviewed: false.[/bold] A generated mutation is a hypothesis "
        "about ground truth. Check that each one breaks its named criterion and nothing "
        "else, then flip the flag."
    )
    missing_required = [t for t in REQUIRED_TYPES if not stats.by_type.get(t)]
    if missing_required:
        console.print(f"[red]no {', '.join(missing_required)} fixtures were produced[/red]")
        return 1
    return 0 if stats.mutation_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
