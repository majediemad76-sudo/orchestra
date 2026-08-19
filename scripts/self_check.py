"""Offline verification -- no API key, no network, no cost.

Covers the parts that fail silently and expensively: a schema dialect the
vendor rejects at request time, arithmetic that lets a run overspend, and
escalation rules that only misbehave on the unhappy path.

The state machine is exercised against fake roles rather than mocked HTTP. The
point is not that the providers were called correctly -- it is that the loop
stops when it should, escalates when it should, and does not ask a human
anything it was not supposed to ask.

Deliberately dependency-free: no pytest, no fixtures. It is one command that
either prints "all good" or names what broke, which is what makes it usable as
a pre-commit gate on a machine with nothing installed but the venv.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from providers.schema_utils import (  # noqa: E402
    resolve_refs,
    to_anthropic_schema,
    to_gemini_schema,
    to_xai_schema,
)
from schemas import (  # noqa: E402
    CriterionCheck,
    CriticVerdict,
    Escalation,
    ManagerPlan,
    Question,
    Task,
    WorkerOutput,
)

FAILURES: List[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILURES.append(label)


def walk(node: Any):
    """Yield every dict in a nested schema."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def check_pydantic_models() -> None:
    print("\n[1] Pydantic models")
    plan_without_optional = ManagerPlan(
        plan="p",
        worker_prompt="do the thing",
        acceptance_criteria=["under 150 words"],
        worker_type="text",
    )
    check(plan_without_optional.question is None, "ManagerPlan builds without the optional field")
    check(plan_without_optional.needs_user_input is False, "needs_user_input defaults to False")

    plan_with_optional = ManagerPlan(
        plan="p",
        worker_prompt="do the thing",
        acceptance_criteria=["under 150 words"],
        worker_type="code",
        needs_user_input=True,
        question=Question(text="which one?", options=["a", "b", "c"]),
    )
    check(plan_with_optional.question is not None, "ManagerPlan builds with the optional field")

    passing = CriterionCheck(criterion="a", passed=True, evidence="quoted", reason="because")
    failing = CriterionCheck(criterion="b", passed=False, evidence="quoted", reason="because")
    check(CriticVerdict(checks=[passing], verdict="accept").score == 100, "CriticVerdict builds")
    check(
        CriterionCheck(criterion="c", passed=True, evidence="x" * 300, reason="r").evidence.__len__() == 200,
        "CriterionCheck truncates evidence instead of rejecting it",
    )

    # The score is arithmetic, not a model output -- these are the rules the
    # accept decision now rests on.
    check(CriticVerdict(checks=[passing, failing], verdict="accept").score == 50, "score is the pass ratio")
    check(
        CriticVerdict(checks=[passing, passing, failing], verdict="revise").score == 67,
        "score rounds to the nearest whole percent",
    )
    check(CriticVerdict(checks=[passing, failing], verdict="accept").all_passed is False,
          "one failed criterion means not accepted, whatever the model said")
    check(CriticVerdict(checks=[passing, passing], verdict="revise").all_passed is True,
          "all criteria passing means accepted, whatever the model said")
    check(CriticVerdict(checks=[], verdict="accept").all_passed is False,
          "an empty check list fails closed")
    check(CriticVerdict(checks=[], verdict="accept").score == 0, "no checks scores 0, not 100")
    mixed = CriticVerdict(checks=[passing, failing], verdict="revise")
    check(mixed.met_criteria == ["a"] and mixed.failed_criteria == ["b"], "met/failed derive from checks")
    record = mixed.as_record()
    check(
        {"checks", "score", "all_passed", "met_criteria", "failed_criteria"} <= set(record),
        "as_record writes the derived values into the log",
    )
    check(WorkerOutput(result="x").ok is True, "WorkerOutput builds")
    check(Task(goal="g").max_rounds == 4, "Task builds with defaults")
    check(
        Escalation(
            trigger="budget_exceeded",
            reason="over budget",
            question=Question(text="what now?", options=["stop", "raise"]),
        ).trigger
        == "budget_exceeded",
        "Escalation builds",
    )

    for bad_options in ([], ["only-one"], ["a", "b", "c", "d", "e"]):
        try:
            Question(text="q", options=bad_options)
            check(False, f"Question rejects {len(bad_options)} options")
        except Exception:
            check(True, f"Question rejects {len(bad_options)} options")


def check_anthropic() -> None:
    print("\n[2] Anthropic converter (forced tool input_schema)")
    for model in (ManagerPlan, CriticVerdict, WorkerOutput):
        schema = to_anthropic_schema(model)
        blob = json.dumps(schema)
        check("$ref" not in blob, f"{model.__name__}: no $ref left")
        check("$defs" not in blob, f"{model.__name__}: no $defs left")
        check(schema.get("type") == "object", f"{model.__name__}: top level is an object")
    verdict_schema = to_anthropic_schema(CriticVerdict)
    check(
        "criterion" in json.dumps(verdict_schema["properties"]["checks"]),
        "CriticVerdict.checks inlines CriterionCheck",
    )
    nested = to_anthropic_schema(ManagerPlan)["properties"]["question"]
    check(
        "properties" in json.dumps(nested),
        "ManagerPlan.question is inlined, not referenced",
    )


def check_xai() -> None:
    print("\n[3] xAI converter (strict json_schema)")
    envelope = to_xai_schema(ManagerPlan)
    check(envelope["type"] == "json_schema", "envelope type is json_schema")
    check(envelope["json_schema"]["strict"] is True, "strict is true")
    schema = envelope["json_schema"]["schema"]
    check("$ref" not in json.dumps(schema), "no $ref left")

    objects = [n for n in walk(schema) if n.get("type") == "object" or "properties" in n]
    check(len(objects) >= 2, "found the nested object too")
    check(
        all(node.get("additionalProperties") is False for node in objects),
        "every object has additionalProperties: false",
    )
    check(
        all(set(node.get("required", [])) == set(node.get("properties", {})) for node in objects),
        "strict mode: every property is listed in required",
    )
    question = schema["properties"]["question"]
    check("null" in json.dumps(question["type"]), "optional field is typed nullable, not omitted")

    nested = to_xai_schema(CriticVerdict)["json_schema"]["schema"]["properties"]["checks"]["items"]
    check(nested.get("additionalProperties") is False,
          "objects inside an array also get additionalProperties: false")
    check(set(nested.get("required", [])) == set(nested.get("properties", {})),
          "objects inside an array also list every property in required")


def check_gemini() -> None:
    print("\n[4] Gemini converter (responseSchema)")
    schema = to_gemini_schema(ManagerPlan)
    check("$ref" not in json.dumps(schema), "no $ref left")
    types = [node["type"] for node in walk(schema) if "type" in node]
    check(bool(types), "types are present")
    check(all(t == t.upper() for t in types), "every type name is upper-case")
    check(schema["type"] == "OBJECT", "top level is OBJECT")
    check(schema["properties"]["acceptance_criteria"]["type"] == "ARRAY", "list maps to ARRAY")
    check(schema["properties"]["acceptance_criteria"]["items"]["type"] == "STRING", "list items map to STRING")
    check(schema["properties"]["question"].get("nullable") is True, "Optional -> nullable: true")
    check(
        schema["properties"]["needs_user_input"].get("nullable") is True,
        "field with a default -> nullable: true",
    )
    check("nullable" not in schema["properties"]["plan"], "required field is not nullable")
    check(schema["properties"]["worker_type"]["enum"] == ["text", "code"], "Literal maps to enum")

    check(to_gemini_schema(Task)["properties"]["max_rounds"]["type"] == "INTEGER", "int maps to INTEGER")
    check(to_gemini_schema(WorkerOutput)["properties"]["ok"]["type"] == "BOOLEAN", "bool maps to BOOLEAN")

    # CriticVerdict is the only schema with a list of nested models, which is
    # the shape most likely to break a dialect conversion.
    verdict = to_gemini_schema(CriticVerdict)
    checks = verdict["properties"]["checks"]
    check(checks["type"] == "ARRAY" and checks["items"]["type"] == "OBJECT",
          "list of nested models maps to ARRAY of OBJECT")
    check(checks["items"]["properties"]["passed"]["type"] == "BOOLEAN",
          "nested bool maps to BOOLEAN")
    check("score" not in verdict["properties"],
          "the derived score is never asked of the model")


def check_ref_resolution() -> None:
    print("\n[5] $ref resolution")
    raw = ManagerPlan.model_json_schema()
    check("$defs" in json.dumps(raw), "pydantic really did emit $defs (the test is meaningful)")
    resolved = resolve_refs(raw)
    check("$ref" not in json.dumps(resolved) and "$defs" not in resolved, "resolve_refs inlines everything")


def check_budget() -> None:
    print("\n[6] Budget guard")
    from budget import PRICING, BudgetGuard, price

    check(set(PRICING) == {"claude-sonnet-5", "grok-4.6", "gemini-3.1-flash-lite"}, "all three models priced")
    check(abs(price("claude-sonnet-5", 1_000_000, 1_000_000) - 12.0) < 1e-9, "sonnet: 2 in + 10 out = 12")
    check(abs(price("grok-4.6", 1_000_000, 1_000_000) - 8.0) < 1e-9, "grok: 2 in + 6 out = 8")
    check(abs(price("gemini-3.1-flash-lite", 1_000_000, 1_000_000) - 1.75) < 1e-9, "gemini: 0.25 in + 1.5 out = 1.75")

    guard = BudgetGuard(limit_usd=0.01)
    guard.charge("m", "grok-4.6", 1000, 1000)
    check(not guard.exceeded, "under the ceiling")
    guard.charge_usd("w", "claude-code-headless", 0.02)
    check(guard.exceeded, "crossing the ceiling is detected")
    check(len(guard.entries) == 2 and guard.summary()["calls"] == 2, "every call is recorded")


def _fake_ask(index: int):
    def ask(question: Question) -> Optional[int]:
        assert 2 <= len(question.options) <= 4, "escalation must offer 2-4 options"
        return index

    return ask


def check_controller_state_machine() -> None:
    """Drive the real loop with fake roles.

    ``controller.run_task`` runs unmodified -- only the three role functions are
    swapped. A rewritten test copy of the loop would pass while the real one
    was broken, which is the one thing this file exists to prevent.
    """
    print("\n[7] Controller state machine (fake roles)")
    import controller
    from providers import ProviderResult

    saved = (controller.manager_role.plan, controller.worker_role.execute, controller.critic_role.review)
    tmp = Path(tempfile.mkdtemp(prefix="orchestrator-selfcheck-"))
    saved_runs = controller.RUNS_DIR
    controller.RUNS_DIR = tmp

    calls: Dict[str, int] = {"manager": 0, "worker": 0, "critic": 0}

    def fake_plan(task, previous_plan=None, verdict=None, worker_result="", user_answer="", **_):
        calls["manager"] += 1
        needs_input = task.context == "ask" and calls["manager"] == 1
        plan = ManagerPlan(
            plan="fake plan",
            worker_prompt="write something",
            acceptance_criteria=["c1"],
            worker_type="text",
            needs_user_input=needs_input,
            question=Question(text="which tone?", options=["formal", "casual"]) if needs_input else None,
        )
        return plan, ProviderResult(data={}, model="grok-4.6", input_tokens=100, output_tokens=100)

    def fake_worker(plan, cwd=None):
        calls["worker"] += 1
        return controller.worker_role.WorkerRun(
            output=WorkerOutput(result="fake result"),
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=100,
        )

    def make_critic(passes: bool):
        def fake_review(plan, output):
            calls["critic"] += 1
            return (
                CriticVerdict(
                    checks=[
                        CriterionCheck(
                            criterion="c1",
                            passed=passes,
                            evidence="fake result",
                            reason="because this is a fake",
                        )
                    ],
                    fix_instruction="" if passes else "shorten it",
                    # Deliberately contradicts the checks: the Controller must
                    # follow the checks, not this field.
                    verdict="accept",
                ),
                ProviderResult(data={}, model="gemini-3.1-flash-lite", input_tokens=100, output_tokens=100),
            )

        return fake_review

    try:
        # Baseline: a good score ends the run immediately.
        controller.manager_role.plan = fake_plan
        controller.worker_role.execute = fake_worker
        controller.critic_role.review = make_critic(True)
        summary = controller.run_task(Task(goal="g"), ask=_fake_ask(0), run_id="selfcheck-accept")
        check(summary["status"] == "accepted", "accepts when every criterion passes")
        check(summary["rounds"] == 1, "stops as soon as it is accepted")
        check(summary["budget"]["spent_usd"] > 0, "cost is accumulated")
        lines = [json.loads(l) for l in (tmp / "selfcheck-accept.jsonl").read_text(encoding="utf-8").splitlines()]
        events = {line["event"] for line in lines}
        check(
            {"run_start", "manager_plan", "worker_output", "critic_verdict", "run_end"} <= events,
            "the JSONL log records every stage",
        )

        # Trigger 1. The escalation must fire on the second rejection --
        # firing later means budget was spent that the rule exists to save.
        calls.update({"manager": 0, "worker": 0, "critic": 0})
        controller.critic_role.review = make_critic(False)
        summary = controller.run_task(
            Task(goal="g", max_rounds=6), ask=_fake_ask(2), run_id="selfcheck-reject"
        )
        check(summary["status"] == "accepted_by_user", "two rejections escalate to the user")
        check(summary["rounds"] == 2, "the escalation fires on the second rejection, not later")
        check(
            [e["trigger"] for e in summary["escalations"]] == ["two_rejections"],
            "the escalation is tagged two_rejections",
        )

        # Trigger 2. The answer must reach the Manager, and the round must not
        # be counted against the cap.
        calls.update({"manager": 0, "worker": 0, "critic": 0})
        controller.critic_role.review = make_critic(True)
        summary = controller.run_task(
            Task(goal="g", context="ask"), ask=_fake_ask(1), run_id="selfcheck-ask"
        )
        check(
            [e["trigger"] for e in summary["escalations"]] == ["manager_needs_input"],
            "needs_user_input escalates to the user",
        )
        check(summary["status"] == "accepted", "the run continues after the answer")
        check(calls["manager"] == 2, "the manager re-plans with the user's answer")

        # Trigger 3. A ceiling too small for two rounds is crossed by round 2,
        # where the top-of-loop check catches it before spending more.
        calls.update({"manager": 0, "worker": 0, "critic": 0})
        controller.critic_role.review = make_critic(False)
        summary = controller.run_task(
            Task(goal="g", budget_usd=0.0005, max_rounds=6), ask=_fake_ask(2), run_id="selfcheck-budget"
        )
        check(
            summary["escalations"][0]["trigger"] == "budget_exceeded",
            "blowing the budget escalates to the user",
        )
        check(summary["status"] == "stopped_by_user", "the user's stop choice is obeyed")

        # The failure that would hurt most in CI: an unanswerable question in
        # a non-interactive run must halt, not spin.
        calls.update({"manager": 0, "worker": 0, "critic": 0})
        summary = controller.run_task(
            Task(goal="g", budget_usd=0.0005, max_rounds=6),
            ask=lambda q: None,
            run_id="selfcheck-noanswer",
        )
        check(summary["status"] == "stopped_by_user", "an unanswered escalation stops the run")
    finally:
        controller.manager_role.plan, controller.worker_role.execute, controller.critic_role.review = saved
        controller.RUNS_DIR = saved_runs


def check_no_hardcoded_keys() -> None:
    print("\n[8] No hard-coded keys")
    import re

    sources = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
    literal = re.compile(r"""(sk-[A-Za-z0-9_-]{12,}|xai-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,})""")
    offenders = [str(p.relative_to(ROOT)) for p in sources if literal.search(p.read_text(encoding="utf-8"))]
    check(not offenders, f"no key literal in any source file{(' -> ' + ', '.join(offenders)) if offenders else ''}")

    for var in ("XAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        used = any(var in p.read_text(encoding="utf-8") for p in sources if p.parts[-2] == "providers")
        check(used, f"{var} is read from the environment in providers/")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    check(
        all(f"{v}=" in example for v in ("XAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")),
        ".env.example lists all three keys",
    )
    check(all(line.split("=", 1)[1].strip() == "" for line in example.strip().splitlines()), ".env.example has no values")
    check(not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text(), ".env is git-ignored")


def main() -> int:
    print("self check -- no API keys required, no network calls")
    check_pydantic_models()
    check_anthropic()
    check_xai()
    check_gemini()
    check_ref_resolution()
    check_budget()
    check_controller_state_machine()
    check_no_hardcoded_keys()

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
