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
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, get_args

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import controller as controller_module
from keys import ApiKeys
from providers.schema_utils import (
    resolve_refs,
    to_anthropic_schema,
    to_gemini_schema,
    to_xai_schema,
)
from schemas import (
    Criterion,
    CriterionCheck,
    CriticVerdict,
    Escalation,
    ManagerPlan,
    Question,
    Task,
    WorkerOutput,
)

FAILURES: list[str] = []
CHECKS = 0

# Values that look like credentials to every code path that handles them, and
# are worth nothing if they escape. Every offline check that reaches a function
# now demanding an ApiKeys uses these -- a real key must never be needed to run
# the gate, and a check that silently fell back to the environment would pass on
# a developer machine and fail in CI.
CANARY_KEY = "sk-CANARY-9f3a"
FAKE_KEYS = ApiKeys(xai=CANARY_KEY, anthropic=CANARY_KEY, google=CANARY_KEY)

# Invented credentials in the three shapes the vendors actually issue, used to
# prove the redactor catches a key it was never handed. Listed here, in one
# place, because the hard-coded-key scan has to know which strings in this file
# are deliberate props -- and the alternative, exempting this whole file, would
# switch the scan off exactly where test values live.
UNHELD_KEY_SHAPES = (
    "sk-ant-api03-AbCdEfGh12345678",
    "xai-ZZZZaaaa1111bbbb",
    "AIzaSyD-aaaa1111bbbb",
)
FAKE_CREDENTIALS = (CANARY_KEY, *UNHELD_KEY_SHAPES)

CRITICAL_CRITERION = Criterion(
    text="under 150 words", critical=True, check_method="count the words"
)
OPTIONAL_CRITERION = Criterion(
    text="mentions the release date", critical=False, check_method="search for a date"
)


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
        acceptance_criteria=[CRITICAL_CRITERION],
        worker_type="text",
    )
    check(plan_without_optional.question is None, "ManagerPlan builds without the optional field")
    check(plan_without_optional.needs_user_input is False, "needs_user_input defaults to False")

    plan_with_optional = ManagerPlan(
        plan="p",
        worker_prompt="do the thing",
        acceptance_criteria=[CRITICAL_CRITERION],
        worker_type="code",
        needs_user_input=True,
        question=Question(text="which one?", options=["a", "b", "c"]),
    )
    check(plan_with_optional.question is not None, "ManagerPlan builds with the optional field")

    passing = CriterionCheck(criterion="a", passed=True, evidence="quoted", reason="because")
    failing = CriterionCheck(criterion="b", passed=False, evidence="quoted", reason="because")
    check(CriticVerdict(checks=[passing], verdict="accept").score == 100, "CriticVerdict builds")
    check(
        len(CriterionCheck(criterion="c", passed=True, evidence="x" * 300, reason="r").evidence)
        == 200,
        "CriterionCheck truncates evidence instead of rejecting it",
    )

    # The score is arithmetic, not a model output -- these are the rules the
    # accept decision now rests on.
    check(
        CriticVerdict(checks=[passing, failing], verdict="accept").score == 50,
        "score is the pass ratio",
    )
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

    # Criterion + the critical-gated accept rule.
    check(
        ManagerPlan(plan="p", worker_prompt="w", acceptance_criteria=[CRITICAL_CRITERION],
                    worker_type="text").acceptance_criteria[0].check_method == "count the words",
        "ManagerPlan carries structured criteria",
    )
    crit_pass = CriterionCheck(criterion="a", passed=True, critical=True, evidence="q", reason="r")
    crit_fail = CriterionCheck(criterion="a", passed=False, critical=True, evidence="q", reason="r")
    opt_pass = CriterionCheck(criterion="b", passed=True, critical=False, evidence="q", reason="r")
    opt_fail = CriterionCheck(criterion="b", passed=False, critical=False, evidence="q", reason="r")

    check(CriticVerdict(checks=[crit_pass, opt_pass], verdict="revise").accepted is True,
          "everything passing is accepted")
    check(CriticVerdict(checks=[crit_pass, opt_fail], verdict="revise").accepted is True,
          "a non-critical failure does not block acceptance")
    check(CriticVerdict(checks=[crit_fail, opt_pass], verdict="accept").accepted is False,
          "a critical failure blocks acceptance, whatever the model said")
    check(CriticVerdict(checks=[opt_fail, opt_pass], verdict="accept").accepted is False,
          "with nothing marked critical, every criterion must pass")
    check(CriticVerdict(checks=[], verdict="accept").accepted is False,
          "an empty check list is never accepted")
    check(CriticVerdict(checks=[crit_fail, opt_fail], verdict="revise").blocking_failures == ["a"],
          "blocking_failures lists only the critical ones")
    check(CriticVerdict(checks=[crit_pass, opt_fail], verdict="revise").score == 50,
          "the score still counts every criterion, critical or not")

    # The Critic must not be able to reclassify a criterion it just failed.
    from roles.critic import _reassert_criticality

    plan = ManagerPlan(plan="p", worker_prompt="w", worker_type="text",
                       acceptance_criteria=[CRITICAL_CRITERION, OPTIONAL_CRITERION])
    tampered = CriticVerdict(
        checks=[
            CriterionCheck(criterion=CRITICAL_CRITERION.text, passed=False, critical=False,
                           evidence="q", reason="r"),
            CriterionCheck(criterion=OPTIONAL_CRITERION.text, passed=False, critical=True,
                           evidence="q", reason="r"),
        ],
        verdict="accept",
    )
    restored = _reassert_criticality(tampered, plan)
    check(restored.checks[0].critical is True and restored.checks[1].critical is False,
          "criticality is restored from the plan, not taken from the Critic")
    check(restored.accepted is False, "a downgraded critical failure still blocks acceptance")

    paraphrased = CriticVerdict(
        checks=[CriterionCheck(criterion="under 150 words (roughly)", passed=False,
                               critical=False, evidence="q", reason="r")],
        verdict="accept",
    )
    check(_reassert_criticality(paraphrased, plan).checks[0].critical is True,
          "a paraphrased criterion falls back to its position in the plan")

    invented = CriticVerdict(
        checks=[
            CriterionCheck(criterion=CRITICAL_CRITERION.text, passed=True, critical=True,
                           evidence="q", reason="r"),
            CriterionCheck(criterion=OPTIONAL_CRITERION.text, passed=True, critical=False,
                           evidence="q", reason="r"),
            CriterionCheck(criterion="something the Manager never asked for", passed=False,
                           critical=False, evidence="q", reason="r"),
        ],
        verdict="accept",
    )
    check(_reassert_criticality(invented, plan).checks[2].critical is True,
          "a criterion with no counterpart in the plan is treated as critical")
    mixed = CriticVerdict(checks=[passing, failing], verdict="revise")
    check(
        mixed.met_criteria == ["a"] and mixed.failed_criteria == ["b"],
        "met/failed derive from checks",
    )
    record = mixed.as_record()
    check(
        {"checks", "score", "all_passed", "met_criteria", "failed_criteria"} <= set(record),
        "as_record writes the derived values into the log",
    )
    check(WorkerOutput(result="x").ok is True, "WorkerOutput builds")
    check(Task(goal="g").max_rounds == 4, "Task builds with defaults")
    check(
        set(get_args(controller_module.EscalationTrigger))
        == set(get_args(Escalation.model_fields["trigger"].annotation)),
        "the Controller's trigger literal matches the Escalation schema",
    )
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
    check(
        schema["properties"]["acceptance_criteria"]["items"]["properties"]["critical"]["type"]
        == "BOOLEAN",
        "nested Criterion fields map through the array",
    )
    check(schema["properties"]["question"].get("nullable") is True, "Optional -> nullable: true")
    check(
        schema["properties"]["needs_user_input"].get("nullable") is True,
        "field with a default -> nullable: true",
    )
    check("nullable" not in schema["properties"]["plan"], "required field is not nullable")
    check(schema["properties"]["worker_type"]["enum"] == ["text", "code"], "Literal maps to enum")

    check(
        to_gemini_schema(Task)["properties"]["max_rounds"]["type"] == "INTEGER",
        "int maps to INTEGER",
    )
    check(
        to_gemini_schema(WorkerOutput)["properties"]["ok"]["type"] == "BOOLEAN",
        "bool maps to BOOLEAN",
    )

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
    check(
        "$ref" not in json.dumps(resolved) and "$defs" not in resolved,
        "resolve_refs inlines everything",
    )


def check_budget() -> None:
    print("\n[6] Budget guard")
    from budget import PRICING, BudgetGuard, price

    check(
        set(PRICING) == {"claude-sonnet-5", "grok-4.6", "gemini-3.1-flash-lite"},
        "all three models priced",
    )
    check(
        abs(price("claude-sonnet-5", 1_000_000, 1_000_000) - 12.0) < 1e-9,
        "sonnet: 2 in + 10 out = 12",
    )
    check(abs(price("grok-4.6", 1_000_000, 1_000_000) - 8.0) < 1e-9, "grok: 2 in + 6 out = 8")
    check(
        abs(price("gemini-3.1-flash-lite", 1_000_000, 1_000_000) - 1.75) < 1e-9,
        "gemini: 0.25 in + 1.5 out = 1.75",
    )

    guard = BudgetGuard(limit_usd=0.01)
    guard.charge("m", "grok-4.6", 1000, 1000)
    check(not guard.exceeded, "under the ceiling")
    guard.charge_usd("w", "claude-code-headless", 0.02)
    check(guard.exceeded, "crossing the ceiling is detected")
    check(len(guard.entries) == 2 and guard.summary()["calls"] == 2, "every call is recorded")


def _fake_ask(index: int):
    def ask(question: Question) -> int | None:
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

    saved = (
        controller.manager_role.plan,
        controller.worker_role.execute,
        controller.critic_role.review,
    )
    tmp = Path(tempfile.mkdtemp(prefix="orchestrator-selfcheck-"))
    saved_runs = controller.RUNS_DIR
    controller.RUNS_DIR = tmp

    calls: dict[str, int] = {"manager": 0, "worker": 0, "critic": 0}

    def fake_plan(task, previous_plan=None, verdict=None, worker_result="", user_answer="", **_):
        calls["manager"] += 1
        needs_input = task.context == "ask" and calls["manager"] == 1
        plan = ManagerPlan(
            plan="fake plan",
            worker_prompt="write something",
            acceptance_criteria=[CRITICAL_CRITERION],
            worker_type="text",
            needs_user_input=needs_input,
            question=Question(text="which tone?", options=["formal", "casual"])
            if needs_input
            else None,
        )
        return plan, ProviderResult(data={}, model="grok-4.6", input_tokens=100, output_tokens=100)

    def fake_worker(plan, cwd=None, **_):
        calls["worker"] += 1
        return controller.worker_role.WorkerRun(
            output=WorkerOutput(result="fake result"),
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=100,
        )

    def make_critic(passes: bool):
        def fake_review(plan, output, **_):
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
                ProviderResult(
                    data={}, model="gemini-3.1-flash-lite", input_tokens=100, output_tokens=100
                ),
            )

        return fake_review

    try:
        # Baseline: a good score ends the run immediately.
        controller.manager_role.plan = fake_plan
        controller.worker_role.execute = fake_worker
        controller.critic_role.review = make_critic(True)
        summary = controller.run_task(
            Task(goal="g"), ask=_fake_ask(0), run_id="selfcheck-accept", keys=FAKE_KEYS
        )
        check(summary["status"] == "accepted", "accepts when every criterion passes")
        check(summary["rounds"] == 1, "stops as soon as it is accepted")
        check(summary["budget"]["spent_usd"] > 0, "cost is accumulated")
        log_text = (tmp / "selfcheck-accept.jsonl").read_text(encoding="utf-8")
        lines = [json.loads(line) for line in log_text.splitlines()]
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
            Task(goal="g", max_rounds=6),
            ask=_fake_ask(2),
            run_id="selfcheck-reject",
            keys=FAKE_KEYS,
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
            Task(goal="g", context="ask"),
            ask=_fake_ask(1),
            run_id="selfcheck-ask",
            keys=FAKE_KEYS,
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
            Task(goal="g", budget_usd=0.0005, max_rounds=6),
            ask=_fake_ask(2),
            run_id="selfcheck-budget",
            keys=FAKE_KEYS,
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
            keys=FAKE_KEYS,
        )
        check(summary["status"] == "stopped_by_user", "an unanswered escalation stops the run")
    finally:
        (
            controller.manager_role.plan,
            controller.worker_role.execute,
            controller.critic_role.review,
        ) = saved
        controller.RUNS_DIR = saved_runs


def check_fixture_builder() -> None:
    """The three ways the fixture generator lost paid-for work.

    Every one of these was a live failure, not a hypothetical: mutations were
    generated, billed, and thrown away. They are cheap to re-break and
    expensive to notice, so they are pinned here.
    """
    print("\n[9] Fixture builder (scripts/make_fixtures.py)")
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_fixtures as mf

    source = mf.RunSource(
        run_id="r", path="p", goal="g", worker_prompt="w",
        criteria=[
            {
                "text": "The output consists of exactly one sentence.",
                "critical": True,
                "check_method": "",
            },
            {"text": "fewer than 17 words", "critical": True, "check_method": ""},
        ],
        output="the original text", criteria_format="structured",
    )
    stats = mf.Stats()

    def mutation(**kwargs):
        base = dict(mutation_type="factual", broken_criterion_index=1,
                    broken_criterion="x", mutated_output="changed", explanation="e")
        base.update(kwargs)
        return mf.Mutation(**base)

    # 1. The criterion is resolved by index; a paraphrased echo cannot mislabel it.
    resolved = mf.validate(mutation(broken_criterion_index=2, broken_criterion="Use one sentence."),
                           source, stats)
    check(resolved is not None and resolved.broken_criterion == "fewer than 17 words",
          "the criterion index wins over a paraphrased echo")
    check(
        mf.validate(mutation(broken_criterion_index=99, broken_criterion=" Fewer than 17 words. "),
                    source, stats).broken_criterion == "fewer than 17 words",
        "an out-of-range index falls back to normalised text",
    )
    check(mf.validate(mutation(broken_criterion_index=99, broken_criterion="keep it short"),
                      source, stats) is None,
          "an unmatched paraphrase is dropped, never guessed at")
    check(mf.validate(mutation(mutated_output="the original text"), source, stats) is None,
          "a mutation that changed nothing is dropped")
    check(mf.validate(mutation(mutation_type="vibes"), source, stats) is None,
          "an unknown mutation type is dropped")

    # 2. A payload whose list arrived as a JSON string is recovered.
    item = {"mutation_type": "factual", "broken_criterion_index": 1,
            "broken_criterion": "c", "mutated_output": "o", "explanation": "e"}
    for shape in (
        {"mutations": json.dumps({"mutations": [item]})},
        {"mutations": json.dumps([item])},
        {"mutations": [item]},
    ):
        unwrapped = mf._unwrap(shape)
        check(isinstance(unwrapped["mutations"], list) and len(unwrapped["mutations"]) == 1,
              "a stringified mutations payload is recovered")
    check(mf._unwrap({"mutations": "not json at all"})["mutations"] == "not json at all",
          "an unparseable payload is left alone rather than mangled")

    # 3. One malformed entry must not discard the batch it arrived with.
    from unittest.mock import patch

    from providers import ProviderResult

    payload = {
        "mutations": [
            item,
            {**item, "broken_criterion_index": ""},
            {**item, "mutation_type": "tone"},
        ]
    }
    with patch.object(mf.anthropic, "call_structured",
                      return_value=ProviderResult(data=payload, model="claude-sonnet-5",
                                                  input_tokens=10, output_tokens=10)):
        kept, problems, _, _ = mf.request_mutations(
            source, ["factual"], "claude-sonnet-5", 100, keys=FAKE_KEYS
        )
    check(len(kept) == 2 and len(problems) == 1,
          "one malformed mutation is skipped, the rest of the batch survives")

    # Accepted runs only -- a run the user waved through is not an accept case.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    def write_log(name, status):
        events = [
            {"event": "run_start", "task": {"goal": "g"}},
            {"event": "manager_plan", "round": 1, "plan": {"worker_prompt": "w",
             "acceptance_criteria": [{"text": "c", "critical": True, "check_method": "m"}]}},
            {"event": "worker_output", "round": 1, "ok": True, "result": "the output"},
            {"event": "run_end", "status": status},
        ]
        (tmp / name).write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    write_log("20260101-000000-a.jsonl", "accepted")
    write_log("20260101-000001-b.jsonl", "accepted_by_user")
    write_log("20260101-000002-c.jsonl", "max_rounds")
    found = mf.read_accepted_runs(tmp, include_evals=True)
    check([f.run_id for f in found] == ["20260101-000000-a"],
          "only Critic-accepted runs become accept fixtures")

    # Benign mutations are machine-verified before they may claim "accept".
    benign_criteria = [
        {"text": "The output has fewer than 17 words."},
        {"text": "The output consists of exactly one sentence."},
        {"text": "Output contains exactly three non-empty lines and no other lines."},
        {"text": "The output contains the word task (case-insensitive)."},
        {"text": "The output does not contain the phrase the thing that (case-insensitive)."},
        {"text": "The output is in English."},
    ]
    one_line = [c for c in benign_criteria if "lines" not in c["text"]]

    check(
        mf.machine_violations("The system takes a task and then performs it.", one_line) == [],
        "a genuinely benign rewrite passes the machine check",
    )
    check(
        bool(mf.machine_violations("The system " + "very " * 20 + "takes a task.", one_line)),
        "a word-count breach is caught",
    )
    check(
        bool(mf.machine_violations("The system takes a task. It performs it.", one_line)),
        "a sentence-count breach is caught",
    )
    check(
        bool(mf.machine_violations("The system takes it and performs it.", one_line)),
        "a missing required word is caught",
    )
    check(
        bool(mf.machine_violations("The system takes the thing that is a task.", one_line)),
        "a forbidden phrase is caught",
    )
    check(
        bool(
            mf.machine_violations(
                "Le système prend une tâche dans la file avec le modèle.", one_line
            )
        ),
        "a language switch is caught",
    )
    check(
        bool(mf.machine_violations("a\nb", [c for c in benign_criteria if "lines" in c["text"]])),
        "a line-count breach is caught, spelled out or in digits",
    )
    check(
        mf.machine_violations("anything at all", [{"text": "The tone should feel welcoming."}])
        == [],
        "a criterion it cannot parse is left alone rather than guessed at",
    )

    benign = mf.Mutation(
        mutation_type="benign", broken_criterion_index=0, broken_criterion="",
        mutated_output="The system takes a task. It then performs it.", explanation="e",
    )
    benign_source = mf.RunSource(
        run_id="r", path="p", goal="g", worker_prompt="w",
        criteria=[{"text": "The output consists of exactly one sentence.", "critical": True,
                   "check_method": "count the sentences"}],
        output="The system takes a task and performs it.", criteria_format="structured",
    )
    check(
        mf.validate(benign, benign_source, mf.Stats()) is None,
        "a benign mutation that violates a criterion is discarded, never saved as accept",
    )
    benign.mutated_output = "The system accepts a task and carries it out."
    check(
        mf.validate(benign, benign_source, mf.Stats()) is not None,
        "a benign mutation that violates nothing is kept",
    )

    # Legacy string criteria are promoted rather than discarded.
    legacy, fmt = mf._normalise_criteria(["under 150 words", "no jargon"])
    check(
        fmt == "legacy"
        and legacy[0] == {"text": "under 150 words", "critical": True, "check_method": ""},
        "legacy string criteria are promoted to critical, not dropped",
    )


def check_critic_harness() -> None:
    """Negative control, run entirely offline against hand-built verdicts.

    A suite reporting 100% is either measuring a good judge or measuring
    nothing at all. The only way to tell them apart is to feed the grader an
    answer key that is wrong everywhere and confirm the score collapses.

    No provider is called and nothing is stubbed at the network layer: the
    ``CriticVerdict`` objects below are constructed here and handed straight to
    ``eval_critic.grade_outcome``, which is the same function the live run
    uses. The inverted keys exist only in this function -- the fixture file on
    disk never contains a wrong expectation.
    """
    print("\n[11] Critic harness (offline negative control)")
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_critic as ec

    def invert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flip every expected verdict. Defined here, and only here.

        This used to live in eval_critic with a --negative-control flag, which
        meant the live harness could be pointed at the real Critic with a
        deliberately wrong answer key: real money, a report file that looks
        like a genuine evaluation, and every row marked wrong on purpose.
        Inverting is a property of the *test*, not of the tool, so it stays in
        the test.
        """
        return [
            {
                **row,
                "expected_verdict": "revise" if row["expected_verdict"] == "accept" else "accept",
            }
            for row in rows
        ]

    check(ec.is_correct("accept", accepted=True), "accept fixture + accepted = correct")
    check(not ec.is_correct("accept", accepted=False), "accept fixture + rejected = wrong")
    check(ec.is_correct("revise", accepted=False), "revise fixture + rejected = correct")
    check(not ec.is_correct("revise", accepted=True), "revise fixture + accepted = wrong")

    def verdict(passed: bool) -> CriticVerdict:
        """A synthetic verdict: one critical criterion, met or not."""
        return CriticVerdict(
            checks=[
                CriterionCheck(
                    criterion="c1",
                    passed=passed,
                    critical=True,
                    evidence="quoted from the output",
                    reason="synthetic",
                )
            ],
            fix_instruction="" if passed else "fix it",
            # Deliberately always "accept": the grader must follow the checks,
            # never this field.
            verdict="accept",
        )

    good = {
        "fixture_id": "f-accept",
        "expected_verdict": "accept",
        "mutation_type": None,
        "broken_criterion": None,
    }
    broken = {
        "fixture_id": "f-revise",
        "expected_verdict": "revise",
        "mutation_type": "factual",
        "broken_criterion": "c1",
    }

    # The truthful key: a Critic that passes good work and fails broken work.
    check(
        ec.grade_outcome(good, verdict(passed=True)).correct,
        "truthful key: accept case scores correct",
    )
    check(
        ec.grade_outcome(broken, verdict(passed=False)).correct,
        "truthful key: revise case scores correct",
    )

    # The same verdicts against an inverted key must all be marked wrong.
    flipped = invert([good, broken])
    check(
        [f["expected_verdict"] for f in flipped] == ["revise", "accept"],
        "invert() flips every expected verdict",
    )
    check(
        [f["expected_verdict"] for f in [good, broken]] == ["accept", "revise"],
        "invert() copies rather than mutating the fixtures it was given",
    )
    inverted_outcomes = [
        ec.grade_outcome(flipped[0], verdict(passed=True)),
        ec.grade_outcome(flipped[1], verdict(passed=False)),
    ]
    check(
        not any(o.correct for o in inverted_outcomes),
        "inverted key: the grader reports failure on every case",
    )
    check(
        all(o.accepted is not None for o in inverted_outcomes),
        "the verdict itself is unchanged -- only the expectation moved",
    )

    # A judge that rejects everything and one that accepts everything must not
    # be able to score the same, which is the whole reason the two rates are
    # reported separately.
    always_accept = [
        ec.grade_outcome(good, verdict(True)),
        ec.grade_outcome(broken, verdict(True)),
    ]
    always_reject = [
        ec.grade_outcome(good, verdict(False)),
        ec.grade_outcome(broken, verdict(False)),
    ]
    check(
        [o.correct for o in always_accept] == [True, False],
        "a judge that accepts everything fails the revise case",
    )
    check(
        [o.correct for o in always_reject] == [False, True],
        "a judge that rejects everything fails the accept case",
    )

    # Naming the right criterion is tracked separately from the verdict.
    caught = ec.grade_outcome(broken, verdict(passed=False))
    check(
        caught.caught_named_criterion is True,
        "the named criterion is credited when it failed",
    )
    missed = ec.grade_outcome(
        {**broken, "broken_criterion": "some other criterion"}, verdict(passed=False)
    )
    check(
        missed.correct and missed.caught_named_criterion is False,
        "rejecting for the wrong reason counts as a rejection but not as a catch",
    )
    check(
        ec.grade_outcome(good, verdict(passed=True)).caught_named_criterion is None,
        "a fixture that names no criterion is not scored on naming one",
    )


def check_critic_null_handling() -> None:
    """A null anywhere in the Critic's reply must not take down the run.

    Gemini sends null for fields its own schema marked nullable. Stripping only
    the top level looked sufficient until a live run died on
    ``checks[6].critical = None`` -- two levels down, one fixture lost, the
    whole verdict rejected. Every one of these has cost a real call, so each
    depth is pinned here.
    """
    print("\n[13] Critic null handling")
    from unittest.mock import patch

    from providers import ProviderResult
    from roles import critic as critic_role

    check(critic_role.drop_nulls({"a": None, "b": 1}) == {"b": 1}, "top-level nulls are dropped")
    check(
        critic_role.drop_nulls({"a": {"b": None, "c": 2}}) == {"a": {"c": 2}},
        "nulls one level down are dropped",
    )
    check(
        critic_role.drop_nulls({"checks": [{"critical": None, "passed": True}]})
        == {"checks": [{"passed": True}]},
        "nulls inside a list of objects are dropped -- the case that killed a live run",
    )
    check(
        critic_role.drop_nulls({"checks": [None, {"passed": True}]})
        == {"checks": [{"passed": True}]},
        "a null list element is dropped rather than validated",
    )
    check(
        critic_role.drop_nulls({"a": 0, "b": "", "c": False, "d": []})
        == {"a": 0, "b": "", "c": False, "d": []},
        "falsy values that are not null survive",
    )

    plan = ManagerPlan(
        plan="p",
        worker_prompt="w",
        acceptance_criteria=[
            Criterion(text="c1", critical=True, check_method="m"),
            Criterion(text="c2", critical=False, check_method="m"),
        ],
        worker_type="text",
    )
    payload = {
        "checks": [
            {
                "criterion": "c1",
                "passed": True,
                "critical": None,        # the field that killed the live run
                "evidence": "quoted",
                "reason": "because",
            },
            {
                "criterion": "c2",
                "passed": False,
                "critical": True,
                "evidence": "quoted",
                "reason": None,          # a null two levels down, different field
            },
        ],
        "fix_instruction": None,          # the null that bit first, at the top level
        "verdict": "accept",
    }
    with patch.object(
        critic_role.google,
        "call_structured",
        return_value=ProviderResult(
            data=payload, model="gemini-3.1-flash-lite", input_tokens=10, output_tokens=10
        ),
    ):
        verdict, _ = critic_role.review(plan, WorkerOutput(result="anything"), keys=FAKE_KEYS)
    check(len(verdict.checks) == 2, "a verdict carrying nested nulls still parses")
    check(verdict.fix_instruction == "", "a null fix_instruction falls back to its default")
    check(verdict.checks[1].reason == "", "a null reason falls back to its default")
    check(
        verdict.checks[0].critical is True and verdict.checks[1].critical is False,
        "criticality is still restored from the plan, not from the stripped null",
    )
    check(verdict.accepted is True, "the surviving verdict is still gradeable")

    # A check with no criterion or no verdict cannot be graded; it is dropped
    # rather than guessed at, and the empty-list rule then fails closed.
    salvaged = critic_role._repair({"checks": [{"criterion": "c1"}, {"passed": True}]})
    check(salvaged["checks"] == [], "a check missing criterion or passed is dropped, not invented")
    kept = critic_role._repair({"checks": [{"criterion": "c1", "passed": True}]})
    check(
        kept["checks"][0]["reason"] == "" and kept["checks"][0]["evidence"] == "",
        "descriptive fields are backfilled so a paid verdict is not lost",
    )
    check(
        critic_role._repair({"checks": [{"criterion": "c", "passed": True, "reason": "kept"}]})
        ["checks"][0]["reason"] == "kept",
        "a reason that was actually sent is never overwritten",
    )


def check_fixture_file_integrity() -> None:
    """The committed fixture file must never contain an inverted expectation.

    The negative control inverts expectations in memory, for one function call.
    If an inverted row ever reached the file on disk it would look identical to
    a real fixture and would silently mark a correct Critic wrong for as long
    as nobody noticed. These invariants are cheap; that failure is not.
    """
    print("\n[12] Fixture file integrity")
    path = ROOT / "evals" / "critic_fixtures.jsonl"
    if not path.exists():
        check(True, "no fixture file yet -- nothing to verify")
        return

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_fixtures as mf

    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    check(bool(rows), "the fixture file has rows")
    check(
        all(r["expected_verdict"] in {"accept", "revise"} for r in rows),
        "every expected_verdict is accept or revise",
    )
    check(all(r.get("reviewed") for r in rows), "every fixture is marked reviewed")
    check(
        len({r["fixture_id"] for r in rows}) == len(rows),
        "fixture ids are unique",
    )
    check(
        all(r["expected_verdict"] == "accept" for r in rows if r.get("mutation_type") == "benign"),
        "every benign row expects accept -- a benign row expecting revise is an inverted row",
    )
    check(
        all(
            r["expected_verdict"] == "revise"
            for r in rows
            if r.get("mutation_type") and r["mutation_type"] not in mf.BENIGN_TYPES
        ),
        "every breaking mutation expects revise",
    )
    check(
        all(r.get("broken_criterion") for r in rows if r["expected_verdict"] == "revise"),
        "every revise row names the criterion it breaks",
    )
    check(
        all(
            r["broken_criterion"] in {c["text"] for c in r["acceptance_criteria"]}
            for r in rows
            if r["expected_verdict"] == "revise"
        ),
        "every named criterion exists in that fixture's criteria",
    )
    benign = [r for r in rows if r.get("mutation_type") == "benign"]
    check(
        all(not mf.machine_violations(r["output"], r["acceptance_criteria"]) for r in benign),
        "every benign row still passes the machine check",
    )


def check_retry_policy() -> None:
    """A provider that says 'retry in 30s' must not be second-guessed.

    The exponential backoff alone tops out below a rate-limit window, so a run
    would exhaust its attempts inside a window that was never going to open.
    This cost four fixtures on the first live grading run.
    """
    print("\n[10] Retry policy")
    import httpx

    from providers.retry_utils import (
        MAX_RETRY_WAIT,
        ProviderError,
        is_retryable,
        parse_retry_after,
        wait_policy,
    )

    gemini_429 = (
        '{"error":{"code":429,"message":"Quota exceeded. Please retry in 30.6s.","status":"X"}}'
    )
    check(parse_retry_after(httpx.Response(429, text=gemini_429)) == 30.6,
          "the delay is read out of a Gemini 429 body")
    check(parse_retry_after(httpx.Response(429, text='{"retryDelay": "31s"}')) == 31.0,
          "the delay is read out of a retryDelay field")
    check(parse_retry_after(httpx.Response(429, headers={"retry-after": "12"}, text="{}")) == 12.0,
          "the standard Retry-After header is read")
    check(parse_retry_after(httpx.Response(500, text="boom")) is None,
          "a response with no stated delay yields None")

    class _State:
        def __init__(self, exc, attempt=1):
            class _Outcome:
                def exception(inner_self):
                    return exc
            self.outcome = _Outcome()
            self.attempt_number = attempt
            self.idle_for = 0.0

    hinted = wait_policy(_State(ProviderError("google", "x", 429, retry_after=30.6)))
    check(31 <= hinted <= 32, "a stated delay is waited out, with a little slack")
    check(
        wait_policy(_State(ProviderError("google", "x", 429, retry_after=9999))) == MAX_RETRY_WAIT,
        "an absurd delay is capped rather than obeyed",
    )
    check(wait_policy(_State(ProviderError("google", "x", 500), attempt=3)) > 0,
          "with no stated delay it falls back to exponential backoff")

    # A daily allowance and a per-minute limit arrive as the same 429. Treating
    # them alike left one run blocked for half an hour on a quota that would
    # not reopen until the next day.
    from providers.retry_utils import QuotaExhausted, looks_like_daily_quota, raise_for_status

    minute_body = (
        '{"error":{"code":429,"message":"Quota exceeded for metric: '
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15. "
        'Please retry in 30.6s."}}'
    )
    daily_body = (
        '{"error":{"code":429,"message":"Quota exceeded","details":[{"violations":'
        '[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}'
    )
    check(
        not looks_like_daily_quota(minute_body, 30.6),
        "a per-minute 429 is not mistaken for a daily one",
    )
    check(
        looks_like_daily_quota(daily_body, None),
        "a daily quota is recognised from the quota name in the body",
    )
    check(
        looks_like_daily_quota('{"error":"429"}', 3600.0),
        "a delay longer than any minute window is recognised as a daily quota",
    )
    try:
        raise_for_status("google", httpx.Response(429, text=daily_body))
        check(False, "a daily quota raises QuotaExhausted")
    except QuotaExhausted as exc:
        check(True, "a daily quota raises QuotaExhausted")
        check(not is_retryable(exc), "QuotaExhausted is never retried -- waiting cannot help")
    except Exception:
        check(False, "a daily quota raises QuotaExhausted")
    try:
        raise_for_status("google", httpx.Response(429, text=minute_body))
        check(False, "a per-minute 429 still raises the retryable ProviderError")
    except QuotaExhausted:
        check(False, "a per-minute 429 still raises the retryable ProviderError")
    except ProviderError as exc:
        check(is_retryable(exc), "a per-minute 429 still raises the retryable ProviderError")

    check(is_retryable(ProviderError("google", "x", 429)), "429 is retryable")
    check(is_retryable(ProviderError("google", "x", 503)), "5xx is retryable")
    check(not is_retryable(ProviderError("google", "x", 400)), "400 is not retryable")
    check(not is_retryable(ProviderError("google", "x", 401)), "401 is not retryable")


def check_no_hardcoded_keys() -> None:
    print("\n[8] No hard-coded keys")
    import re

    sources = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
    literal = re.compile(
        r"""(sk-[A-Za-z0-9_-]{12,}|xai-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,})"""
    )
    # The deliberate fakes in this file are removed before the scan rather than
    # being allowed to slip under the length threshold: relying on "it happens
    # not to match" means the check silently stops covering this file the day
    # one of them gets longer.
    def without_props(text: str) -> str:
        for prop in FAKE_CREDENTIALS:
            text = text.replace(prop, "")
        return text

    offenders = [
        str(p.relative_to(ROOT))
        for p in sources
        if literal.search(without_props(p.read_text(encoding="utf-8")))
    ]
    check(
        not offenders,
        f"no key literal in any source file{(' -> ' + ', '.join(offenders)) if offenders else ''}",
    )

    # The inverse of what this file used to assert. Ambient credentials are the
    # bug now: a provider that can reach os.environ cannot serve two callers
    # with different keys, and will quietly use the wrong one instead of
    # failing. Reading the environment is the entry point's job.
    # claude_code.py is the one exception, and it is the opposite of a
    # loophole: it reads the environment in order to build a small allowlist and
    # hand the subprocess *that* instead of everything this process holds. A
    # source-text rule cannot tell that apart from a credential lookup, so the
    # rule is replaced for this file by the behavioural checks in section [16].
    ENV_READER_BY_DESIGN = {"claude_code.py"}
    provider_sources = [p for p in sources if p.parts[-2] == "providers"]
    check(bool(provider_sources), "there are provider modules to inspect")
    for path in provider_sources:
        if path.name in ENV_READER_BY_DESIGN:
            continue
        body = path.read_text(encoding="utf-8")
        check(
            "os.environ" not in body and "getenv" not in body,
            f"providers/{path.name} does not read the environment",
        )

    # Exactly one place may, so there is exactly one place to audit. Matched as
    # a call rather than as a substring, so that naming the pattern in prose --
    # in this check, in a docstring, in a comment -- does not count as using it.
    env_call = re.compile(r"\bos\.environ\s*(?:\.get\s*\(|\[)")
    readers = sorted(
        str(p.relative_to(ROOT))
        for p in sources
        if env_call.search(p.read_text(encoding="utf-8"))
    )
    check(
        readers == ["keys.py", "providers/claude_code.py"],
        "the environment is read only in keys.py and in the subprocess allowlist "
        f"-> {readers}",
    )
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    check(
        all(f"{v}=" in example for v in ("XAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")),
        ".env.example lists all three keys",
    )
    check(
        all(line.split("=", 1)[1].strip() == "" for line in example.strip().splitlines()),
        ".env.example has no values",
    )
    check(
        not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text(),
        ".env is git-ignored",
    )


def check_canary_never_escapes() -> None:
    """Break every provider error path on purpose and hunt for the credential.

    The rule "never log a key" is the kind that holds until the day a vendor
    answers a 400 by quoting the request back, headers included. Asserting it
    by reading the code does not survive a refactor; this drives the actual
    paths with a value that must never appear and then searches everything they
    produced -- return values, exception text, stdout, and the JSONL on disk.

    Entirely offline: the responses are hand-built, the roles are fakes, and the
    only key in play is worthless.
    """
    print("\n[14] Canary: no credential in any output")
    import contextlib
    import io as _io
    import re

    import httpx

    import controller
    from providers import ProviderResult
    from providers.redact import redact, redact_exc
    from providers.retry_utils import ProviderError, raise_for_status

    def clean(label: str, *blobs: str) -> None:
        hit = next((b for b in blobs if CANARY_KEY in b), None)
        check(hit is None, f"{label} carries no canary")

    # 1. The repr of the carrier itself. This is what lands in a traceback frame.
    clean("repr(ApiKeys)", repr(FAKE_KEYS), str(FAKE_KEYS), f"{FAKE_KEYS}")
    check(
        "set" in repr(FAKE_KEYS) and "missing" in repr(ApiKeys(xai=CANARY_KEY)),
        "repr still says which keys are present",
    )

    # 2. A vendor 400 that quotes our own request back, headers and all -- the
    #    exact shape that makes this more than a hypothetical.
    echoed = (
        '{"error":{"code":400,"message":"invalid request",'
        '"request":{"headers":{"Authorization":"Bearer ' + CANARY_KEY + '",'
        '"x-api-key":"' + CANARY_KEY + '"}}}}'
    )
    for provider in ("xai", "anthropic", "google"):
        response = httpx.Response(400, text=echoed, request=httpx.Request("POST", "https://x/y"))
        try:
            raise_for_status(provider, response, CANARY_KEY)
        except ProviderError as exc:
            clean(f"{provider} 400 error", str(exc), repr(exc), exc.message, redact_exc(exc))
        else:
            check(False, f"{provider} 400 raised nothing")

    # 3. A key we were never handed -- the backstop patterns, not exact removal.
    for shape in UNHELD_KEY_SHAPES:
        body = '{"error":"bad key ' + shape + '"}'
        response = httpx.Response(401, text=body, request=httpx.Request("POST", "https://x/y"))
        try:
            raise_for_status("xai", response)  # deliberately no secrets passed
        except ProviderError as exc:
            check(shape not in str(exc), f"an unheld {shape[:7]}... key is still redacted")

    # 4. A whole run whose Worker dies inside the provider, driven end to end.
    #    Fake roles, real Controller, real RunLog, real file on disk.
    saved = (
        controller.manager_role.plan,
        controller.worker_role.execute,
        controller.critic_role.review,
    )
    saved_runs = controller.RUNS_DIR
    tmp = Path(tempfile.mkdtemp(prefix="orchestrator-canary-"))
    controller.RUNS_DIR = tmp
    buffer = _io.StringIO()
    try:
        def fake_plan(task, previous_plan=None, verdict=None, worker_result="", **_):
            return (
                ManagerPlan(
                    plan="p",
                    worker_prompt="w",
                    acceptance_criteria=[CRITICAL_CRITERION],
                    worker_type="text",
                ),
                ProviderResult(data={}, model="grok-4.6", input_tokens=1, output_tokens=1),
            )

        def exploding_worker(plan, cwd=None, **_):
            raise ProviderError(
                "anthropic",
                f"HTTP 401: {echoed}",  # unredacted on purpose: this is the hostile case
                status=401,
            )

        controller.manager_role.plan = fake_plan
        controller.worker_role.execute = exploding_worker
        controller.critic_role.review = lambda *a, **k: None

        raised = ""
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                controller.run_task(
                    Task(goal="g"), ask=lambda q: None, run_id="canary", keys=FAKE_KEYS
                )
            except BaseException as exc:
                raised = redact_exc(exc, *FAKE_KEYS.secrets())

        clean("the redacted exception the entry point would show", raised)
        clean("everything the run printed", buffer.getvalue())

        written = sorted(tmp.rglob("*.jsonl"))
        check(bool(written), "the run really did write a log to inspect")
        for path in written:
            clean(f"runs/{path.name}", path.read_text(encoding="utf-8"))
    finally:
        (
            controller.manager_role.plan,
            controller.worker_role.execute,
            controller.critic_role.review,
        ) = saved
        controller.RUNS_DIR = saved_runs
        shutil.rmtree(tmp, ignore_errors=True)

    # 5. The scrubber keeps the message readable rather than blanking it.
    scrubbed = redact(f"HTTP 401: bad key {CANARY_KEY} for account 42", CANARY_KEY)
    check("HTTP 401" in scrubbed and "account 42" in scrubbed, "redaction keeps the diagnosis")
    check(re.search(r"\[REDACTED\]", scrubbed) is not None, "redaction leaves a visible marker")


def check_http_api() -> None:
    """Drive the HTTP surface with fake roles: no network, no keys, no spend.

    The API is an observer, so what is worth asserting is not that it can run a
    task -- the Controller does that -- but that it never becomes a second
    source of truth and never leaks the credentials it now holds. Both are
    things a reader cannot verify by looking.
    """
    print("\n[15] HTTP API (fake roles)")
    import json as _json
    import time as _time
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    import api
    import controller
    from providers import ProviderResult

    saved = (
        controller.manager_role.plan,
        controller.worker_role.execute,
        controller.critic_role.review,
    )
    saved_runs = controller.RUNS_DIR
    tmp = Path(tempfile.mkdtemp(prefix="orchestrator-api-"))
    controller.RUNS_DIR = tmp
    seen: dict[str, int] = {"manager": 0, "keys_threaded": 0}

    try:
        def fake_plan(task, previous_plan=None, verdict=None, worker_result="", **kw):
            seen["manager"] += 1
            seen["keys_threaded"] += int("keys" in kw)
            asks = task.context == "ask" and seen["manager"] == 1
            return (
                ManagerPlan(
                    plan="p",
                    worker_prompt="w",
                    acceptance_criteria=[CRITICAL_CRITERION],
                    worker_type="text",
                    needs_user_input=asks,
                    question=Question(text="which tone?", options=["formal", "casual"])
                    if asks
                    else None,
                ),
                ProviderResult(data={}, model="grok-4.6", input_tokens=10, output_tokens=10),
            )

        def fake_worker(plan, cwd=None, **kw):
            seen["keys_threaded"] += int("keys" in kw)
            return controller.worker_role.WorkerRun(
                output=WorkerOutput(result="fake result"),
                model="claude-sonnet-5",
                input_tokens=10,
                output_tokens=10,
            )

        def fake_review(plan, output, **kw):
            seen["keys_threaded"] += int("keys" in kw)
            return (
                CriticVerdict(
                    checks=[
                        CriterionCheck(
                            criterion=CRITICAL_CRITERION.text,
                            passed=True,
                            critical=True,
                            evidence="fake result",
                            reason="fake",
                        )
                    ],
                    verdict="accept",
                ),
                ProviderResult(
                    data={}, model="gemini-3.1-flash-lite", input_tokens=10, output_tokens=10
                ),
            )

        controller.manager_role.plan = fake_plan
        controller.worker_role.execute = fake_worker
        controller.critic_role.review = fake_review

        client = TestClient(api.app)
        bundle = {"xai": CANARY_KEY, "anthropic": CANARY_KEY, "google": CANARY_KEY}

        def wait_for(task_id: str, wanted: set[str], limit: float = 20.0) -> dict[str, Any]:
            deadline = _time.time() + limit
            body: dict[str, Any] = {}
            while _time.time() < deadline:
                body = client.get(f"/task/{task_id}").json()
                if body["status"] in wanted:
                    return body
                _time.sleep(0.02)
            check(False, f"timed out waiting for {wanted}; last was {body.get('status')}")
            return body

        # 1. A run starts and finishes, and the response is an id rather than a result.
        created = client.post("/task", json={"goal": "g", "budget_usd": 1.0, "keys": bundle})
        check(created.status_code == 202, "POST /task accepts with 202, not 200")
        task_id = created.json()["task_id"]
        done = wait_for(task_id, {"finished", "failed"})
        check(done["status"] == "finished", "the run reaches finished")
        check(done["summary"]["status"] == "accepted", "the summary is the Controller's, untouched")
        check(
            [e["event"] for e in done["events"]][:2] == ["round_start", "manager_plan"],
            "progress events are relayed in order",
        )
        check(seen["keys_threaded"] >= 3, "keys reached all three roles through run_task")

        # 2. Nothing anywhere in the response carries the credential.
        check(CANARY_KEY not in _json.dumps(done), "GET /task never echoes a key")
        check(
            CANARY_KEY not in _json.dumps(client.get("/openapi.json").json()),
            "the OpenAPI document never carries a key",
        )
        record = api._TASKS[task_id]
        check(record.keys.secrets() == (), "the record's keys are cleared once the run ends")
        check(CANARY_KEY not in repr(record.keys), "the cleared record still reprs safely")
        for path in tmp.rglob("*.jsonl"):
            check(CANARY_KEY not in path.read_text(encoding="utf-8"), f"runs/{path.name} is clean")

        # 3. The escalation round trip: question out, label in, run resumes.
        seen["manager"] = 0
        created = client.post(
            "/task", json={"goal": "g", "context": "ask", "budget_usd": 1.0, "keys": bundle}
        )
        ask_id = created.json()["task_id"]
        waiting = wait_for(ask_id, {"waiting_for_answer"})
        check(waiting["question"]["options"] == ["formal", "casual"], "the question is exposed")
        answered = client.post(f"/task/{ask_id}/answer", json={"answer": "formal"})
        check(answered.status_code == 200, "an option label is accepted")
        resumed = wait_for(ask_id, {"finished", "failed"})
        check(resumed["status"] == "finished", "the run resumes after the answer")
        check(
            client.post(f"/task/{ask_id}/answer", json={"answer": "formal"}).status_code == 409,
            "answering a finished task is a conflict, not a silent no-op",
        )

        # 4. Stop reaches a run that is parked on a question -- the latch alone
        #    cannot, because nothing is reading it while the thread blocks.
        seen["manager"] = 0
        created = client.post(
            "/task", json={"goal": "g", "context": "ask", "budget_usd": 1.0, "keys": bundle}
        )
        stop_id = created.json()["task_id"]
        wait_for(stop_id, {"waiting_for_answer"})
        check(client.post(f"/task/{stop_id}/stop").status_code == 200, "stop is accepted")
        stopped = wait_for(stop_id, {"finished", "failed"})
        check(
            stopped["summary"]["status"] == "escalated_unanswered",
            "stopping at a question ends the run without inventing an answer",
        )

        # 5. Nobody answers. The run must end on its own, write one terminal
        #    record, and let go of the credentials -- a task parked forever on a
        #    question is also a task holding three keys forever.
        seen["manager"] = 0
        with patch.object(api, "ESCALATION_TIMEOUT_SECONDS", 0.25):
            created = client.post(
                "/task", json={"goal": "g", "context": "ask", "budget_usd": 1.0, "keys": bundle}
            )
            silent_id = created.json()["task_id"]
            timed = wait_for(silent_id, {"finished", "failed"})
        check(timed["status"] == "finished", "an unanswered escalation still finishes")
        check(timed["timed_out"] is True, "the timeout is carried by the flag, not by the status")
        check(timed["error"] is None, "finishing on a timeout is not an error")
        # The point of the fix: the run reached its own ending, so what it had
        # already done survives. Raising from inside the hook destroyed all of
        # this to report that nobody clicked.
        check(timed["summary"] is not None, "the summary survives the timeout")
        check(
            timed["summary"]["status"] == "escalated_unanswered",
            "the Controller ended it through its own no-answer path",
        )
        check(
            timed["summary"]["budget"]["spent_usd"] > 0,
            "the spend that already happened is still reported",
        )
        check(timed["summary"]["rounds"] >= 1, "the rounds already run are still reported")
        check(timed["question"] is None, "the stale question is cleared")
        silent = api._TASKS[silent_id]
        check(
            silent.keys.secrets() == (),
            "the timed-out run released its credentials in the finally",
        )
        check(CANARY_KEY not in _json.dumps(timed), "the timeout record carries no key")

        # 6. Unknown ids are 404, not 500 or an empty record.
        check(client.get("/task/nope").status_code == 404, "an unknown task id is a 404")
        check(
            client.post("/task/nope/answer", json={"answer": "x"}).status_code == 404,
            "answering an unknown task id is a 404",
        )

        # 7. The API must not have grown a policy of its own.
        source = (ROOT / "api.py").read_text(encoding="utf-8")
        for banned in ("max_rounds >", "consecutive_rejections", "two_rejections", "spent_usd >"):
            check(banned not in source, f"api.py contains no orchestration logic ({banned})")
    finally:
        (
            controller.manager_role.plan,
            controller.worker_role.execute,
            controller.critic_role.review,
        ) = saved
        controller.RUNS_DIR = saved_runs
        with api._LOCK:
            api._TASKS.clear()
        shutil.rmtree(tmp, ignore_errors=True)


def check_code_worker_isolation() -> None:
    """The code worker's environment, and the gate in front of it. Offline.

    Two claims that a reader cannot verify by looking, and that a source-text
    rule cannot check either:

      * the subprocess is handed an explicit allowlist, so no credential in this
        process reaches a program that can write files and run commands,
      * with the gate closed, the code path is not merely refused at the end --
        it is never entered, so ``subprocess.run`` is never reached at all.

    The second is the one worth driving rather than asserting. A refusal placed
    one line too late still runs the subprocess.
    """
    print("\n[16] Code worker isolation")
    from unittest.mock import patch

    from providers import claude_code
    from roles import worker as worker_role

    # 1. The allowlist keeps what the CLI needs and drops what it must not see.
    poisoned = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "LANG": "en_US.UTF-8",
        "XAI_API_KEY": CANARY_KEY,
        "ANTHROPIC_API_KEY": CANARY_KEY,
        "GOOGLE_API_KEY": CANARY_KEY,
        "AWS_SECRET_ACCESS_KEY": CANARY_KEY,
        "SOME_OTHER_TOKEN": CANARY_KEY,
    }
    with patch.dict("os.environ", poisoned, clear=True):
        env = claude_code.child_env()
    check(env.get("PATH") == "/usr/bin", "PATH is passed through")
    check(env.get("HOME") == "/home/someone", "HOME is passed through")
    for name in claude_code.FORBIDDEN_ENV:
        check(name not in env, f"{name} is not in the subprocess environment")
    check(
        CANARY_KEY not in "\x00".join(env.values()),
        "no value in the subprocess environment carries a credential",
    )
    extra = sorted(set(env) - set(claude_code.INHERITED_ENV))
    check(not extra, f"nothing outside the allowlist leaks in -> {extra}")
    check(
        "AWS_SECRET_ACCESS_KEY" not in env and "SOME_OTHER_TOKEN" not in env,
        "unrelated secrets in the parent environment are dropped too",
    )

    # 2. A key name added to the allowlist by mistake is still dropped.
    mistaken = (*claude_code.INHERITED_ENV, "GOOGLE_API_KEY")
    with (
        patch.object(claude_code, "INHERITED_ENV", mistaken),
        patch.dict("os.environ", poisoned, clear=True),
    ):
        env2 = claude_code.child_env()
    check("GOOGLE_API_KEY" not in env2, "a key name allowlisted by mistake is still removed")

    # 3. The gate: closed means the subprocess is never reached.
    code_plan = ManagerPlan(
        plan="p",
        worker_prompt="edit the file",
        acceptance_criteria=[CRITICAL_CRITERION],
        worker_type="code",
    )
    with patch.object(claude_code, "run") as never:
        run = worker_role.execute(code_plan, keys=FAKE_KEYS, allow_code_worker=False)
    check(not never.called, "with the gate closed, claude_code.run is never called")
    check(run.output.ok is False, "the refusal comes back as a failed WorkerRun")
    check(bool(run.failure_reason), "the refusal carries a reason the Manager can act on")
    check(
        "text worker" in run.failure_reason,
        "the reason tells the Manager what to do instead, not just that it failed",
    )
    check(CANARY_KEY not in run.failure_reason, "the reason carries no credential")

    # 4. Default is closed. A caller that says nothing does not get the filesystem.
    with patch.object(claude_code, "run") as never_either:
        default_run = worker_role.execute(code_plan, keys=FAKE_KEYS)
    check(not never_either.called, "the code path is off by default, not on")
    check(default_run.output.ok is False, "the default refusal is a failed WorkerRun too")

    # 5. Open means it really does run -- otherwise this whole check could pass
    #    against a worker that had simply lost the ability to run code.
    with patch.object(claude_code, "run", return_value=claude_code.CodeResult(
        ok=True, result="done", num_turns=2, cost_usd=0.01
    )) as called:
        allowed = worker_role.execute(code_plan, keys=FAKE_KEYS, allow_code_worker=True)
    check(called.called, "with the gate open, the code worker is actually invoked")
    check(allowed.output.ok is True, "an allowed code run comes back ok")

    # 6. Headless has nobody to approve a write, so the command must carry its
    #    own permissions. Without this the CLI runs, burns its turns, exits 0,
    #    and explains in prose that it could not create the file -- a failure
    #    that costs real money and looks like the model being unhelpful.
    captured: dict[str, Any] = {}

    class _Done:
        returncode = 0
        stdout = '{"result": "done", "num_turns": 1, "total_cost_usd": 0.01}'
        stderr = ""

    def _capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _Done()

    with (
        patch.object(claude_code.shutil, "which", return_value="/usr/local/bin/claude"),
        patch.object(claude_code.subprocess, "run", _capture),
    ):
        claude_code.run("do a thing", cwd="/tmp")
    cmd = captured["cmd"]
    check("--allowedTools" in cmd, "the headless command carries --allowedTools")
    granted = cmd[cmd.index("--allowedTools") + 1].split(",")
    check("Write" in granted, "Write is granted, or the worker cannot create a file")
    check("Bash" in granted, "Bash is granted, or the worker cannot run what it wrote")
    check(
        set(granted) == set(claude_code.DEFAULT_ALLOWED_TOOLS),
        f"the grant is exactly the documented set -> {granted}",
    )
    check(
        captured["env"] is not None and "ANTHROPIC_API_KEY" not in captured["env"],
        "the real subprocess call is the one getting the filtered environment",
    )

    # 7. The refusal is a normal rejected round, not a new exception type.
    from roles import critic as critic_role

    verdict = critic_role.failed_worker_verdict(run.failure_reason, [CRITICAL_CRITERION])
    check(verdict.accepted is False, "the refusal becomes a rejection the loop understands")
    check(len(verdict.checks) == 1, "the synthetic verdict still covers every criterion")


def main() -> int:
    print("self check -- no API keys required, no network calls")
    check_pydantic_models()
    check_anthropic()
    check_xai()
    check_gemini()
    check_ref_resolution()
    check_budget()
    check_controller_state_machine()
    check_fixture_builder()
    check_critic_harness()
    check_critic_null_handling()
    check_fixture_file_integrity()
    check_retry_policy()
    check_no_hardcoded_keys()
    check_canary_never_escapes()
    check_http_api()
    check_code_worker_isolation()

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
