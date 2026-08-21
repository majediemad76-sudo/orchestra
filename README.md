# Multi-Model Orchestrator

A four-role orchestration loop — Manager, Worker, Critic, Controller — that
decomposes a task, executes it, grades the result against explicit criteria,
and revises until it passes or the budget says stop.

Three vendors are involved on purpose. The control flow is not.

```
        ┌─────────────────────────────────────────────────────┐
        │  Controller  (plain Python — deterministic)         │
        │  rounds · budget · escalation · JSONL log           │
        └───┬──────────────┬───────────────┬──────────────────┘
            │              │               │
            ▼              ▼               ▼
      ┌──────────┐   ┌──────────┐   ┌──────────────┐
      │ Manager  │──▶│  Worker  │──▶│    Critic    │
      │ grok-4.6 │   │ sonnet-5 │   │ gemini-flash │
      │   xAI    │   │Anthropic │   │    Google    │
      └──────────┘   └──────────┘   └──────┬───────┘
            ▲                              │
            └───────── fix_instruction ────┘
```

---

## Design philosophy

### The controller is code, never a model

Every consequential decision in this system — how many retries are allowed,
when to stop, when to interrupt a human, when the money has run out — is made
in [controller.py](controller.py), in Python, by code you can read end to end.

Models are asked for judgement: *is this output good?*, *what should the worker
do next?* They are never asked what happens next. The distinction matters
because a model deciding its own retry budget will always find one more thing
worth trying, and a model deciding when to ask for help will either never ask
or ask constantly. Neither failure is visible until the bill arrives.

This is also why there is no agent framework here. The loop is ~200 lines of
explicit state transitions. A framework would add a dependency, a DSL, and a
layer of indirection between "what should happen" and "what happens" — in
exchange for abstractions this system does not need. When something misbehaves,
the stack trace points at a line in a `while` loop rather than into a library's
internal scheduler.

The same reasoning rules out node-based tools (n8n and friends) for the loop
itself. Conditional retry logic, budget arithmetic, and a three-trigger
escalation policy are *code* — they want version control, diffs, and tests.
A UI can sit on top of `run_task()` later; the logic stays here.

### Exactly three reasons to interrupt a human

An orchestrator that asks whenever it is unsure is an orchestrator nobody
leaves running. There are three triggers, and adding a fourth means editing a
`Literal` in [schemas.py](schemas.py) — a design decision that shows up in
review as one.

| Trigger | Condition | Why it stops here |
|---|---|---|
| `two_rejections` | Critic rejected twice in a row | One rejection is the loop working. Two means revision is not converging, and a third round burns budget rather than fixing it. |
| `manager_needs_input` | Manager set `needs_user_input` | Information is genuinely absent and guessing carries risk. The Manager reports; the Controller decides. |
| `budget_exceeded` | Spend crossed the dollar ceiling | Checked at the top of each round, before committing to another three calls. |

Everything else resolves inside the loop. A Critic `escalate` verdict counts as
a rejection and returns to the Manager — who may then trigger #2. A Claude Code
timeout counts as a rejection. Hitting the round cap ends the run with a report.

**Every escalation is one question with 2–4 concrete options**, enforced by the
type system (`Question.options` is bounded 2..4). An escalation interrupts a
human, so it must be answerable in one keystroke. "What should I do?" hands the
problem back untouched.

### Cross-vendor triangulation

The Critic runs on a different vendor from the Worker. This is load-bearing,
not incidental.

A model grading its own output is the worst available judge — the same weights
that produced the mistake rate it as fine. A sibling model from the same family
is barely better: shared training data means shared blind spots. Independence
between the party that writes and the party that grades is what makes the score
worth reading at all.

| Role | Model | Vendor | Why this one |
|---|---|---|---|
| Manager | `grok-4.6` | xAI | Decomposition and instruction-writing. Never executes, never controls the loop. |
| Worker (text) | `claude-sonnet-5` | Anthropic | Strong long-form execution; the deliverable itself. |
| Worker (code) | Claude Code headless | Anthropic | Subprocess with filesystem access — what an API call cannot do. |
| Critic | `gemini-3.1-flash-lite` | Google | Independent judgement, cheap enough to grade every round. |

If you change the Critic's model, change it to a *third vendor* — not to
whatever is cheapest that week.

### Structured output at the API layer, not in the prompt

"Reply with JSON only" is a request. A schema enforced by the API is a
guarantee. Every role's output is parsed by code that must not have to guess,
so the prompt instruction is treated as a backup and the vendor mechanism as
the contract.

The three vendors share nothing but ancestry, and
[providers/schema_utils.py](providers/schema_utils.py) is the single place
their quirks are allowed to live. One Pydantic model in, three dialects out:

| Vendor | Mechanism | The part that bites |
|---|---|---|
| **Anthropic** | Forced tool call: `tools=[{…"input_schema": schema}]` + `tool_choice={"type":"tool","name":"emit_result"}` | Claude has no `response_format`. Forcing the tool is *stronger*, not weaker — prose is not an available move. |
| **xAI** | `response_format: {"type":"json_schema", "json_schema":{…,"strict":true}}` (OpenAI-compatible endpoint) | Strict mode adds two rules JSON Schema does not have: `additionalProperties: false` on **every** object, and **every** property listed in `required`. Optional fields are widened to `["string","null"]` instead of omitted. |
| **Google** | `generationConfig.responseMimeType` + `generationConfig.responseSchema` | Its own dialect: upper-case type names (`STRING`/`OBJECT`/`ARRAY`/`INTEGER`/`BOOLEAN`), no union types — nullability is a sibling `nullable: true` flag. Unknown keys are ignored *silently*, so the allow-list is explicit. |

All three paths start by inlining `$ref`: Pydantic factors nested models into
`$defs` and references them, which Anthropic and xAI reject and Gemini quietly
ignores. Flattening happens once, in `resolve_refs`.

### Failure is a signal, not an exception

The code Worker runs Claude Code headless as a subprocess with `--max-turns 15`
and `timeout=600`. A timeout, a missing CLI, a non-zero exit, or unparseable
stdout each return a result with `ok=False` and a reason — never a raised
exception. The Controller translates that into a Critic-style rejection, and
the loop revises and retries.

A crash would throw away the run's accumulated context and its budget along
with it. A rejection is information the Manager can use.

The retry policy in [providers/retry_utils.py](providers/retry_utils.py) splits
on the same question — not "did it fail" but "will waiting help":

- **429, 5xx, transport errors** → the provider's problem, usually transient.
  Three attempts, exponential backoff.
- **400, 401** → our problem. A wrong key returns the identical error three
  times, three times slower, with the real cause buried under a retry trace.
  These fail immediately.

---

## Quick start

```bash
git clone <this repo> && cd orchestrator
make install                      # creates .venv, installs pinned deps
make test                         # 68 offline checks — no API keys needed
```

Then supply keys for a live run:

```bash
cp .env.example .env              # fill in XAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
./.venv/bin/python controller.py "Write a 120-word changelog entry for v2.1" \
    --budget 0.25 --max-rounds 3
```

No key is ever read from anywhere but `.env` / the environment. `.env` is
git-ignored; `.env.example` is the template and holds no values.

### Make targets

| Target | What it does |
|---|---|
| `make test` | Offline self check — no keys, no network, no cost |
| `make lint` | Byte-compile every source file |
| `make check` | `lint` then `test` — the pre-commit gate |
| `make run` | Live smoke run at a small budget (needs `.env`; **costs money**) |
| `make ui` | Streamlit observer over the same `run_task` |
| `make serve` | HTTP API on `127.0.0.1:8000` — keys arrive per request, not from `.env` |
| `make clean` | Drop bytecode caches and self-check logs, keep real run logs |
| `make distclean` | Also drop every run log and the venv |

### CLI

```
controller.py GOAL [--context TEXT] [--max-rounds N]
                   [--budget USD] [--accept-score 0-100] [--cwd DIR]
```

`--cwd` sets the working directory for the code Worker. Exit code is 0 when the
run was accepted (by the Critic or by the user), 1 otherwise.

---

## HTTP API

`make serve` puts the same `run_task` behind three endpoints. Like the
Streamlit app, it is an observer: it starts a run, relays progress, carries one
answer back, and reports the summary. It owns no round counting, no budget
arithmetic, and no escalation policy — those stay in `controller.py`, and
`self_check` asserts that `api.py` has not grown a copy of them.

| Endpoint | What it does |
|---|---|
| `POST /task` | Credentials, goal, and ceilings in the body. Returns `202` and a `task_id` immediately. |
| `GET /task/{id}` | Status, the progress events so far, an open question if there is one, and the summary once it finishes. |
| `POST /task/{id}/answer` | Answers an escalation with an **option label** — the text the caller was shown, not an index. |
| `POST /task/{id}/stop` | Sets the cancel latch, and posts an empty answer too, since a run parked on a question is not reading the latch. |

An escalation that goes unanswered for `ESCALATION_TIMEOUT_SECONDS` (300s, shared
with the Streamlit app in [escalation.py](escalation.py)) ends the run the same
way `/stop` does: the surface hands the Controller an empty answer, the loop
reaches its own ending, and `GET /task/{id}` returns `status: finished` with the
**full summary** — rounds, spend, and whatever output had been reached — plus
`timed_out: true`. The credentials are released either way.

`timed_out` is a flag next to the result rather than a different kind of exit,
and that is the whole point: raising out of the escalation hook would abort
`run_task` from the inside and destroy the summary it was about to return, so a
caller who stepped away would lose work that had already been paid for.

```bash
curl -s localhost:8000/task -H 'content-type: application/json' -d '{
  "goal": "Write a two-sentence summary of what a multi-model orchestrator does.",
  "budget_usd": 0.25,
  "keys": {"xai": "...", "anthropic": "...", "google": "..."}
}'
# {"task_id":"a1b2c3d4e5f6","status":"running"}

curl -s localhost:8000/task/a1b2c3d4e5f6
```

### Keys are per request

Credentials arrive in the body of `POST /task`, live in that task's record for
as long as the run does, and are dropped when it ends. They are never logged,
never written to disk, never returned by any endpoint, and never part of the
OpenAPI document. `ApiKeys.__repr__` prints `set`/`missing`, so one landing in
a traceback frame does not print itself.

That "dropped" is a dropped reference, not a wiped buffer: Python strings are
immutable and nothing here can overwrite one in place. It bounds how long the
value is reachable, which is the real risk; it is not a guarantee about process
memory and is not described as one anywhere in the code.

### Do not deploy this without TLS

It binds `127.0.0.1` and refuses to pretend that is a limitation to be removed
later. **API keys travel in the request body**, so a plaintext listener on any
reachable interface publishes three credentials to anyone on the path. Passing
`--host` prints a warning and does not make it safe.

Before this is reachable from anywhere else it needs, at minimum: TLS
terminated in front of it, authentication of its own (it has none — anyone who
can reach it can start a run), and a bound on how many tasks one caller may
create. None of that is implemented.

### What it does not do

State is a dict in memory. There is no database, no Redis, and no broker,
because those buy durability across restarts — **a restart loses every
in-flight run**, and the records of finished ones. Tasks accumulate for the
life of the process; nothing evicts them. That is the honest shape of a
single-node service, written down rather than discovered.

---

## Testing without spending money

[scripts/self_check.py](scripts/self_check.py) runs with no API key set and
makes no network call. It covers the parts that fail silently and expensively:

1. **Schemas** build with and without their optional fields; `Question` rejects
   0, 1, and 5 options.
2. **Anthropic converter** leaves no `$ref` or `$defs` anywhere in the tree.
3. **xAI converter** puts `additionalProperties: false` on every nested object
   and lists every property in `required`.
4. **Gemini converter** upper-cases every type and maps both flavours of
   optional (`Optional[X]` and "has a default") to `nullable: true`.
5. **Budget** arithmetic against the published per-million rates.
6. **The state machine**, driven with fake roles — all three escalation
   triggers, the round accounting, the JSONL log, and the non-interactive halt.

That last one runs `controller.run_task` **unmodified**, swapping only the
three role functions. A rewritten test copy of the loop would pass while the
real one was broken, which is the failure this file exists to prevent.

It is deliberately dependency-free — no pytest, no fixtures. One command that
either prints `all good` or names what broke.

---

## Layout

```
controller.py           the loop: rounds, budget checks, escalation, logging
api.py                  HTTP surface over run_task -- observer only, no logic
app.py                  Streamlit surface over run_task -- observer only, no logic
keys.py                 ApiKeys: the only module that reads the environment
schemas.py              Pydantic contracts — also the source of every vendor schema
budget.py               per-call cost accounting and the ceiling
roles/
    manager.py          decomposition → worker_prompt + acceptance_criteria
    worker.py           text backend / code backend, one result shape
    critic.py           grading, and the synthetic rejection for a dead worker
prompts/
    manager.md          the Manager's standing instructions
    critic.md           the Critic's standing instructions
providers/
    xai.py              Manager transport (OpenAI-compatible, strict json_schema)
    anthropic.py        Worker transport (forced tool call + plain text)
    google.py           Critic transport (responseSchema)
    claude_code.py      headless subprocess, never raises
    schema_utils.py     one Pydantic model → three dialects
    retry_utils.py      retry what waiting fixes, fail fast on the rest
    redact.py           scrub credentials out of anything about to be shown
scripts/
    self_check.py       the offline gate
runs/                   one JSONL file per run (git-ignored)
```

## Observability

Every run writes `runs/<run_id>.jsonl`, append-only: `run_start`,
`manager_plan`, `worker_output`, `critic_verdict`, `escalation`,
`escalation_answer`, `budget_raised`, `run_end` — each with the cost of that
step. JSONL rather than one JSON document because a run that dies partway still
leaves a readable file, and `tail -f` works while it is in flight.

"Why did this run cost $2" is answerable after the fact rather than
reconstructible at best.

## Costs

USD per million tokens, August 2026 — a table in `budget.py`, not a live
lookup, because a pricing call on every step would add latency and a failure
mode to the accountant.

| Model | Input | Output |
|---|---|---|
| `claude-sonnet-5` | $2.00 | $10.00 |
| `grok-4.6` | $2.00 | $6.00 |
| `gemini-3.1-flash-lite` | $0.25 | $1.50 |

The code Worker reports its own `total_cost_usd` — an authoritative figure for
a whole multi-turn session that no token count here could reconstruct.

## Conventions

Project rules that outlive any one session are recorded in
[CLAUDE.md](CLAUDE.md). The short version: the Controller is code, the loop
logic stays in Python, the Critic stays on a different vendor from the Worker,
and no API key is ever hardcoded.
