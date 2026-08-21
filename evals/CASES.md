# Cases — real runs worth keeping

**These are not fixtures, and nothing here is ever counted in the Critic's
score.** `critic_fixtures.jsonl` is a graded set: every row has a reviewed
ground-truth label, and a rate is computed over all of them. A case is the
opposite — one real run, kept because of what it showed, with no claim that it
is representative of anything. Mixing the two would put an unreplicated
anecdote into a denominator.

Read a case for the mechanism it exposed. Do not read a rate out of it.

Each entry is transcribed from the run log named in it, not from memory. The
logs are git-ignored, so a case that outlives its log keeps the numbers here.

---

## Case 1 — the code worker's first two real runs

**Date:** 2026-08-21 (UTC) · **Goal, identical in both runs:**

> Create a file named hello.py in the current directory that prints the sum of
> two numbers, then run it and show the output.

Both ran through `make run` with `--cwd` pointed at an empty temporary
directory, `RUN_BUDGET=0.20`, `RUN_ROUNDS=2`.

### Run A — every machine signal said success; nothing was created

| | |
|---|---|
| run_id | `20260820-214429-203d31` |
| log | `runs/20260820-214429-203d31.jsonl` |
| started | 2026-08-21T01:44:29Z |
| worker | `claude-code-headless`, `worker_type=code` |
| turns | **14** |
| cost_usd | **0.322987** |
| cost_basis | `subscription_equivalent` |
| `WorkerRun.ok` | **True** |
| `failure_reason` | `""` (empty) |
| file on disk | **no** — the working directory was still empty afterwards |
| Critic verdict | **not accepted**, score 50 (4 of 8) |
| escalation | `budget_exceeded` — $0.20 ceiling, $0.3331 spent |
| final status | `stopped_by_user` (unanswered escalation; the run was not on a tty) |
| rounds · total spend · duration | 2 · $0.333111 · 102.34s |

The eight criteria as the Critic judged them:

| passed | criterion |
|:-:|---|
| ✗ | A file named exactly hello.py exists in the current working directory. |
| ✗ | hello.py parses as valid Python 3 with no syntax errors. |
| ✓ | hello.py contains an addition of two numeric values. |
| ✓ | hello.py prints the result of that addition to stdout. |
| ✓ | hello.py does not read from stdin or sys.argv for the two numbers. |
| ✗ | The worker executed hello.py with a Python interpreter. |
| ✗ | The captured program stdout contains a number equal to the sum of the two addends in hello.py. |
| ✓ | Any commentary in the worker response is in English. |

The cause was not the model. `claude -p` was invoked without `--allowedTools`,
so the CLI treated every write as needing approval, found no interactive session
to ask, and denied it. Fixed in `providers/claude_code.py` by defaulting
`allowed_tools` to `DEFAULT_ALLOWED_TOOLS`.

### Run B — same goal, same directory, after the fix

| | |
|---|---|
| run_id | `20260820-214835-05334f` |
| log | `runs/20260820-214835-05334f.jsonl` |
| started | 2026-08-21T01:48:35Z |
| turns | **3** |
| cost_usd | **0.102777** |
| cost_basis | `subscription_equivalent` |
| `WorkerRun.ok` | True |
| file on disk | **yes** — `hello.py`, 13 bytes, containing `print(3 + 5)`; running it prints `8` |
| Critic verdict | **accepted**, score 100 (7 of 7) |
| final status | `accepted` |
| rounds · total spend · duration | 1 · $0.112532 · 76.23s |

All seven criteria passed:

| passed | criterion |
|:-:|---|
| ✓ | A file named exactly hello.py exists in the current working directory. |
| ✓ | hello.py contains the integer literals 3 and 5. |
| ✓ | hello.py prints the result of adding those two numbers. |
| ✓ | hello.py was executed with a Python interpreter. |
| ✓ | The captured stdout of that run is the number 8 with at most a trailing newline. |
| ✓ | The worker's final response includes the program output 8. |
| ✓ | The worker's prose (non-code) response is in English. |

### Why this case is kept

Every mechanical signal available to the Controller reported success in Run A:
the process exited 0, the JSON envelope carried `is_error: false`, the parsed
`WorkerRun` had `ok=True` and an empty `failure_reason`, and 14 turns of work
had been billed. Nothing in the transport layer could tell that the run had
produced nothing, because from the transport's point of view it had not failed.
The only thing that caught it was the Critic.

And it caught it because the criteria were checkable against the world rather
than against taste: *a file with this exact name exists*, *stdout is the number
8*. "The code is good" would have passed — the code the worker described was
correct, which is exactly why three of the eight criteria did pass. The four
that failed were the four that asked whether anything had actually happened.

### The limit of this case

**This is n=1.** It says the Critic caught this failure, on this task, with
these criteria. It says nothing about what fraction of silent code-worker
failures it would catch, and it is not evidence that objective criteria are
reliably produced — the Manager wrote good ones here, once. Neither number
belongs in any rate, which is why this file is not `critic_fixtures.jsonl`.
