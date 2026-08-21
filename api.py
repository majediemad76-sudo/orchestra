"""HTTP surface over ``run_task``. An observer, exactly like ``app.py``.

This module holds no orchestration logic and must not acquire any: no round
counting, no budget arithmetic, no escalation policy. It starts a run, relays
progress, carries one answer back, and reports the summary. Every decision
stays in ``controller.py``, which is the only place that can be audited for
them. A second copy of the loop's rules living here would drift from the first
and there would be no way to tell which one was right.

Concurrency follows the pattern ``app.py`` already proved:

  * the run owns a background thread; the request handlers never block on it
    except where a human is deliberately being waited for,
  * the thread talks through queues, never by mutating shared state in place,
  * cancellation is a ``threading.Event`` latch rather than a queue message,
    because it has to be readable while the thread is blocked somewhere else,
  * every exit path posts exactly one terminal record, so a poller is never
    left waiting on a thread that has already died.

State is a dict in memory behind a lock. No database, no Redis, no broker --
those buy durability across restarts, which this does not have and does not
claim: a restart loses every in-flight run. That is the honest trade for a
single-node service, and it is written down here rather than discovered later.

Credentials arrive per request, live in the record for the length of that run,
and are dropped when it ends. They are never logged, never returned by any
endpoint, and never written to disk.

Binding is loopback-only and that is not a placeholder for "we will add TLS
later": keys travel in the request body, so a plaintext listener on any
reachable interface publishes them. See the deployment note in the README.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from controller import run_task
from escalation import ESCALATION_TIMEOUT_SECONDS, EscalationTimeout
from keys import ApiKeys
from providers.redact import redact_exc
from schemas import Question, Task

# Loopback only. Changing this without terminating TLS in front is how the keys
# in POST /task end up on the wire in plaintext.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

RunStatus = Literal["running", "waiting_for_answer", "finished", "failed"]


# --- request and response bodies -------------------------------------------


class KeyBundle(BaseModel):
    """The caller's credentials, for this run only.

    Kept in the request body rather than a header so that all three arrive
    together and none is mistaken for an auth token for *this* service, which
    has none. Nothing echoes these back.
    """

    xai: str = Field(min_length=1)
    anthropic: str = Field(min_length=1)
    google: str = Field(min_length=1)


class TaskRequest(BaseModel):
    goal: str = Field(min_length=1)
    context: str = ""
    max_rounds: int = Field(default=3, ge=1, le=10)
    budget_usd: float = Field(default=0.50, gt=0, le=20.0)
    cwd: str | None = None
    keys: KeyBundle


class TaskCreated(BaseModel):
    task_id: str
    status: RunStatus


class PendingQuestion(BaseModel):
    """The open escalation, as the caller sees it.

    No trigger field: ``Question`` does not carry which of the three triggers
    fired, and inventing a constant for it would be a field that looks like
    information and is not. The trigger is in the run log and in the summary's
    ``escalations`` list, both of which are already reachable.
    """

    text: str
    options: list[str]


class TaskState(BaseModel):
    task_id: str
    status: RunStatus
    created_at: float
    events: list[dict[str, Any]]
    question: PendingQuestion | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    # Set when the run ended because nobody answered an escalation. A separate
    # field because a poller cannot otherwise tell "no one was there" from "a
    # provider broke", and the two call for different reactions: come back and
    # answer, versus look at what failed.
    timed_out: bool = False


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


class AnswerAccepted(BaseModel):
    task_id: str
    accepted: str


# --- in-memory state --------------------------------------------------------


class TaskRecord:
    """Everything known about one run. Guarded by ``_LOCK`` for every read."""

    def __init__(self, task_id: str, keys: ApiKeys):
        self.task_id = task_id
        self.status: RunStatus = "running"
        self.created_at = time.time()
        self.events: list[dict[str, Any]] = []
        self.question: PendingQuestion | None = None
        self.summary: dict[str, Any] | None = None
        self.error: str | None = None
        self.timed_out = False
        self.answers: queue.Queue[str] = queue.Queue()
        self.stop_flag = threading.Event()
        # Held only while the run needs them. Cleared in the thread's finally.
        self.keys = keys
        self.thread: threading.Thread | None = None

    def snapshot(self) -> TaskState:
        return TaskState(
            task_id=self.task_id,
            status=self.status,
            created_at=self.created_at,
            events=list(self.events),
            question=self.question,
            summary=self.summary,
            error=self.error,
            timed_out=self.timed_out,
        )


_TASKS: dict[str, TaskRecord] = {}
_LOCK = threading.Lock()

app = FastAPI(
    title="Multi-Model Orchestrator",
    description="HTTP surface over run_task. Loopback only; do not expose without TLS.",
    version="1.0.0",
)


def _record(task_id: str) -> TaskRecord:
    with _LOCK:
        record = _TASKS.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no task {task_id}")
    return record


# --- the run thread ---------------------------------------------------------


def _run(record: TaskRecord, task: Task, cwd: str | None) -> None:
    """Body of the run thread. Mutates the record only under the lock.

    Mirrors ``app.py._run_in_background``: the hooks are observation and
    relay, never decision. ``on_escalation`` blocks on the answer queue on
    purpose -- the Controller is mid-decision and must not continue on a guess.
    """
    keys = record.keys

    def on_progress(event: str, data: dict[str, Any]) -> None:
        with _LOCK:
            record.events.append({"event": event, "data": data, "at": time.time()})

    def on_escalation(question: Question) -> str:
        with _LOCK:
            record.question = PendingQuestion(
                text=question.text, options=list(question.options)
            )
            record.status = "waiting_for_answer"
        try:
            answer = record.answers.get(timeout=ESCALATION_TIMEOUT_SECONDS)
        except queue.Empty:
            # Raised rather than returned as an empty answer, matching app.py.
            # An empty answer is what the *stop* endpoint posts, so returning
            # one here would make "nobody came back" indistinguishable from
            # "the caller ended it" in the record afterwards. Either way the
            # run ends; neither becomes a default choice.
            raise EscalationTimeout(
                f"No answer within {ESCALATION_TIMEOUT_SECONDS:.0f}s: {question.text}"
            ) from None
        finally:
            with _LOCK:
                record.question = None
                if record.status == "waiting_for_answer":
                    record.status = "running"
        return answer

    try:
        summary = run_task(
            task,
            cwd=cwd,
            on_progress=on_progress,
            on_escalation=on_escalation,
            stop_flag=record.stop_flag,
            keys=keys,
            # Off, and stated rather than left to the default. An HTTP caller
            # is not the operator of this host: nothing in a request body
            # should be able to write files here. A goal the Manager reads as
            # a coding task comes back as a rejected round with a reason, and
            # the loop re-plans it for the text worker.
            allow_code_worker=False,
        )
        with _LOCK:
            record.summary = summary
            record.status = "finished"
    except EscalationTimeout as exc:
        with _LOCK:
            record.error = str(exc)
            record.timed_out = True
            record.status = "failed"
    except BaseException as exc:
        # Broad on purpose: whatever went wrong, the poller must get exactly one
        # terminal record or it waits forever on a thread that is already gone.
        with _LOCK:
            record.error = redact_exc(exc, *keys.secrets())
            record.status = "failed"
    finally:
        # The run is over, so the credentials have no further use. This drops
        # the reference the record holds; it is not a claim about process
        # memory, which Python does not let us make about strings.
        keys.clear()
        with _LOCK:
            record.question = None


# --- endpoints --------------------------------------------------------------


@app.post("/task", response_model=TaskCreated, status_code=202)
def create_task(request: TaskRequest) -> TaskCreated:
    """Start a run and return immediately with its id.

    202, not 200: the work has been accepted, not completed. A caller that
    treats this as the result will get an id and no output, which is the point.
    """
    task_id = uuid.uuid4().hex[:12]
    keys = ApiKeys(
        xai=request.keys.xai.strip(),
        anthropic=request.keys.anthropic.strip(),
        google=request.keys.google.strip(),
    )
    record = TaskRecord(task_id, keys)
    task = Task(
        goal=request.goal,
        context=request.context,
        max_rounds=request.max_rounds,
        budget_usd=request.budget_usd,
    )
    thread = threading.Thread(
        target=_run,
        args=(record, task, request.cwd),
        name=f"orchestrator-{task_id}",
        daemon=True,  # a shutting-down server must not be held open by a run
    )
    record.thread = thread
    with _LOCK:
        _TASKS[task_id] = record
    thread.start()
    return TaskCreated(task_id=task_id, status="running")


@app.get("/task/{task_id}", response_model=TaskState)
def get_task(task_id: str) -> TaskState:
    """Everything known about the run. Never includes credentials."""
    return _record(task_id).snapshot()


@app.post("/task/{task_id}/answer", response_model=AnswerAccepted)
def answer_task(task_id: str, request: AnswerRequest) -> AnswerAccepted:
    """Hand one option label to a run blocked at an escalation.

    The label, not an index: what a caller holds is the text it was shown, and
    an index silently means something else the moment the options change. An
    unrecognised label reaches the Controller and is read there as no answer --
    this endpoint does not second-guess it.
    """
    record = _record(task_id)
    with _LOCK:
        waiting = record.question is not None
        status = record.status
    if status in {"finished", "failed"}:
        raise HTTPException(status_code=409, detail=f"task {task_id} is {status}")
    if not waiting:
        raise HTTPException(status_code=409, detail=f"task {task_id} is not waiting for an answer")
    record.answers.put(request.answer)
    return AnswerAccepted(task_id=task_id, accepted=request.answer)


@app.post("/task/{task_id}/stop", response_model=TaskState)
def stop_task(task_id: str) -> TaskState:
    """Ask a run to stop at the next round boundary.

    Sets the latch and also posts an empty answer, because a run parked on an
    escalation is not reading the latch -- exactly the case app.py hit.
    """
    record = _record(task_id)
    record.stop_flag.set()
    with _LOCK:
        waiting = record.question is not None
    if waiting:
        record.answers.put("")
    return record.snapshot()


@app.get("/health")
def health() -> dict[str, Any]:
    with _LOCK:
        running = sum(1 for r in _TASKS.values() if r.status in {"running", "waiting_for_answer"})
        total = len(_TASKS)
    return {"ok": True, "tasks": total, "running": running}


def main() -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the orchestrator over HTTP")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if args.host != DEFAULT_HOST:
        print(
            f"WARNING: binding {args.host}, not loopback. API keys travel in the "
            "request body; terminate TLS in front of this or they go out in plaintext."
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
