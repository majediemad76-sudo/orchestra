"""Streamlit front end -- an observer, not a second controller.

Every decision still belongs to ``controller.run_task``. This module imports
it, hands it three hooks, and renders what comes back. No round counting, no
budget arithmetic, no escalation policy: duplicating any of that here would
create a second source of truth that drifts from the first.

Threading contract, which Streamlit makes non-negotiable:

  * ``run_task`` runs in a background thread. It is long-running and blocking,
    and Streamlit's script thread must stay free to re-render.
  * That thread touches ``st.session_state`` and ``st.*`` never -- neither is
    thread-safe, and Streamlit raises without a script context. Two queues are
    the only channels across the boundary.
  * The thread and its queues live in ``st.session_state`` so they survive the
    reruns Streamlit performs on every interaction.
  * Live output comes from draining the outbox and calling ``st.rerun()``, not
    from a loop that would pin the script thread and freeze the page.

Two queues, because escalation is the one thing that has to travel the other
way:

  * ``outbox``  -- thread to UI: progress events, questions, terminal results.
  * ``answers`` -- UI to thread: the option a human picked. The worker thread
    blocks on it, which is exactly right: the Controller is mid-decision and
    must not proceed until the answer arrives.

Cancellation is a third channel, a ``threading.Event`` read by the Controller
at the top of each round. It is not a queue because it is not a message -- it
is a latch, and it must be observable even while the thread is blocked
elsewhere.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv

from budget import SUBSCRIPTION_EQUIVALENT
from controller import run_task
from schemas import Question, Task

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

# Newest runs only. The directory grows without bound and nobody scrolls
# past the first page of history.
HISTORY_LIMIT = 50

# How long the script thread idles before re-rendering while a run is in
# flight. Short enough to feel live, long enough not to spin.
POLL_SECONDS = 0.4

# Queue messages that are not progress events. Prefixed so they cannot collide
# with an event name the Controller emits.
FINISHED = "_finished"
FAILED = "_failed"
ESCALATION = "_escalation"

# How long the worker thread waits for a human before giving up. Long enough
# to read a question and think; short enough that a forgotten tab does not
# leave a thread parked forever.
ESCALATION_TIMEOUT_SECONDS = 300

API_KEYS = {
    "XAI_API_KEY": ("Manager", "grok-4.6"),
    "ANTHROPIC_API_KEY": ("Worker", "claude-sonnet-5"),
    "GOOGLE_API_KEY": ("Critic", "gemini-3.1-flash-lite"),
}

STATUS_LABELS = {
    "accepted": ("✅", "Accepted", "The Critic passed the output."),
    "accepted_by_user": ("✅", "Accepted by you", "You chose to keep the best output so far."),
    "stopped_by_user": ("⏹️", "Stopped by you", "You ended the run at an escalation."),
    "stopped_by_flag": ("⏹️", "Stopped", "Cancelled with the Stop button."),
    "escalated_unanswered": ("⚠️", "Escalation unanswered", "The loop asked a question nobody answered."),
    "max_rounds": ("⚠️", "Round limit reached", "The output was never accepted within the round cap."),
}

VERDICT_ICONS = {"accept": "✅", "revise": "🔁", "escalate": "⚠️"}

COST_BASIS_NOTE = (
    "Subscription-equivalent — Claude Code reports what the session would cost "
    "at API rates. On a personal Claude plan it draws on included usage, so it "
    "was not billed against API credit."
)


class EscalationTimeout(RuntimeError):
    """Nobody answered an escalation within the timeout."""


def _run_in_background(
    task: Task,
    outbox: "queue.Queue[Tuple[str, Dict[str, Any]]]",
    answers: "queue.Queue[str]",
    stop_flag: threading.Event,
) -> None:
    """Body of the worker thread. Speaks only through the two queues.

    Nothing here may touch Streamlit -- not session_state, not st.*, not even
    a spinner. Every exit path posts exactly one terminal message so the UI
    can never be left waiting on a thread that has already died.
    """

    def on_progress(event: str, data: Dict[str, Any]) -> None:
        outbox.put((event, data))

    def on_escalation(question: Question) -> str:
        """Hand the question to the UI and block until a human answers.

        Blocking is the correct behaviour, not a limitation: the Controller is
        mid-decision and the whole point of an escalation is that it must not
        proceed on a guess. Only this thread is parked -- Streamlit's script
        thread keeps rendering.
        """
        outbox.put((ESCALATION, {"text": question.text, "options": list(question.options)}))
        try:
            return answers.get(timeout=ESCALATION_TIMEOUT_SECONDS)
        except queue.Empty:
            # Raising is what keeps the thread from hanging on a tab nobody
            # came back to. The Controller sees the hook fail rather than
            # return a made-up answer -- which is the same outcome as the user
            # declining to choose, and never an option they did not pick.
            raise EscalationTimeout(
                f"No answer within {ESCALATION_TIMEOUT_SECONDS}s: {question.text}"
            ) from None

    try:
        summary = run_task(
            task,
            on_progress=on_progress,
            on_escalation=on_escalation,
            stop_flag=stop_flag,
        )
        outbox.put((FINISHED, summary))
    except EscalationTimeout as exc:
        outbox.put((FAILED, {"error": str(exc), "timeout": True}))
    except Exception as exc:  # noqa: BLE001 -- the thread must report, not vanish
        outbox.put((FAILED, {"error": f"{type(exc).__name__}: {exc}"}))


# --- state -----------------------------------------------------------------


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "queue": None,
        "answers": None,
        "stop_flag": None,
        "thread": None,
        "running": False,
        "rounds": {},
        "summary": None,
        "error": None,
        "pending_question": None,
        "escalation_seq": 0,
        "answered": [],
        "live_spend": 0.0,
        "used_subscription_worker": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _drain_queue() -> None:
    """Move everything the thread has posted into session_state.

    Runs on the script thread, which is the only one allowed to write here.
    """
    outbox = st.session_state["queue"]
    if outbox is None:
        return

    while True:
        try:
            event, data = outbox.get_nowait()
        except queue.Empty:
            break

        if event == FINISHED:
            st.session_state["summary"] = data
            st.session_state["running"] = False
        elif event == FAILED:
            st.session_state["error"] = data["error"]
            st.session_state["running"] = False
        elif event == ESCALATION:
            # The thread is now parked on the answer queue. Polling stops
            # until a human submits, so the radio is not re-rendered underneath
            # them mid-selection.
            st.session_state["pending_question"] = data
            st.session_state["escalation_seq"] += 1
        elif event == "run_end":
            st.session_state["live_spend"] = data["budget"]["spent_usd"]
        else:
            _record_round_event(event, data)

    # A thread that died without posting a terminal message would otherwise
    # leave the page polling forever. A thread parked on the answer queue is
    # alive, so this cannot misfire during an escalation.
    thread = st.session_state["thread"]
    if st.session_state["running"] and thread is not None and not thread.is_alive():
        if outbox.empty():
            st.session_state["running"] = False
            st.session_state["pending_question"] = None


def _record_round_event(event: str, data: Dict[str, Any]) -> None:
    """Group stage events by round for rendering."""
    round_no = data.get("round")
    if round_no is None:
        return
    bucket = st.session_state["rounds"].setdefault(
        round_no, {"manager": None, "worker": None, "critic": None}
    )
    if event == "round_start":
        st.session_state["live_spend"] = data.get("spent_usd", st.session_state["live_spend"])
    elif event == "manager_plan":
        bucket["manager"] = data
    elif event == "worker_output":
        bucket["worker"] = data
        # Remembered for the cost metric: once any round has run through Claude
        # Code, the total is no longer purely an API bill.
        if data.get("cost_basis") == SUBSCRIPTION_EQUIVALENT:
            st.session_state["used_subscription_worker"] = True
    elif event == "critic_verdict":
        bucket["critic"] = data
        st.session_state["live_spend"] = data.get("spent_usd", st.session_state["live_spend"])


def _start_run(goal: str, budget: float, max_rounds: int) -> None:
    outbox: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
    answers: "queue.Queue[str]" = queue.Queue()
    stop_flag = threading.Event()
    task = Task(goal=goal, budget_usd=budget, max_rounds=max_rounds)
    thread = threading.Thread(
        target=_run_in_background,
        args=(task, outbox, answers, stop_flag),
        name="orchestrator-run",
        daemon=True,  # a closed tab must not keep the process alive
    )
    st.session_state.update(
        queue=outbox,
        answers=answers,
        stop_flag=stop_flag,
        thread=thread,
        running=True,
        rounds={},
        summary=None,
        error=None,
        pending_question=None,
        answered=[],
        live_spend=0.0,
        used_subscription_worker=False,
    )
    thread.start()


def _submit_answer(label: str) -> None:
    """Hand the user's choice to the blocked thread and resume polling."""
    question = st.session_state["pending_question"]
    st.session_state["answers"].put(label)
    st.session_state["answered"].append({"text": question["text"], "answer": label})
    st.session_state["pending_question"] = None


def _request_stop() -> None:
    """Set the cancellation latch, and unblock the thread if it is waiting.

    Order matters. Setting the flag alone would leave a thread parked on an
    unanswered question for the full timeout -- the Controller cannot read the
    latch while it is inside the escalation hook. Feeding the queue an empty
    answer releases it into the Controller's own stop path, which ends the run
    with a proper summary instead of an exception.
    """
    stop_flag = st.session_state["stop_flag"]
    if stop_flag is not None:
        stop_flag.set()
    if st.session_state["pending_question"] is not None:
        st.session_state["answers"].put("")
        st.session_state["pending_question"] = None


# --- reading run logs ------------------------------------------------------


@st.cache_data(show_spinner=False)
def _parse_run_log(path_str: str, mtime: float, size: int) -> Dict[str, Any]:
    """Turn one JSONL run log into a render-ready record.

    Cached on (path, mtime, size): during a live run this page re-renders every
    0.4s, and re-parsing every historical log each time would make the poll
    loop quadratic in the size of the runs directory.

    Fault tolerance is the point, not a nicety. These files are appended to
    while a run is in flight, so the last line is routinely half-written; a
    crashed run has no ``run_end`` at all. Every malformed line is counted and
    skipped, and a log that yields nothing still returns a record rather than
    raising -- one bad file must not take down the History tab.
    """
    record: Dict[str, Any] = {
        "path": path_str,
        "run_id": Path(path_str).stem,
        "timestamp": "",
        "goal": "",
        "status": "incomplete",
        "rounds": 0,
        "score": None,
        "cost_usd": 0.0,
        "escalations": [],
        "round_details": {},
        "bad_lines": 0,
        "is_eval": Path(path_str).stem.startswith("eval-"),
    }

    try:
        text = Path(path_str).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        record["status"] = "unreadable"
        record["goal"] = f"({exc.strerror})"
        return record

    accrued_cost = 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            record["bad_lines"] += 1
            continue
        if not isinstance(event, dict):
            record["bad_lines"] += 1
            continue

        name = event.get("event")
        if name == "run_start":
            record["timestamp"] = event.get("ts", "")
            record["goal"] = (event.get("task") or {}).get("goal", "")
        elif name == "run_end":
            record["status"] = event.get("status", "incomplete")
            record["rounds"] = event.get("rounds", record["rounds"])
            record["score"] = event.get("score")
            record["cost_usd"] = (event.get("budget") or {}).get("spent_usd", accrued_cost)
            record["escalations"] = [
                item.get("trigger", "?") for item in event.get("escalations") or []
            ]
        elif name in {"manager_plan", "worker_output", "critic_verdict"}:
            accrued_cost += event.get("cost_usd") or 0.0
            round_no = event.get("round")
            if round_no is None:
                continue
            record["rounds"] = max(record["rounds"], round_no)
            bucket = record["round_details"].setdefault(
                str(round_no), {"manager": None, "worker": None, "critic": None}
            )
            bucket[_STAGE_KEYS[name]] = _normalise_stage(name, event)
        elif name == "escalation":
            record["escalations"].append(event.get("trigger", "?"))

    if record["status"] == "incomplete":
        # No run_end: the process died, or the run is still going. Report what
        # the individual steps add up to rather than nothing.
        record["cost_usd"] = accrued_cost
        last = record["round_details"].get(str(record["rounds"]), {})
        critic = last.get("critic")
        record["score"] = critic["score"] if critic else None

    return record


_STAGE_KEYS = {"manager_plan": "manager", "worker_output": "worker", "critic_verdict": "critic"}


def _normalise_stage(name: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Map a log record onto the shape the live renderer already expects.

    The log and the progress hook carry the same facts in different shapes --
    the log nests the plan and the verdict, the hook flattens them. Normalising
    here means History and Live Run share one renderer, so the two views cannot
    drift apart.
    """
    if name == "manager_plan":
        plan = event.get("plan") or {}
        return {
            "round": event.get("round"),
            "plan": plan.get("plan", ""),
            "worker_type": plan.get("worker_type", "?"),
            "acceptance_criteria": plan.get("acceptance_criteria") or [],
            "cost_usd": event.get("cost_usd", 0.0),
            "model": event.get("model", "?"),
        }
    if name == "worker_output":
        return {
            "round": event.get("round"),
            "worker_type": event.get("worker_type", "?"),
            "ok": event.get("ok", False),
            "result": event.get("result", ""),
            "notes": event.get("notes", ""),
            "failure_reason": event.get("failure_reason", ""),
            "cost_usd": event.get("cost_usd", 0.0),
            "cost_basis": event.get("cost_basis"),
            "model": event.get("model", "?"),
        }
    verdict = event.get("verdict") or {}
    return {
        "round": event.get("round"),
        "score": verdict.get("score", 0),
        "verdict": verdict.get("verdict", "?"),
        "met_criteria": verdict.get("met_criteria") or [],
        "failed_criteria": verdict.get("failed_criteria") or [],
        "fix_instruction": verdict.get("fix_instruction", ""),
    }


def _load_runs(directory: Path = RUNS_DIR) -> List[Dict[str, Any]]:
    """Newest first. A directory that does not exist is simply empty."""
    try:
        paths = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    records = []
    for path in paths[:HISTORY_LIMIT]:
        try:
            stat = path.stat()
            records.append(_parse_run_log(str(path), stat.st_mtime, stat.st_size))
        except OSError:
            continue
    return records


def _short_timestamp(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return iso or "—"


# --- rendering -------------------------------------------------------------


def _render_sidebar() -> Tuple[float, int]:
    with st.sidebar:
        st.header("⚙️ Run Settings")
        budget = st.slider(
            "Budget Ceiling",
            min_value=0.05,
            max_value=5.00,
            value=0.50,
            step=0.05,
            format="$%.2f",
            help="Hard dollar cap. The Controller stops and asks when it is crossed.",
            disabled=st.session_state["running"],
        )
        max_rounds = st.slider(
            "Max Rounds",
            min_value=1,
            max_value=5,
            value=3,
            help="One round is Manager → Worker → Critic.",
            disabled=st.session_state["running"],
        )

        st.divider()
        st.subheader("🔑 API Keys")
        # Presence only. The value is never read into a widget, a label, or a
        # log line -- not even a masked prefix, which is still key material.
        missing = []
        for var, (role, model) in API_KEYS.items():
            present = bool(os.environ.get(var, "").strip())
            if not present:
                missing.append(role)
            st.markdown(
                f"{'✅' if present else '❌'} **{role}** · `{model}` — "
                f"{'Yes' if present else 'No'}"
            )
        if missing:
            st.warning(
                f"Missing key for: {', '.join(missing)}. "
                "Copy `.env.example` to `.env` and fill it in."
            )

    return budget, max_rounds


def _round_headline(number: int, stages: Dict[str, Any]) -> str:
    critic = stages["critic"]
    if critic:
        icon = VERDICT_ICONS.get(critic["verdict"], "•")
        return f"{icon}  Round {number} · Score {critic['score']} · {critic['verdict']}"
    if stages["worker"]:
        return f"⏳  Round {number} · reviewing"
    if stages["manager"]:
        return f"⏳  Round {number} · working"
    return f"⏳  Round {number} · planning"


def _render_round(number: int, stages: Dict[str, Any], expanded: bool) -> None:
    manager, worker, critic = stages["manager"], stages["worker"], stages["critic"]

    with st.expander(_round_headline(number, stages), expanded=expanded):
        st.markdown("##### 🧭 Manager Plan")
        if manager:
            with st.container(border=True):
                st.write(manager["plan"])
                st.caption(f"Worker backend: `{manager['worker_type']}`")
            st.markdown("##### ✅ Acceptance Criteria")
            for criterion in manager["acceptance_criteria"]:
                st.markdown(f"- {criterion}")
        else:
            st.caption("Waiting…")

        st.divider()
        st.markdown("##### 🛠️ Worker Output")
        if worker:
            if worker["ok"]:
                st.code(worker["result"], language=None)
            else:
                st.error(worker.get("failure_reason") or "The worker produced no usable output.")
            note = f"`{worker.get('model', '?')}` · ${worker.get('cost_usd', 0.0):.4f}"
            if worker.get("cost_basis") == SUBSCRIPTION_EQUIVALENT:
                note += " · subscription-equivalent"
            st.caption(note)
        else:
            st.caption("Waiting…")

        st.divider()
        st.markdown("##### 🔍 Critic Verdict")
        if critic:
            left, right = st.columns([1, 3])
            left.metric("Score", critic["score"])
            with right:
                if critic["failed_criteria"]:
                    st.markdown("**Failed criteria**")
                    for criterion in critic["failed_criteria"]:
                        st.markdown(f"- ❌ {criterion}")
                else:
                    st.markdown("**Failed criteria** — none.")
                if critic["met_criteria"]:
                    st.caption(f"Met: {', '.join(critic['met_criteria'])}")
            if critic["fix_instruction"]:
                st.info(f"**Fix instruction** — {critic['fix_instruction']}")
        else:
            st.caption("Waiting…")


def _render_escalation() -> None:
    """The one place the UI stops observing and starts participating.

    The options come from the Controller and are rendered verbatim: this form
    cannot invent a choice, only relay one. That is what keeps the escalation
    policy in ``controller.py`` even though the question is answered here.
    """
    question = st.session_state["pending_question"]
    seq = st.session_state["escalation_seq"]

    st.warning("⏸️ The loop needs your decision before it can continue.")
    with st.container(border=True):
        st.markdown(f"**{question['text']}**")
        choice = st.radio(
            "Options",
            question["options"],
            key=f"escalation_choice_{seq}",  # a fresh key per question, so a
            label_visibility="collapsed",     # stale selection cannot leak over
        )
        submitted = st.button("Submit Answer", type="primary", key=f"escalation_submit_{seq}")
        st.caption(
            f"The run stops on its own if there is no answer within "
            f"{ESCALATION_TIMEOUT_SECONDS} seconds."
        )
    if submitted:
        _submit_answer(choice)
        st.rerun()


def _render_outcome() -> None:
    summary = st.session_state["summary"]
    if summary is None:
        return

    icon, label, blurb = STATUS_LABELS.get(
        summary["status"], ("ℹ️", summary["status"], "")
    )
    st.subheader(f"{icon} {label}")
    if blurb:
        st.caption(blurb)

    score, rounds, cost = st.columns(3)
    score.metric("Final Score", summary["score"] if summary["score"] is not None else "—")
    rounds.metric("Rounds Used", summary["rounds"])
    cost.metric("Total Cost", f"${summary['budget']['spent_usd']:.4f}")
    if st.session_state["used_subscription_worker"]:
        # The total mixes billed API usage with Claude Code's API-equivalent
        # figure. Saying so is the difference between a number and a bill.
        cost.caption(COST_BASIS_NOTE)

    if summary["escalations"]:
        st.markdown("##### 🙋 Escalations")
        for item in summary["escalations"]:
            answer = item["answer"] or "unanswered (stopped)"
            st.markdown(f"- `{item['trigger']}` → {answer}")

    st.markdown("##### 📄 Final Output")
    st.code(summary["result"] or "—", language=None)
    st.caption(f"Run log: `{summary['log_path']}`")


def _render_history() -> None:
    """Read-only view over runs/*.jsonl. Touches no controller state."""
    st.markdown("### 📜 Run History")

    records = _load_runs()
    if not records:
        st.info(
            "No runs found yet. Start one from the Live Run tab, or run "
            "`make run` in a terminal — each run writes a JSONL log to `runs/`."
        )
        return

    hide_evals = st.checkbox(
        "Hide eval-suite runs",
        value=any(not r["is_eval"] for r in records),
        help="Runs started by scripts/eval_suite.py share this directory.",
    )
    if hide_evals:
        records = [r for r in records if not r["is_eval"]]
    if not records:
        st.info("Only eval-suite runs are present. Untick the box above to see them.")
        return

    damaged = sum(r["bad_lines"] for r in records)
    if damaged:
        # Worth saying out loud: a truncated tail is normal for a run that is
        # still writing, but a steady count means something else is wrong.
        st.caption(f"{damaged} unreadable line(s) were skipped across these logs.")

    st.dataframe(
        [
            {
                "Timestamp": _short_timestamp(record["timestamp"]),
                "Task Goal": (record["goal"][:60] + "…") if len(record["goal"]) > 60 else record["goal"],
                "Rounds Used": record["rounds"],
                # Every cell in a column must share a type: Arrow rejects a
                # column of ints with one "—" in it, and the table silently
                # falls back to a worse rendering.
                "Final Score": "—" if record["score"] is None else str(record["score"]),
                "Total Cost": f"${record['cost_usd']:.4f}",
                "Status": record["status"],
                "Escalation": ", ".join(record["escalations"]) or "—",
            }
            for record in records
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Details")
    for record in records:
        icon, label, _ = STATUS_LABELS.get(record["status"], ("•", record["status"], ""))
        goal = record["goal"][:60] or "(no goal recorded)"
        header = (
            f"{icon}  {_short_timestamp(record['timestamp'])} · {goal} · "
            f"{record['rounds']} round(s) · ${record['cost_usd']:.4f}"
        )
        with st.expander(header):
            top = st.columns(3)
            top[0].metric("Final Score", "—" if record["score"] is None else record["score"])
            top[1].metric("Rounds Used", record["rounds"])
            top[2].metric("Total Cost", f"${record['cost_usd']:.4f}")
            st.markdown(f"**Status** — {label} (`{record['status']}`)")
            if record["goal"]:
                st.markdown(f"**Task Goal** — {record['goal']}")
            if record["escalations"]:
                st.markdown("**Escalations** — " + ", ".join(f"`{e}`" for e in record["escalations"]))

            details = record["round_details"]
            if details:
                for key in sorted(details, key=lambda k: int(k)):
                    # Same renderer as the live tab, so the two views cannot
                    # describe the same run differently.
                    _render_round(int(key), details[key], expanded=False)
            else:
                st.caption("This log holds no completed rounds.")

            raw = _read_raw_log(record["path"])
            if raw is not None:
                st.download_button(
                    "⬇️ Download JSONL",
                    data=raw,
                    file_name=Path(record["path"]).name,
                    mime="application/x-ndjson",
                    key=f"download_{record['run_id']}",
                )
            st.caption(f"`{record['path']}`")


def _read_raw_log(path_str: str) -> bytes | None:
    """Raw bytes for the download button. Missing file is not an error here."""
    try:
        return Path(path_str).read_bytes()
    except OSError:
        return None


def main() -> None:
    st.set_page_config(page_title="Multi-Model Orchestrator", page_icon="🎛️", layout="wide")
    load_dotenv(ROOT / ".env")
    _init_state()
    _drain_queue()

    st.title("🎛️ Multi-Model Orchestrator")
    st.caption(
        "Manager `grok-4.6` (xAI)  →  Worker `claude-sonnet-5` (Anthropic)  →  "
        "Critic `gemini-3.1-flash-lite` (Google) — the loop itself is plain Python."
    )

    budget, max_rounds = _render_sidebar()

    live_tab, history_tab = st.tabs(["🚀 Live Run", "📜 History"])
    with history_tab:
        # Rendered on every rerun because Streamlit tabs are client-side --
        # hence the cache on the log parser.
        _render_history()

    with live_tab:
        _render_live(budget, max_rounds)

    # The live-update mechanism: yield briefly, then let Streamlit re-run the
    # script, which drains whatever the thread posted in the meantime.
    #
    # Polling pauses while a question is on screen. The thread is blocked and
    # has nothing to post, and re-rendering every 0.4s would reset the radio
    # under the user's cursor.
    if st.session_state["running"] and st.session_state["pending_question"] is None:
        time.sleep(POLL_SECONDS)
        st.rerun()


def _render_live(budget: float, max_rounds: int) -> None:
    goal = st.text_area(
        "Task Goal",
        placeholder="e.g. Write a two-sentence summary of what this orchestrator does.",
        height=120,
        disabled=st.session_state["running"],
    )
    start_col, stop_col, _ = st.columns([1, 1, 4])
    if start_col.button(
        "▶️ Run Task", type="primary", disabled=st.session_state["running"],
        width="stretch",
    ):
        if goal.strip():
            _start_run(goal.strip(), budget, max_rounds)
            st.rerun()
        else:
            st.warning("Write a task first.")
    # Enabled only while a run exists to cancel, and never disabled by the same
    # condition that disables the start button -- an emergency stop that is
    # greyed out during an emergency is not a stop button.
    if stop_col.button(
        "⏹️ Stop", disabled=not st.session_state["running"], width="stretch"
    ):
        _request_stop()
        st.rerun()

    if st.session_state["running"] and st.session_state["pending_question"] is None:
        stop_flag = st.session_state["stop_flag"]
        if stop_flag is not None and stop_flag.is_set():
            st.info("⏹️ Stop requested — finishing the current round.")
        else:
            st.info(
                f"⏳ Running… ${st.session_state['live_spend']:.4f} of "
                f"${budget:.2f} spent so far."
            )

    if st.session_state["pending_question"]:
        _render_escalation()

    if st.session_state["error"]:
        st.error(f"The run stopped with an error: {st.session_state['error']}")

    rounds = st.session_state["rounds"]
    if rounds:
        st.divider()
        st.markdown("### Rounds")
        last = max(rounds)
        for number in sorted(rounds):
            _render_round(number, rounds[number], expanded=number == last)

    if st.session_state["summary"]:
        st.divider()
        _render_outcome()


# Streamlit executes this file as "__main__", so the guard costs nothing there
# and keeps the module importable for tests that drive it headlessly.
if __name__ == "__main__":
    main()
