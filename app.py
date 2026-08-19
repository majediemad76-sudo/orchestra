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

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv

from controller import run_task
from schemas import Question, Task

ROOT = Path(__file__).resolve().parent

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
    "XAI_API_KEY": "Manager · grok-4.6",
    "ANTHROPIC_API_KEY": "Worker · claude-sonnet-5",
    "GOOGLE_API_KEY": "Critic · gemini-3.1-flash-lite",
}

STATUS_LABELS = {
    "accepted": ("✅", "پذیرفته شد"),
    "accepted_by_user": ("✅", "با تأیید کاربر پذیرفته شد"),
    "stopped_by_user": ("⏹️", "با انتخاب کاربر متوقف شد"),
    "stopped_by_flag": ("⏹️", "با دکمه‌ی توقف متوقف شد"),
    "escalated_unanswered": ("⚠️", "ارجاع بی‌پاسخ ماند"),
    "max_rounds": ("⚠️", "به سقف دورها رسید"),
}


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
                f"پاسخی ظرف {ESCALATION_TIMEOUT_SECONDS} ثانیه دریافت نشد: {question.text}"
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


# --- rendering -------------------------------------------------------------


def _render_sidebar() -> Tuple[float, int]:
    with st.sidebar:
        st.header("تنظیمات اجرا")
        budget = st.slider(
            "سقف بودجه (دلار)",
            min_value=0.05,
            max_value=2.00,
            value=0.10,
            step=0.05,
            disabled=st.session_state["running"],
        )
        max_rounds = st.slider(
            "حداکثر دور",
            min_value=1,
            max_value=5,
            value=3,
            disabled=st.session_state["running"],
        )

        st.divider()
        st.subheader("کلیدهای API")
        # Presence only. The value is never read into a widget, a label, or a
        # log line -- not even a masked prefix, which is still key material.
        for var, role in API_KEYS.items():
            present = bool(os.environ.get(var, "").strip())
            st.write(f"{'✅' if present else '❌'} {role} — {'بله' if present else 'خیر'}")
        if not all(os.environ.get(var, "").strip() for var in API_KEYS):
            st.warning("یک یا چند کلید تنظیم نشده. `.env` را از روی `.env.example` بساز.")

    return budget, max_rounds


def _render_round(number: int, stages: Dict[str, Any], expanded: bool) -> None:
    critic = stages["critic"]
    headline = f"دور {number}"
    if critic:
        headline += f" — نمره {critic['score']} ({critic['verdict']})"
    elif stages["worker"]:
        headline += " — در حال ارزیابی…"
    elif stages["manager"]:
        headline += " — در حال اجرا…"

    with st.expander(headline, expanded=expanded):
        manager, worker = stages["manager"], stages["worker"]

        st.markdown("**نقشه‌ی Manager**")
        if manager:
            st.write(manager["plan"])
            st.caption(f"نوع عامل مجری: `{manager['worker_type']}`")
            st.markdown("**معیارهای پذیرش**")
            for criterion in manager["acceptance_criteria"]:
                st.markdown(f"- {criterion}")
        else:
            st.caption("در انتظار…")

        st.divider()
        st.markdown("**خروجی Worker**")
        if worker:
            if worker["ok"]:
                st.code(worker["result"], language=None)
            else:
                st.error(worker["failure_reason"] or "عامل مجری خروجی قابل استفاده‌ای نداد.")
        else:
            st.caption("در انتظار…")

        st.divider()
        st.markdown("**نظر Critic**")
        if critic:
            st.metric("نمره", critic["score"])
            if critic["failed_criteria"]:
                st.markdown("**معیارهای ردشده**")
                for criterion in critic["failed_criteria"]:
                    st.markdown(f"- ❌ {criterion}")
            else:
                st.markdown("همه‌ی معیارها پذیرفته شد.")
            if critic["fix_instruction"]:
                st.info(critic["fix_instruction"])
        else:
            st.caption("در انتظار…")


def _render_escalation() -> None:
    """The one place the UI stops observing and starts participating.

    The options come from the Controller and are rendered verbatim: this form
    cannot invent a choice, only relay one. That is what keeps the escalation
    policy in ``controller.py`` even though the question is answered here.
    """
    question = st.session_state["pending_question"]
    seq = st.session_state["escalation_seq"]

    st.warning("حلقه منتظر تصمیم توست.")
    with st.container(border=True):
        st.markdown(f"**{question['text']}**")
        choice = st.radio(
            "گزینه‌ها",
            question["options"],
            key=f"escalation_choice_{seq}",  # a fresh key per question, so a
            label_visibility="collapsed",     # stale selection cannot leak over
        )
        submitted = st.button("ثبت پاسخ", type="primary", key=f"escalation_submit_{seq}")
        st.caption(f"اگر تا {ESCALATION_TIMEOUT_SECONDS} ثانیه پاسخی ندهی، اجرا متوقف می‌شود.")
    if submitted:
        _submit_answer(choice)
        st.rerun()


def _render_outcome() -> None:
    summary = st.session_state["summary"]
    if summary is None:
        return

    icon, label = STATUS_LABELS.get(summary["status"], ("ℹ️", summary["status"]))
    st.subheader(f"{icon} {label}")

    left, middle, right = st.columns(3)
    left.metric("نمره‌ی نهایی", summary["score"] if summary["score"] is not None else "—")
    middle.metric("دورهای مصرف‌شده", summary["rounds"])
    right.metric("هزینه‌ی کل", f"${summary['budget']['spent_usd']:.4f}")

    if summary["escalations"]:
        st.markdown("**ارجاع‌ها به کاربر**")
        for item in summary["escalations"]:
            answer = item["answer"] or "بی‌پاسخ (توقف)"
            st.markdown(f"- `{item['trigger']}` → {answer}")

    st.markdown("**خروجی نهایی**")
    st.code(summary["result"] or "—", language=None)
    st.caption(f"لاگ: `{summary['log_path']}`")


def main() -> None:
    st.set_page_config(page_title="Multi-Model Orchestrator", page_icon="🎛️", layout="wide")
    load_dotenv(ROOT / ".env")
    _init_state()
    _drain_queue()

    st.title("🎛️ Multi-Model Orchestrator")
    st.caption("Manager · grok-4.6  →  Worker · claude-sonnet-5  →  Critic · gemini-3.1-flash-lite")

    budget, max_rounds = _render_sidebar()

    goal = st.text_area(
        "تسک",
        placeholder="مثلاً: یک خلاصه‌ی دوجمله‌ای از کاری که این ارکستراتور می‌کند بنویس.",
        height=120,
        disabled=st.session_state["running"],
    )
    start_col, stop_col = st.columns([1, 1])
    if start_col.button("اجرا", type="primary", disabled=st.session_state["running"],
                        use_container_width=True):
        if goal.strip():
            _start_run(goal.strip(), budget, max_rounds)
            st.rerun()
        else:
            st.warning("اول تسک را بنویس.")
    # Enabled only while a run exists to cancel, and never disabled by the same
    # condition that disables the start button -- an emergency stop that is
    # greyed out during an emergency is not a stop button.
    if stop_col.button("توقف", disabled=not st.session_state["running"],
                       use_container_width=True):
        _request_stop()
        st.rerun()

    if st.session_state["running"] and st.session_state["pending_question"] is None:
        stop_flag = st.session_state["stop_flag"]
        if stop_flag is not None and stop_flag.is_set():
            st.info("درخواست توقف ثبت شد؛ در پایان دور جاری متوقف می‌شود.")
        else:
            st.info(f"در حال اجرا… تاکنون ${st.session_state['live_spend']:.4f} خرج شده.")

    if st.session_state["pending_question"]:
        _render_escalation()

    if st.session_state["error"]:
        st.error(f"اجرا با خطا متوقف شد: {st.session_state['error']}")

    rounds = st.session_state["rounds"]
    if rounds:
        st.divider()
        last = max(rounds)
        for number in sorted(rounds):
            _render_round(number, rounds[number], expanded=number == last)

    if st.session_state["summary"]:
        st.divider()
        _render_outcome()

    # The live-update mechanism: yield briefly, then let Streamlit re-run the
    # script, which drains whatever the thread posted in the meantime.
    #
    # Polling pauses while a question is on screen. The thread is blocked and
    # has nothing to post, and re-rendering every 0.4s would reset the radio
    # under the user's cursor.
    if st.session_state["running"] and st.session_state["pending_question"] is None:
        time.sleep(POLL_SECONDS)
        st.rerun()


# Streamlit executes this file as "__main__", so the guard costs nothing there
# and keeps the module importable for tests that drive it headlessly.
if __name__ == "__main__":
    main()
