"""The observer half of the escalation contract, shared by every front end.

``run_task``'s ``on_escalation`` hook is allowed to block -- that is the whole
point of it. The Controller is mid-decision and must not continue on a guess,
so it waits for a human. What the Controller cannot decide is how long a
particular front end should wait, because that depends on who is on the other
end and whether anyone is still there.

So the ceiling lives here rather than in ``controller.py``: it is a property of
the surface, not of the loop, and nothing in the state machine reads it. It
lives here rather than in either surface because ``app.py`` and ``api.py`` must
agree on it and neither can import the other -- the Streamlit app would drag
Streamlit into the API process, and the API would drag FastAPI into the UI.

Both surfaces raise ``EscalationTimeout`` rather than returning an empty answer.
The difference matters to whoever is reading the result afterwards: an empty
answer is indistinguishable from someone deliberately choosing to stop, and
"nobody was there" is a different fact about the run than "the operator ended
it". Neither is turned into a default choice; the run ends either way.
"""

from __future__ import annotations

# Long enough to read a question and think about it, short enough that a tab
# nobody came back to does not park a thread for the life of the process.
ESCALATION_TIMEOUT_SECONDS = 300.0


class EscalationTimeout(RuntimeError):
    """Nobody answered an escalation before the surface gave up waiting.

    Raised by the ``on_escalation`` hook, which means it surfaces out of
    ``run_task`` to whoever called it. Every front end catches it on the way to
    its own terminal record -- and, importantly, on the way to releasing the
    credentials the run was holding.
    """
