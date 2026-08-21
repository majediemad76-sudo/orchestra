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

On expiry a surface returns the *empty answer* -- exactly what its stop control
posts -- and records the timeout as a flag of its own. It does not raise.

The reason is not stylistic. Raising aborts ``run_task`` from the inside, so
the summary it was about to return is destroyed: the rounds, the
spend, and whatever output had already been accepted all go with it. The user
loses work that was already paid for, in order to report that nobody clicked a
button. Returning the empty answer lets the loop reach its own ending -- the
Controller reads "no answer" as stop, which is what it means -- and hand back a
complete summary.

"Nobody was there" is still a different fact from "the operator ended it", and
both surfaces keep it. It travels as a flag next to the result rather than as
the shape of the exit, which is the only way to keep the result at all.
"""

from __future__ import annotations

# Long enough to read a question and think about it, short enough that a tab
# nobody came back to does not park a thread for the life of the process.
ESCALATION_TIMEOUT_SECONDS = 300.0


# The answer a surface hands back when nobody replied in time. Deliberately the
# same value the stop controls post: to the Controller both mean "no option was
# chosen", and it already knows what to do with that.
NO_ANSWER = ""
