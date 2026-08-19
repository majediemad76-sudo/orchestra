"""Dollar accounting for a run.

Deliberately dumb: this module adds numbers and answers "over the line yet?".
It never stops anything. The Controller reads ``exceeded`` and decides, which
keeps the one irreversible decision in the system -- spending money -- in code
that can be read in full on one screen.

Prices are USD per million tokens as of August 2026 and will go stale. They are
a table, not a lookup: a pricing API call on every step would itself cost
latency and add a failure mode to the accountant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# How a charge was actually incurred. Both are real spend; they are not the
# same kind of spend, and a log that blurs them cannot answer "what did this
# run put on the API bill?".
API_BILLED = "api_billed"
SUBSCRIPTION_EQUIVALENT = "subscription_equivalent"

PRICING: Dict[str, Tuple[float, float]] = {
    # model: (input $/1M, output $/1M)
    "claude-sonnet-5": (2.00, 10.00),
    "grok-4.6": (2.00, 6.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call.

    An unknown model costs 0 rather than raising: a mispriced model must not
    take down a run that is otherwise going fine. It stays visible -- it
    appears in ``by_model()`` with a zero total, which is a conspicuous thing
    to see next to a real spend.
    """
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


@dataclass
class BudgetEntry:
    step: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class BudgetGuard:
    """Accumulates spend and reports whether the ceiling has been crossed.

    Detection is after the fact: a call is charged once it has already
    happened, so the ceiling is a stop-line rather than a pre-authorisation.
    The overshoot is bounded by one step's cost, and the Controller checks at
    the top of each round -- before committing to another three calls.

    ``limit_usd`` is mutable because the user is allowed to raise it in
    response to a budget escalation. That is the only thing that writes to it.
    """

    limit_usd: float
    spent_usd: float = 0.0
    entries: List[BudgetEntry] = field(default_factory=list)

    def charge(self, step: str, model: str, input_tokens: int, output_tokens: int) -> BudgetEntry:
        cost = price(model, input_tokens, output_tokens)
        return self._record(BudgetEntry(step, model, input_tokens, output_tokens, cost))

    def charge_usd(self, step: str, model: str, cost_usd: float) -> BudgetEntry:
        """Charge a known dollar amount, for backends that report cost directly.

        Claude Code headless returns ``total_cost_usd`` in its JSON envelope --
        an authoritative figure covering a whole multi-turn session, which no
        token count of ours could reconstruct.
        """
        return self._record(BudgetEntry(step, model, 0, 0, max(0.0, cost_usd)))

    def _record(self, entry: BudgetEntry) -> BudgetEntry:
        self.entries.append(entry)
        self.spent_usd += entry.cost_usd
        return entry

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.spent_usd

    @property
    def exceeded(self) -> bool:
        return self.spent_usd >= self.limit_usd

    def by_model(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for entry in self.entries:
            totals[entry.model] = totals.get(entry.model, 0.0) + entry.cost_usd
        return totals

    def summary(self) -> Dict[str, object]:
        return {
            "limit_usd": round(self.limit_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "calls": len(self.entries),
            "by_model": {k: round(v, 6) for k, v in self.by_model().items()},
        }
