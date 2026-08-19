"""HTTP clients for the three vendors, behind one result type.

Uniformity here is what lets the Controller charge the budget, log a step, and
retry a call without knowing which vendor produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResult:
    """One provider call, normalised.

    ``data`` is already schema-shaped; the role layer validates it into a
    Pydantic model. Token counts come from three differently-named vendor
    fields and are unified here so ``budget.py`` needs only one code path --
    they default to 0 so an unreported usage block undercounts rather than
    crashes mid-run.

    ``raw`` is retained for the run log: when a provider does something
    surprising, the normalised view is exactly the wrong thing to have kept.
    """

    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
