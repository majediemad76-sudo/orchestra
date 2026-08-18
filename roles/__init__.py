"""Manager, Worker, Critic -- one model and one schema each.

Prompts live in ``prompts/*.md`` rather than in string literals so they can be
edited, diffed, and reviewed as the specifications they are.
"""

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")
