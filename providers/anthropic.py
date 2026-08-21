"""Anthropic Messages API -- the text Worker.

Raw HTTP rather than the official SDK, deliberately: all three vendors sit
behind the same ``ProviderResult`` and the same retry policy, and one uniform
transport is easier to reason about than three SDK surfaces with three
different error hierarchies. The cost is that the request shape below is
maintained by hand.

Claude has no ``response_format`` parameter, so structured output is a forced
tool call instead::

    tools=[{"name": "emit_result", "input_schema": <schema>}]
    tool_choice={"type": "tool", "name": "emit_result"}

The schema itself is built in ``schema_utils.anthropic_tool``, which is where
the reasoning for that mechanism lives.

Note on thinking: on the Claude API a forced ``tool_choice`` coexists with the
model's default adaptive thinking, so ``thinking`` is deliberately left unset
here. Bedrock is the exception -- it requires ``thinking: {"type": "disabled"}``
alongside a forced tool -- which matters only if this client is ever pointed at
a Bedrock endpoint.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from . import ProviderResult
from .retry_utils import ProviderError, raise_for_status, with_retry
from .schema_utils import anthropic_tool

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
TOOL_NAME = "emit_result"


@with_retry
def call_structured(
    schema_model: type[BaseModel],
    system: str,
    user: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8000,
    timeout: float = 180.0,
) -> ProviderResult:
    """One Messages call whose only legal reply is a ``schema_model`` object."""
    tool = anthropic_tool(schema_model, name=TOOL_NAME)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [tool],
        # Without this line the tool is merely offered and the model may
        # answer in prose instead. With it, the schema is the only exit.
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(API_URL, headers=headers, json=payload)
    raise_for_status("anthropic", response, api_key)
    body = response.json()

    data = _extract_tool_input(body)
    usage = body.get("usage", {}) or {}
    return ProviderResult(
        data=data,
        model=model,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        raw=body,
    )


def _extract_tool_input(body: dict[str, Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = body.get("content", []) or []
    for block in blocks:
        if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            return block.get("input", {})
    # Unreachable while tool_choice is forced. Kept because the alternative to
    # a salvage attempt is discarding a response that is already paid for: a
    # future model, or a relaxed tool_choice, could put us here.
    for block in blocks:
        if block.get("type") == "text":
            parsed = _loads_or_none(block.get("text", ""))
            if parsed is not None:
                return parsed
    raise ProviderError("anthropic", f"no {TOOL_NAME} tool call in response: {str(body)[:400]}")


def _loads_or_none(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@with_retry
def call_text(
    system: str,
    user: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8000,
    timeout: float = 180.0,
) -> ProviderResult:
    """Unstructured completion for the text Worker.

    The Worker is the one role whose product is the deliverable itself, not a
    control signal, so imposing a schema on it would only get in the way. The
    Critic judges the prose; the Controller never parses it.
    """
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(API_URL, headers=headers, json=payload)
    raise_for_status("anthropic", response, api_key)
    body = response.json()
    text = "".join(
        block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
    )
    usage = body.get("usage", {}) or {}
    return ProviderResult(
        data={"text": text},
        model=model,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        raw=body,
    )
