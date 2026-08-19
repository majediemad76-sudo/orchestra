"""xAI chat completions -- the Manager.

The endpoint is OpenAI-compatible, which buys the OpenAI structured-output
mechanism: ``response_format: {"type": "json_schema", ..., "strict": true}``.
Strict mode is the whole point -- without it the schema is advisory and the
Manager can hand back a plan the Controller cannot act on.

Its two hard requirements (``additionalProperties: false`` on every object,
every property in ``required``) are enforced in ``schema_utils.to_xai_schema``,
not here, so that the rules live next to the code that can violate them.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel

from . import ProviderResult
from .retry_utils import ProviderError, raise_for_status, with_retry
from .schema_utils import to_xai_schema

API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.6"


def _api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise ProviderError("xai", "XAI_API_KEY is not set (see .env.example)")
    return key


@with_retry
def call_structured(
    schema_model: type[BaseModel],
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8000,
    timeout: float = 180.0,
) -> ProviderResult:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": to_xai_schema(schema_model),
    }
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(API_URL, headers=headers, json=payload)
    raise_for_status("xai", response)
    body = response.json()

    choices = body.get("choices") or []
    if not choices:
        raise ProviderError("xai", f"empty choices: {str(body)[:400]}")
    content = choices[0].get("message", {}).get("content") or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "xai", f"schema-constrained reply was not JSON: {content[:400]}"
        ) from exc
    if not isinstance(data, dict):
        raise ProviderError("xai", f"expected a JSON object, got {type(data).__name__}")

    usage = body.get("usage", {}) or {}
    return ProviderResult(
        data=data,
        model=model,
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        raw=body,
    )
