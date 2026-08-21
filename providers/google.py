"""Google Gemini -- the Critic.

Neither Anthropic's forced tools nor OpenAI's ``json_schema`` block exist here.
Gemini wants two fields set together: ``responseMimeType`` declaring JSON, and
``responseSchema`` in its own upper-cased dialect (see
``schema_utils.to_gemini_schema``). Setting the MIME type alone yields JSON of
whatever shape the model felt like.

Cheapest of the three models by an order of magnitude, which is what makes a
grading pass on every round affordable.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from . import ProviderResult
from .retry_utils import ProviderError, raise_for_status, with_retry
from .schema_utils import to_gemini_schema

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-flash-lite"


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
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": to_gemini_schema(schema_model),
            "maxOutputTokens": max_tokens,
        },
    }
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
    url = f"{API_BASE}/{model}:generateContent"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
    raise_for_status("google", response, api_key)
    body = response.json()

    candidates = body.get("candidates") or []
    if not candidates:
        raise ProviderError("google", f"no candidates: {str(body)[:400]}")
    parts = candidates[0].get("content", {}).get("parts", []) or []
    text = "".join(part.get("text", "") for part in parts).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "google", f"schema-constrained reply was not JSON: {text[:400]}"
        ) from exc
    if not isinstance(data, dict):
        raise ProviderError("google", f"expected a JSON object, got {type(data).__name__}")

    usage = body.get("usageMetadata", {}) or {}
    return ProviderResult(
        data=data,
        model=model,
        input_tokens=int(usage.get("promptTokenCount", 0)),
        output_tokens=int(usage.get("candidatesTokenCount", 0)),
        raw=body,
    )
