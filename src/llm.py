"""Thin provider-agnostic wrapper around the optional cloud LLM call.

Supports LLM_PROVIDER=anthropic (Claude, via the anthropic SDK), or
LLM_PROVIDER=deepseek / gemini (both OpenAI-compatible chat-completions
APIs, called directly over HTTP - no extra SDK dependency needed). Used
for exactly two things: cluster naming (label.py) and the Ask AI Q&A layer
(rag_qa.py).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

from .config import settings

_anthropic_client = None

# base_url -> unix timestamp until which we skip calling that provider,
# set after a 429 so a batch job (e.g. labeling hundreds of clusters) fails
# fast on the network instead of firing - and waiting on - a doomed request
# per remaining item once the per-minute quota is used up.
_cooldown_until: dict[str, float] = {}


class LLMUnavailable(RuntimeError):
    """Raised when no LLM provider is configured/keyed."""


class RateLimited(LLMUnavailable):
    """Raised when the provider is rate-limited (429), including during our
    own cooldown skip - both mean "don't call the network right now"."""


def is_configured() -> bool:
    if settings.llm_provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if settings.llm_provider == "deepseek":
        return bool(settings.deepseek_api_key)
    if settings.llm_provider == "gemini":
        return bool(settings.gemini_api_key)
    return False


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _openai_compatible_chat(
    base_url: str, api_key: str, model: str, system: str | None, messages: list[dict], max_tokens: int
) -> str:
    now = time.time()
    cooldown = _cooldown_until.get(base_url, 0.0)
    if now < cooldown:
        raise RateLimited(f"Rate-limited - retrying in {cooldown - now:.0f}s")

    payload_messages = ([{"role": "system", "content": system}] if system else []) + messages
    body = json.dumps(
        {
            "model": model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = _parse_retry_delay(e) or 60.0
            _cooldown_until[base_url] = time.time() + retry_after
            raise RateLimited(f"Rate-limited - cooling down {retry_after:.0f}s") from e
        raise
    return data["choices"][0]["message"]["content"].strip()


def _parse_retry_delay(error: "urllib.error.HTTPError") -> float | None:
    """Best-effort read of the provider's suggested retry delay (Gemini's 429
    body includes e.g. `"retryDelay": "58s"`) so the cooldown roughly matches
    its own per-minute quota window instead of a guessed constant."""
    try:
        match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', error.read().decode("utf-8"))
        return float(match.group(1)) if match else None
    except Exception:
        return None


def chat(messages: list[dict], max_tokens: int, system: str | None = None) -> str:
    """One-shot chat completion using whichever provider is configured."""
    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        resp = _get_anthropic_client().messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            system=system or "",
            messages=messages,
        )
        return resp.content[0].text.strip()

    if settings.llm_provider == "deepseek" and settings.deepseek_api_key:
        return _openai_compatible_chat(
            settings.deepseek_base_url, settings.deepseek_api_key, settings.deepseek_model,
            system, messages, max_tokens,
        )

    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return _openai_compatible_chat(
            settings.gemini_base_url, settings.gemini_api_key, settings.gemini_model,
            system, messages, max_tokens,
        )

    raise LLMUnavailable(
        "No cloud LLM configured. Set LLM_PROVIDER to 'anthropic', 'deepseek', or 'gemini' "
        "(plus the matching API key) in your .env, then restart the app."
    )
