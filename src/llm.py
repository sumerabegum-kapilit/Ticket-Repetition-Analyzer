"""Thin provider-agnostic wrapper around the optional cloud LLM call.

Supports LLM_PROVIDER=anthropic (Claude, via the anthropic SDK), or
LLM_PROVIDER=deepseek / gemini / openrouter (all OpenAI-compatible
chat-completions APIs, called directly over HTTP - no extra SDK dependency
needed). Used for the LLM touchpoints in this project: cluster naming
(label.py), category cleanup (classify.py), pre-embedding elaboration
(elaborate.py), and the Ask AI Q&A layer (rag_qa.py).
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request

from .config import settings

_anthropic_client = None

# (base_url, model) -> unix timestamp until which we skip calling that
# model, set after a 429 so a batch job (e.g. labeling hundreds of
# clusters) fails fast on the network instead of firing - and waiting on -
# a doomed request per remaining item once the per-minute quota is used up.
# Keyed by model too (not just base_url) so OpenRouter's pool of many
# models sharing one base_url each get their own independent cooldown.
_cooldown_until: dict[tuple[str, str], float] = {}

# OpenRouter free-tier chat models, rotated in random order per request
# with failover to the next model on rate-limit/unavailable (see
# _openrouter_chat below). Override with OPENROUTER_MODELS (comma-
# separated) - the free tier changes over time, so refresh this list
# periodically; an unknown/unavailable slug is just skipped, not fatal.
DEFAULT_OPENROUTER_FREE_MODELS = [
    'nvidia/nemotron-3-ultra-550b-a55b:free',
    'nvidia/nemotron-3-super-120b-a12b:free',
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
    'google/gemma-4-31b-it:free',
    'google/gemma-4-26b-a4b-it:free',
    'deepseek/deepseek-v4-flash-0731:free',
    'z-ai/glm-5.2:free',
    'qwen/qwen3.8-27b:free',
    'liquid/lfm-2.5-2.6b:free',
    'cohere/north-mini-code:free',
]


class LLMUnavailable(RuntimeError):
    """Raised when no LLM provider is configured/keyed."""


class RateLimited(LLMUnavailable):
    """Raised when the provider is rate-limited (429), including during our
    own cooldown skip - both mean "don't call the network right now"."""


def _provider_for(task: str) -> str:
    """Resolve which provider a task should use: its own
    LLM_PROVIDER_<TASK> override if set, else the global LLM_PROVIDER.
    `task` is one of "label", "rag", "classify", "elaborate", or "" for the
    plain global default."""
    override = getattr(settings, f"llm_provider_{task}", "") if task else ""
    return override or settings.llm_provider


def _key_for(provider: str) -> str:
    return {
        "anthropic": settings.anthropic_api_key,
        "deepseek": settings.deepseek_api_key,
        "gemini": settings.gemini_api_key,
        "openrouter": settings.openrouter_api_key,
    }.get(provider, "")


def _openrouter_model_pool() -> list[str]:
    configured = [m.strip() for m in settings.openrouter_models_raw.split(",") if m.strip()]
    return configured or DEFAULT_OPENROUTER_FREE_MODELS


def is_configured(task: str = "") -> bool:
    provider = _provider_for(task)
    return bool(provider) and bool(_key_for(provider))


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    system: str | None,
    messages: list[dict],
    max_tokens: int,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 60,
) -> str:
    cooldown_key = (base_url, model)
    now = time.time()
    cooldown = _cooldown_until.get(cooldown_key, 0.0)
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
            **(extra_headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = _parse_retry_delay(e) or 60.0
            _cooldown_until[cooldown_key] = time.time() + retry_after
            raise RateLimited(f"Rate-limited - cooling down {retry_after:.0f}s") from e
        raise

    if isinstance(data, dict) and "error" in data:
        # OpenRouter (proxying many upstream providers) reports some
        # failures - most commonly a free model being temporarily
        # overloaded - as HTTP 200 with an {"error": ...} body instead of a
        # real HTTP error status. Without this, that would crash below on
        # the missing "choices" key instead of letting pool failover
        # (_openrouter_chat) move on to the next model.
        err = data["error"]
        message = err.get("message") if isinstance(err, dict) else str(err)
        _cooldown_until[cooldown_key] = time.time() + 30.0
        raise RateLimited(f"Upstream error: {message}")

    content = data["choices"][0]["message"].get("content")
    if not content:
        # Some models (reasoning models especially) can come back with a
        # null/empty content even on a clean "stop"/"length" response - e.g.
        # cut off mid "reasoning" before ever writing to `content`. Treat it
        # as retryable so pool failover moves on instead of crashing.
        raise RateLimited(f"Empty response content from model '{model}'")
    return content.strip()


def _parse_retry_delay(error: "urllib.error.HTTPError") -> float | None:
    """Best-effort read of the provider's suggested retry delay (Gemini's 429
    body includes e.g. `"retryDelay": "58s"`) so the cooldown roughly matches
    its own per-minute quota window instead of a guessed constant."""
    try:
        match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', error.read().decode("utf-8"))
        return float(match.group(1)) if match else None
    except Exception:
        return None


def _openrouter_chat(system: str | None, messages: list[dict], max_tokens: int) -> str:
    """Tries OpenRouter's model pool in random order, failing over to the
    next model on rate-limit (429) or an unrecognized/unavailable model id
    (400/404 - the free tier's lineup changes over time) instead of failing
    the whole request. A pinned OPENROUTER_MODEL always goes first. Capped
    at OPENROUTER_MAX_RETRIES+1 attempts so a bad run can't try every model
    in a large pool before giving up."""
    pool = _openrouter_model_pool()
    order = list(pool)
    random.shuffle(order)
    if settings.openrouter_model:
        order = [settings.openrouter_model] + [m for m in order if m != settings.openrouter_model]
    order = order[: max(1, settings.openrouter_max_retries + 1)]

    # Several free models in the pool (e.g. Nvidia's Nemotron line) spend
    # part of their token budget "thinking" in a separate `reasoning` field
    # before writing the visible answer to `content` - a caller-requested
    # max_tokens tuned for a short label/category (20-30) can get spent
    # entirely on that reasoning, leaving `content` null. These models are
    # all free (cost: 0), so there's no downside to asking for more room
    # than the caller strictly needs.
    effective_max_tokens = max(max_tokens, 300)

    headers = {"HTTP-Referer": settings.openrouter_site_url, "X-Title": settings.openrouter_app_name}
    last_error: Exception | None = None
    for model in order:
        try:
            return _openai_compatible_chat(
                settings.openrouter_base_url,
                settings.openrouter_api_key,
                model,
                system,
                messages,
                effective_max_tokens,
                extra_headers=headers,
                timeout=settings.openrouter_timeout_seconds,
            )
        except RateLimited as exc:
            last_error = exc
            continue
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                last_error = exc
                continue
            raise

    raise LLMUnavailable(
        f"All {len(order)} attempted OpenRouter model(s) failed (rate-limited or unavailable)."
    ) from last_error


def chat(messages: list[dict], max_tokens: int, system: str | None = None, task: str = "") -> str:
    """One-shot chat completion using whichever provider is configured for
    `task` (see _provider_for) - "label", "rag", "classify", "elaborate",
    or "" for the plain LLM_PROVIDER default."""
    provider = _provider_for(task)

    if provider == "anthropic" and settings.anthropic_api_key:
        resp = _get_anthropic_client().messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            system=system or "",
            messages=messages,
        )
        return resp.content[0].text.strip()

    if provider == "deepseek" and settings.deepseek_api_key:
        return _openai_compatible_chat(
            settings.deepseek_base_url, settings.deepseek_api_key, settings.deepseek_model,
            system, messages, max_tokens,
        )

    if provider == "gemini" and settings.gemini_api_key:
        return _openai_compatible_chat(
            settings.gemini_base_url, settings.gemini_api_key, settings.gemini_model,
            system, messages, max_tokens,
        )

    if provider == "openrouter" and settings.openrouter_api_key:
        return _openrouter_chat(system, messages, max_tokens)

    raise LLMUnavailable(
        "No cloud LLM configured. Set LLM_PROVIDER (or the task-specific "
        f"LLM_PROVIDER_{task.upper()}) to 'anthropic', 'deepseek', 'gemini', or 'openrouter' "
        "(plus the matching API key) in your .env, then restart the app."
        if task
        else "No cloud LLM configured. Set LLM_PROVIDER to 'anthropic', 'deepseek', 'gemini', or "
        "'openrouter' (plus the matching API key) in your .env, then restart the app."
    )
