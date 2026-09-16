"""Give each cluster a human-readable issue name.

Default (free, offline): use the representative ticket's own subject line.
Optional (LLM_PROVIDER=anthropic): one cheap LLM call per cluster - not per
ticket - asks Claude to turn a handful of sample subjects into a single
clean, canonical issue name. This is the "understands these are the same
issue" layer riding on top of embedding-based grouping, used only to make
the label readable, not to do the (expensive, slow) matching itself.
"""
from __future__ import annotations

from .config import settings

_client = None


def _get_anthropic_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _extractive_label(subjects: list[str]) -> str:
    rep = min(subjects, key=len)
    return rep.strip().rstrip(".").capitalize()


def label_cluster(subjects: list[str]) -> str:
    fallback = _extractive_label(subjects)
    if settings.llm_provider != "anthropic" or not settings.anthropic_api_key:
        return fallback

    try:
        sample = subjects[:6]
        prompt = (
            "These support-ticket subject lines all describe the same underlying issue, "
            "just worded differently by different reporters:\n\n"
            + "\n".join(f"- {s}" for s in sample)
            + "\n\nReply with ONLY a single short (max 8 words) canonical issue name that "
            "captures what they all have in common. No punctuation at the end, no preamble."
        )
        resp = _get_anthropic_client().messages.create(
            model=settings.anthropic_model,
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip().strip('"')
        return text if text else fallback
    except Exception:
        return fallback
