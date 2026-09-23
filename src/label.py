"""Give each cluster a human-readable issue name.

Default (free, offline): use the representative ticket's own subject line.
Optional (LLM_PROVIDER=anthropic or deepseek): one cheap LLM call per
cluster - not per ticket - asks the model to turn a handful of sample
subjects into a single clean, canonical issue name. This is the
"understands these are the same issue" layer riding on top of
embedding-based grouping, used only to make the label readable, not to do
the (expensive, slow) matching itself.
"""
from __future__ import annotations

from . import llm


def _extractive_label(subjects: list[str]) -> str:
    rep = min(subjects, key=len)
    return rep.strip().rstrip(".").capitalize()


def label_cluster(subjects: list[str]) -> str:
    fallback = _extractive_label(subjects)
    if not llm.is_configured(task="label"):
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
        text = llm.chat([{"role": "user", "content": prompt}], max_tokens=30, task="label").strip('"')
        # A weak/free model can ignore the "ONLY a short name" instruction
        # and dump its reasoning as the answer instead - a short check here
        # beats a paragraph showing up as a cluster's display name.
        if not text or len(text) > 80 or "\n" in text:
            return fallback
        return text
    except Exception:
        return fallback
