"""Optional LLM-based text elaboration, run once per ticket before embedding.

A short subject like "Phone number change" gives the embedding model very
little to reason about, so two real duplicates worded differently can still
land far apart in vector space. This asks Claude to restate what the ticket
is actually reporting, in a fuller sentence, using only what's in the
ticket - giving the embedding model a longer, more explicit description to
compare instead of a terse phrase. Opt-in (EMBEDDING_ELABORATION=true) since
it's a paid call per ticket, unlike the other LLM touchpoints in this
project (one call per cluster label, one call per Q&A question).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import settings

_client = None

SYSTEM_PROMPT = (
    "You restate a support ticket as one plain, self-contained sentence describing what "
    "the reporter is asking for or experiencing. Use only information given below - never "
    "invent names, numbers, products, or details that aren't there. If the ticket already "
    "reads as one clear sentence, you may return it close to as-is. Reply with ONLY the "
    "sentence, no preamble, no quotes."
)


def _get_anthropic_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _elaborate_one(ticket: dict, fallback_text: str) -> str:
    subject = (ticket.get("subject") or "").strip()
    description = (ticket.get("description") or "").strip()
    if not subject and not description:
        return fallback_text

    prompt = (
        f"Category: {ticket.get('category') or 'unknown'}\n"
        f"Domain: {ticket.get('domain') or 'unknown'}\n"
        f"Subject: {subject}\n"
        f"Description: {description[:1500]}"
    )
    try:
        resp = _get_anthropic_client().messages.create(
            model=settings.anthropic_model,
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        return text or fallback_text
    except Exception:
        return fallback_text


def elaborate_texts(
    tickets: list[dict], base_texts: list[str], verbose: bool = True, max_workers: int = 8
) -> list[str]:
    """Returns one elaborated string per ticket, same order as `tickets`/
    `base_texts`. Falls back to `base_texts[i]` (the already
    cleaned+synonym-normalized text) per-ticket on any API failure, so one
    bad call never drops a ticket from the run. Runs concurrently since
    these are network-bound calls and a real ticket backlog run
    sequentially would take far too long."""
    if settings.llm_provider != "anthropic" or not settings.anthropic_api_key:
        return base_texts
    if not tickets:
        return base_texts

    results = list(base_texts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_elaborate_one, ticket, base_texts[i]): i for i, ticket in enumerate(tickets)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if verbose and (done % 100 == 0 or done == len(tickets)):
                print(f"      elaborated {done}/{len(tickets)} tickets...")

    return results
