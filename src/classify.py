"""Optional LLM-based category/domain cleanup, run once per ticket before
embedding.

Raw ticket categories are often inconsistent free-text labels from the
source system (e.g. "Easychit" vs "EasyChit Client" for the same product -
see src/cluster.py's clustering docstring), which fragments the dashboard's
category breakdown even though clustering itself never looks at category.
This asks an LLM to fold each ticket's raw category (and, separately,
domain) into one of the values already seen in the dataset - or propose a
short new one only if genuinely nothing fits - so the breakdown reflects
real product/issue groupings instead of raw label variance, and, since
preprocess.py includes both in the embedding text, this also sharpens
clustering itself.

Domain matters just as much as category here: manually-submitted tickets
pick a domain from a generic IT-type dropdown (Hardware/Software/App
Support), while ~97% of the real historical dataset uses domain="product"
- a near-constant field with essentially no real information in it. Left
unnormalized, that mismatch alone was measured to drag a genuine duplicate
ticket's similarity from 0.80 down to 0.63, below the clustering threshold
- worse than the category mismatch, since domain is even less variable in
the real data. Classifying domain the same way naturally folds a manual
ticket's domain toward whatever the real data actually uses (usually
"product"), removing that mismatch instead of embedding it as noise.

Opt-in (AI_CLASSIFICATION=true) since it's a paid call per ticket per
field, same tradeoff as EMBEDDING_ELABORATION.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import llm

MAX_KNOWN_VALUES = 40

_SYSTEM_PROMPTS = {
    "category": (
        "You clean up support-ticket categories. Given a ticket and a list of categories already "
        "used in this dataset, reply with the ONE category from that list that best fits this "
        "ticket. Only if truly none of them fit, reply with a new short category name (2-4 words) "
        "instead. Reply with ONLY the category name - no preamble, no quotes, no punctuation."
    ),
    "domain": (
        "You clean up support-ticket domains (the broad area a ticket belongs to). Given a ticket "
        "and a list of domains already used in this dataset, reply with the ONE domain from that "
        "list that best fits this ticket. Only if truly none of them fit, reply with a new short "
        "domain name (1-3 words) instead. Reply with ONLY the domain name - no preamble, no "
        "quotes, no punctuation."
    ),
}


def _known_values(tickets: list[dict], field: str, limit: int = MAX_KNOWN_VALUES) -> list[str]:
    counts = Counter((t.get(field) or "").strip() for t in tickets if (t.get(field) or "").strip())
    return [name for name, _ in counts.most_common(limit)]


def _classify_one(ticket: dict, known: list[str], fallback: str, field: str) -> str:
    subject = (ticket.get("subject") or "").strip()
    description = (ticket.get("description") or "").strip()
    if not subject and not description:
        return fallback

    label = "categories" if field == "category" else "domains"
    prompt = (
        f"Known {label} in this dataset: {', '.join(known) if known else '(none yet)'}\n\n"
        f"Ticket domain: {ticket.get('domain') or 'unknown'}\n"
        f"Ticket's raw category: {ticket.get('category') or 'unknown'}\n"
        f"Subject: {subject}\n"
        f"Description: {description[:1000]}"
    )
    try:
        text = llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=20,
            system=_SYSTEM_PROMPTS[field],
            task="classify",
        ).strip().strip('"').rstrip(".")
        # A weak/free model can ignore the "ONLY the name" instruction and
        # dump its reasoning as the answer instead - a short check here
        # beats writing a paragraph into ticket["category"]/["domain"].
        if not text or len(text) > 60 or "\n" in text:
            return fallback
        return text
    except Exception:
        return fallback


def _classify_field(
    tickets: list[dict],
    field: str,
    verbose: bool,
    max_workers: int,
    reference_tickets: list[dict] | None,
) -> list[str]:
    fallback_value = "Uncategorized" if field == "category" else "product"
    if not llm.is_configured(task="classify"):
        return [(t.get(field) or fallback_value).strip() or fallback_value for t in tickets]
    if not tickets:
        return []

    known = _known_values(reference_tickets if reference_tickets else tickets, field)
    fallbacks = [(t.get(field) or fallback_value).strip() or fallback_value for t in tickets]

    results = list(fallbacks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_classify_one, ticket, known, fallbacks[i], field): i
            for i, ticket in enumerate(tickets)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if verbose and (done % 100 == 0 or done == len(tickets)):
                print(f"      classified {field} for {done}/{len(tickets)} tickets...")

    return results


def classify_texts(
    tickets: list[dict],
    verbose: bool = True,
    max_workers: int = 8,
    reference_tickets: list[dict] | None = None,
) -> list[str]:
    """Returns one cleaned category per ticket, same order as `tickets`.
    Falls back to the ticket's own raw category (or "Uncategorized") per
    ticket on any API failure. Known categories are seeded from `tickets`
    itself, unless `reference_tickets` is given (e.g. a single
    just-submitted ticket has no batch of its own to seed from, so the
    manual-submission path passes the existing dataset instead) - either
    way this keeps results grounded in categories that actually occur,
    instead of the model inventing an unbounded taxonomy."""
    return _classify_field(tickets, "category", verbose, max_workers, reference_tickets)


def classify_domains(
    tickets: list[dict],
    verbose: bool = True,
    max_workers: int = 8,
    reference_tickets: list[dict] | None = None,
) -> list[str]:
    """Same idea as classify_texts, but for `domain` - see the module
    docstring for why domain needs this even more than category does."""
    return _classify_field(tickets, "domain", verbose, max_workers, reference_tickets)
