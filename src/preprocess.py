"""Clean ticket text and build the string that gets embedded."""
from __future__ import annotations

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_QUOTE_BLOCK = re.compile(
    r"(?is)(On .{0,80} wrote:.*$)|(From:\s*.*$)|(-{2,}\s*Original Message\s*-{2,}.*$)"
)
_SIGNATURE = re.compile(
    r"(?is)\b(regards|thanks|thank you|best regards|sincerely|warm regards)[,.]?\s*$"
)

# Same-field wording that embedding models don't reliably treat as
# synonymous on short ticket text (e.g. "mobile no change" vs "cell number
# change" scored ~0.46 cosine similarity with the default model - well under
# the clustering threshold - even though they're the same request). Folding
# each group to one canonical phrase before embedding (display text is left
# untouched) removes that wording gap instead of chasing it by loosening the
# similarity threshold, which would risk merging unrelated tickets globally.
# Longest phrase wins within a group, and groups are independent of each
# other, so add new groups/phrases here as new near-duplicate wording shows
# up in real tickets.
_SYNONYM_GROUPS: list[tuple[str, list[str]]] = [
    ("phone number", [
        "phone number", "phone no", "telephone number", "telephone no",
        "mobile number", "mobile no", "cell number", "cell no",
        "contact number", "contact no",
    ]),
    ("email address", ["email address", "email id", "e mail", "mail id"]),
    ("username", ["username", "user id", "login id", "user name"]),
    ("password", ["password", "pwd"]),
]
_SYNONYM_PATTERNS = [
    (re.compile(r"(?i)\b" + re.escape(phrase) + r"\b"), canonical)
    for canonical, phrases in _SYNONYM_GROUPS
    for phrase in sorted(phrases, key=len, reverse=True)
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _QUOTE_BLOCK.sub("", text)
    text = _SIGNATURE.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def _normalize_synonyms(text: str) -> str:
    for pattern, canonical in _SYNONYM_PATTERNS:
        text = pattern.sub(canonical, text)
    return text


def build_embedding_text(ticket: dict) -> str:
    """Subject carries most of the "is this the same issue" signal, so it's
    weighted by repetition rather than a hand-tuned vector-concat weight.
    Synonym normalization runs only on the embedding text - the
    subject/description shown in the UI stay as originally written.

    category/domain are prepended as real, already-known context (not
    invented) - on a short subject line like "Phone number change" they give
    the embedding model something concrete to disambiguate on, at no extra
    cost, unlike a generic subject that could belong to several categories."""
    subject = _normalize_synonyms(clean_text(ticket.get("subject", "")))
    description = _normalize_synonyms(clean_text(ticket.get("description", "")))
    category = clean_text(ticket.get("category") or "")
    domain = clean_text(ticket.get("domain") or "")
    context = " ".join(part for part in (domain, category) if part)
    prefix = f"{context}: " if context else ""
    return f"{prefix}{subject}. {subject}. {description}"[:2000]
