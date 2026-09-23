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
    ("issue", ["issue", "problem", "trouble"]),
    # Generic names for "the whole device" - deliberately does NOT include
    # a specific component/peripheral (keyboard, mouse, printer, monitor).
    # Folding a component into this group would make e.g. "keyboard broken"
    # indistinguishable from "system won't boot", which is a real, different
    # complaint - the goal here is only to stop synonymous whole-device
    # wording (a laptop IS a computer IS a machine) from fragmenting.
    ("system", ["system", "machine", "laptop", "computer", "desktop", "pc"]),
    # Domain/category context, not subject/description wording: ~97% of the
    # real historical dataset uses domain="product" (a near-constant,
    # generic value), while the manual ticket form's "App Support" option is
    # the closest equivalent generic choice there - treating them as the
    # same word lets a manually-submitted App Support ticket's context match
    # the real data's dominant domain instead of being a permanent mismatch
    # against it (measured: this exact mismatch alone can drag a genuine
    # duplicate's similarity from ~0.80 down to ~0.63).
    ("product", ["product", "app support"]),
]
_SYNONYM_PATTERNS = [
    (re.compile(r"(?i)\b" + re.escape(phrase) + r"\b"), canonical)
    for canonical, phrases in _SYNONYM_GROUPS
    for phrase in sorted(phrases, key=len, reverse=True)
]

# Branch/location-code prefixes this source system stamps onto subject and
# description (e.g. "MVO KARIMNAGAR - SYSTEM PROBLEM .", "SATHUPALLY - CLINT
# SYSTEM PROBLEM"). Measured directly: two tickets about the identical issue
# ("System issue" vs "SYSTEM PROBLEM .", one with a "MVO KARIMNAGAR - "
# prefix) scored 0.665 cosine similarity - well under the clustering
# threshold - and jumped to 0.849 with just that prefix removed. Requires
# ALL-CAPS words (a branch code, not a real sentence) and a real space after
# the hyphen (so inline reference codes like "KHAT13Z-18" or "KKPB02J-14",
# which have no space around their hyphen, are never touched), and at least
# 3 characters in the first word (so short real tokens like "PR -" or "A -"
# aren't mistaken for a branch code).
_BRANCH_CODE_PREFIX = re.compile(r"^[A-Z][A-Z0-9&/.]{2,}(?:\s+[A-Z0-9&/.]+){0,3}\s*-\s+")


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


def _strip_branch_code_prefix(text: str) -> str:
    return _BRANCH_CODE_PREFIX.sub("", text)


def build_embedding_text(ticket: dict) -> str:
    """Subject carries most of the "is this the same issue" signal, so it's
    weighted by repetition rather than a hand-tuned vector-concat weight.
    Synonym normalization runs only on the embedding text - the
    subject/description shown in the UI stay as originally written.

    category/domain are prepended as real, already-known context (not
    invented) - on a short subject line like "Phone number change" they give
    the embedding model something concrete to disambiguate on, at no extra
    cost, unlike a generic subject that could belong to several categories."""
    subject = _normalize_synonyms(_strip_branch_code_prefix(clean_text(ticket.get("subject", ""))))
    description = _normalize_synonyms(_strip_branch_code_prefix(clean_text(ticket.get("description", ""))))
    category = _normalize_synonyms(clean_text(ticket.get("category") or ""))
    domain = _normalize_synonyms(clean_text(ticket.get("domain") or ""))
    context = " ".join(part for part in (domain, category) if part)
    prefix = f"{context}: " if context else ""
    return f"{prefix}{subject}. {subject}. {description}"[:2000]
