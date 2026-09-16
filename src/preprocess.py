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


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _QUOTE_BLOCK.sub("", text)
    text = _SIGNATURE.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def build_embedding_text(ticket: dict) -> str:
    """Subject carries most of the "is this the same issue" signal, so it's
    weighted by repetition rather than a hand-tuned vector-concat weight."""
    subject = clean_text(ticket.get("subject", ""))
    description = clean_text(ticket.get("description", ""))
    return f"{subject}. {subject}. {description}"[:2000]
