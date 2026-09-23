"""Phase 4: RAG Q&A layer - natural-language questions over the ticket/cluster data.

- Retrieval: embed the question with the same local embedding model used for
  tickets, and pull the most similar ticket vectors from the persisted vector
  store (data/vector_store/). No extra index to maintain - it's the same
  store the clustering pipeline already built.
- Augmentation: pair those individual tickets with the dataset-wide stats the
  pipeline already computed (data/clusters.json) - KPIs, top issues,
  breakdowns, trend - since most "how many / what's trending" questions are
  answered better by the precomputed aggregates than by an LLM eyeballing a
  sample of retrieved tickets.
- Generation: one cloud LLM call (Claude, or DeepSeek) composes an answer
  grounded only in that context, per PROJECT_PLAN.md's hybrid design (local
  embeddings for the bulk clustering work, cloud LLM only for this
  on-demand Q&A).
"""
from __future__ import annotations

import json

import numpy as np

from . import llm
from .config import settings
from .embed import embed_texts
from .vector_store import VectorStore

TOP_K = 12
MAX_HISTORY_TURNS = 6


class RagUnavailable(RuntimeError):
    """Raised when the Q&A layer can't run yet (no LLM configured, or no analyzed data)."""


def _require_llm() -> None:
    if not llm.is_configured(task="rag"):
        raise RagUnavailable(
            "The Ask AI layer needs a cloud LLM. Set LLM_PROVIDER (or LLM_PROVIDER_RAG) to "
            "'anthropic', 'deepseek', 'gemini', or 'openrouter' (plus the matching API key) "
            "in your .env, then restart the app."
        )


def _load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    store = VectorStore(settings.vector_store_dir)
    if store.vectors is None or not store.ids:
        return []

    tickets = _load_json(settings.tickets_path) or []
    tickets_by_no = {t["ticket_no"]: t for t in tickets}
    if not tickets_by_no:
        return []

    q_vec = embed_texts([question])[0]
    sims = store.vectors @ q_vec
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for i in top_idx:
        t = tickets_by_no.get(store.ids[int(i)])
        if t:
            results.append({**t, "similarity": round(float(sims[i]), 3)})
    return results


def _most_recent_tickets(limit: int = 10) -> list[dict]:
    """Semantic retrieval below matches by *meaning*, not by date - it has
    no way to answer "what's the newest/most recent ticket" correctly
    (a question's wording rarely resembles a ticket that just happens to be
    recent). This gives the LLM an explicit, correctly-sorted answer to
    that whole class of question instead."""
    tickets = _load_json(settings.tickets_path) or []
    dated = [t for t in tickets if t.get("created_at")]
    # created_at_ts (full timestamp, when available - e.g. manually
    # submitted tickets) breaks ties between same-day tickets correctly;
    # falls back to the date-only created_at for tickets that only ever had
    # day precision (e.g. bulk-imported ones from the source system).
    return sorted(dated, key=lambda t: t.get("created_at_ts") or t["created_at"], reverse=True)[:limit]


def _build_context(question: str) -> tuple[str, list[dict]]:
    clusters_data = _load_json(settings.clusters_path)
    if not clusters_data:
        return "", []

    retrieved = _retrieve(question)

    parts = [
        "DATASET-WIDE STATISTICS (already computed over the full dataset - "
        "trust these for any total/count/percentage/trend question):\n"
        + json.dumps(
            {
                "kpis": clusters_data.get("kpis"),
                "top_issues": clusters_data.get("top_issues"),
                "breakdowns": clusters_data.get("breakdowns"),
                "overall_trend": clusters_data.get("overall_trend"),
            },
            indent=2,
        )
    ]

    recent = _most_recent_tickets()
    if recent:
        parts.append(
            "MOST RECENTLY CREATED TICKETS OVERALL, sorted newest first (use this - not the "
            "semantic-search sample below - for any \"newest/latest/most recent ticket\" "
            "question, regardless of topic):\n"
            + "\n".join(
                f'- {t["ticket_no"]} | created={t.get("created_at_ts") or t.get("created_at")} | '
                f'subject: {t["subject"]} | category={t.get("category")} | '
                f'{"part of a repeating issue" if t.get("cluster_size", 1) > 1 else "new / not yet repeated"}'
                for t in recent
            )
        )

    new_tickets = clusters_data.get("new_tickets") or []
    if new_tickets:
        parts.append(
            'TICKETS THAT HAVE NOT REPEATED YET ("new issues" in this app\'s own terms - use '
            "this for any \"what's a new ticket / what hasn't repeated\" question), newest "
            "first, capped sample:\n"
            + "\n".join(
                f'- {t["ticket_no"]} | created={t.get("created_at_ts") or t.get("created_at")} | '
                f'subject: {t["subject"]} | category={t.get("category")}'
                for t in new_tickets[:10]
            )
        )

    if retrieved:
        lines = []
        for t in retrieved:
            issue = t.get("cluster_issue") or "unique - not part of a repeating issue"
            lines.append(
                f'- ticket {t["ticket_no"]} | category={t["category"]} | department={t["department"]} '
                f'| priority={t["priority"]} | status={t["status"]} | created={t.get("created_at")}\n'
                f'  recurring issue: "{issue}" (seen {t.get("cluster_size", 1)}x total)\n'
                f'  subject: {t["subject"]}\n'
                f'  description: {(t.get("description") or "")[:300]}'
            )
        parts.append(
            "INDIVIDUAL TICKETS MOST RELEVANT TO THIS QUESTION (semantic search over all "
            "tickets, for examples/detail - this is a sample, not the full dataset):\n"
            + "\n".join(lines)
        )

    return "\n\n".join(parts), retrieved


SYSTEM_PROMPT = (
    "You are the analytics assistant built into the Ticket Repetition Analyzer app. "
    "Answer the user's question using ONLY the CONTEXT given with it - never invent tickets, "
    "counts, or issues that aren't there. "
    "The CONTEXT can have up to four parts: (1) precomputed dataset-wide statistics - use "
    "these for any total/count/percentage/trend/top-N question, they cover the whole "
    "dataset; (2) most recently created tickets overall, already sorted newest-first - use "
    "this, not part (4), for any \"newest/latest/most recent ticket\" question, since part "
    "(4) is matched by meaning and is NOT sorted by date; (3) tickets that haven't repeated "
    "yet (\"new issues\") - use this for \"what's a new/non-repeating ticket\" questions; "
    "(4) a sample of individual tickets semantically relevant to the question's wording - "
    "use these for examples/detail on the question's topic, never treat this sample's size "
    "as the dataset total, and never use it to answer a recency question. "
    "Cite ticket numbers in parentheses when referencing a specific ticket. "
    "Keep answers concise. If the context doesn't have enough information, say so plainly "
    "instead of guessing."
)


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    _require_llm()

    context, retrieved = _build_context(question)
    if not context:
        raise RagUnavailable(
            "No analyzed data found yet - upload a file or run the pipeline first."
        )

    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in (history or [])[-MAX_HISTORY_TURNS:]
        if turn.get("role") in ("user", "assistant") and turn.get("content")
    ]
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})

    answer = llm.chat(messages, max_tokens=700, system=SYSTEM_PROMPT, task="rag")
    sources = [
        {"ticket_no": t["ticket_no"], "subject": t["subject"], "category": t["category"]}
        for t in retrieved[:8]
    ]
    return {"answer": answer, "sources": sources}
