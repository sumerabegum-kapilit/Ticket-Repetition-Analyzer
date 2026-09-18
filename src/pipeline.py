"""End-to-end: extract -> preprocess -> embed -> cluster -> label -> metrics.

Writes data/clusters.json, which generate_report.py turns into the static
HTML dashboard. Kept as one script (not a queue/orchestrator) because the
whole run - even at tens of thousands of tickets - takes seconds to a few
minutes on a laptop; there is no case here yet for anything heavier.

`incremental=True` (phase 5) skips re-reading/re-embedding tickets already
in data/tickets.json + the vector store - it only fetches and embeds
tickets newer than the latest one already processed, then re-clusters and
regenerates the report over the full (old + new) set. Clustering itself
still runs over everything because it's a global operation (a new ticket
can join an existing cluster or merge two), but it's cheap - embedding is
the expensive step this mode actually avoids repeating.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone

from .cluster import cluster_tickets
from .config import settings
from .embed import embed_texts
from .extract import extract_tickets, parse_dt
from .label import label_cluster
from .preprocess import build_embedding_text
from .vector_store import VectorStore

PRIORITY_BUCKETS = {
    "critical": "crit",
    "urgent": "crit",
    "p1": "crit",
    "high": "high",
    "p2": "high",
    "medium": "med",
    "normal": "med",
    "p3": "med",
    "low": "low",
    "p4": "low",
}
PRIORITY_ORDER = {"crit": 0, "high": 1, "med": 2, "low": 3}
OPEN_STATUSES = {"open", "in_progress", "pending"}


def map_priority(raw: str) -> str:
    return PRIORITY_BUCKETS.get((raw or "").lower().strip(), "med")


def _week_bucket(dt: datetime, now: datetime) -> int:
    """0 = the most recent 7-day bucket ending at `now`, 7 = 8 buckets ago."""
    days_ago = (now - dt).days
    return days_ago // 7


def _dimension_breakdown(tickets: list[dict], redundant_ids: set[str], key_fn, top_n: int = 8) -> list[dict]:
    """Ticket volume and repeat rate for one field (category/department/...),
    keeping the top_n biggest slices and folding everything else into "Other"
    so a long-tail field (e.g. 33 departments) still renders as a readable chart."""
    totals = Counter(key_fn(t) for t in tickets)
    redundant = Counter(key_fn(t) for t in tickets if t["ticket_no"] in redundant_ids)
    ranked = totals.most_common()

    def row(name: str, total: int) -> dict:
        red = redundant.get(name, 0)
        return {
            "name": name,
            "total": total,
            "redundant": red,
            "repeat_rate_pct": round(red / total * 100) if total else 0,
        }

    rows = [row(name, total) for name, total in ranked[:top_n]]
    rest = ranked[top_n:]
    if rest:
        # Named distinctly from a literal "Other"/"Others" value that may
        # already exist in the source data (seen in real tickets), so the
        # two are never confused for the same row.
        rest_total = sum(v for _, v in rest)
        rest_redundant = sum(redundant.get(name, 0) for name, _ in rest)
        rows.append(
            {
                "name": f"All other ({len(rest)})",
                "total": rest_total,
                "redundant": rest_redundant,
                "repeat_rate_pct": round(rest_redundant / rest_total * 100) if rest_total else 0,
                "is_tail_bucket": True,
            }
        )
    return rows


def _growth_pct(series: list[int]) -> int:
    """% change between the average of the first half and the second half of
    a weekly series - a simple, explainable stand-in for a trend slope."""
    n = len(series)
    if n < 4:
        return 0
    half = n // 2
    first_avg = sum(series[:half]) / half
    second_avg = sum(series[half:]) / (n - half)
    if first_avg <= 0:
        return 100 if second_avg > 0 else 0
    return round((second_avg - first_avg) / first_avg * 100)


def _recommend_view(
    total_by_week: list[int],
    redundant_by_week: list[int],
    weeks_back: int,
    redundant_total: int,
    crit_high_redundant: int,
    category_breakdown: list[dict],
    total_tickets: int,
) -> tuple[str, str]:
    """Pick which dashboard tab best fits this dataset's shape, with a
    one-line, numbers-backed reason - checked in a fixed priority order
    (a sharp volume swing is the most actionable signal, an even spread
    across the board falls through to the general overview)."""
    # Drop leading weeks with zero tickets before measuring growth - otherwise
    # a lookback window that starts before the data source did (a young
    # system, or a short export) reads as an explosive but meaningless spike.
    first_active = next((i for i, v in enumerate(total_by_week) if v > 0), len(total_by_week))
    growth_pct = _growth_pct(redundant_by_week[first_active:])
    if abs(growth_pct) >= 25 and sum(redundant_by_week) >= 5:
        direction = "up" if growth_pct > 0 else "down"
        return "trends", (
            f"Repeat-ticket volume is trending {direction} {abs(growth_pct)}% over the last "
            f"{weeks_back} weeks — worth a closer look."
        )

    crit_high_share = round(crit_high_redundant / redundant_total * 100) if redundant_total else 0
    if crit_high_share >= 40 and redundant_total >= 5:
        return "resolution", (
            f"{crit_high_share}% of redundant tickets are high or critical priority — "
            "these repeats carry outsized cost."
        )

    sizable = [
        c for c in category_breakdown if not c.get("is_tail_bucket") and c["total"] >= max(10, total_tickets * 0.02)
    ]
    if len(sizable) >= 3:
        by_rate = sorted(sizable, key=lambda c: c["repeat_rate_pct"])
        spread = by_rate[-1]["repeat_rate_pct"] - by_rate[0]["repeat_rate_pct"]
        if spread >= 25:
            hi, lo = by_rate[-1], by_rate[0]
            return "clusters", (
                f'Repeat rates vary sharply by category — "{hi["name"]}" repeats '
                f'{hi["repeat_rate_pct"]}% of the time vs {lo["repeat_rate_pct"]}% for "{lo["name"]}". '
                "Filter the table below by category to see it."
            )

    return "overview", "Repeat rates and priority mix look fairly even across the board — start with the big picture."


def _load_known_tickets(store: VectorStore) -> list[dict]:
    """Reconstruct the full ticket records already embedded, from data/tickets.json
    - lets an incremental run reuse them (and their vectors, already in `store`)
    without re-reading or re-embedding anything that hasn't changed."""
    if not settings.tickets_path.exists():
        return []
    saved = json.loads(settings.tickets_path.read_text(encoding="utf-8"))
    by_no = {
        t["ticket_no"]: {
            "ticket_no": t["ticket_no"],
            "subject": t["subject"],
            "description": t["description"],
            "domain": t["domain"],
            "category": t["category"],
            "priority": t["priority"],
            "status": t["status"],
            "department": t["department"],
            "company": None,
            "created_at": parse_dt(t["created_at"]),
            "closed_at": parse_dt(t.get("closed_at")),
        }
        for t in saved
    }
    # Ordered to match store.ids, so ticket[i] lines up with vectors[i].
    return [by_no[i] for i in store.ids if i in by_no]


def run_pipeline(
    source: str = "auto",
    limit: int | None = None,
    verbose: bool = True,
    file_path: str | None = None,
    incremental: bool = False,
) -> dict | None:
    store = VectorStore(settings.vector_store_dir)

    if incremental:
        if source not in ("auto", "mongo"):
            raise RuntimeError("incremental=True only supports source='mongo' (there's nothing to poll for a file/sample run).")
        known_tickets = _load_known_tickets(store)
        if verbose:
            print(f"[1/5] Checking MongoDB for tickets ({len(known_tickets)} already known)...")
        known_nos = {t["ticket_no"] for t in known_tickets}
        # Reads the whole collection (cheap - narrow projection, no embeddings) rather
        # than filtering by created_at server-side: created_at may be a native Date or
        # a string depending on the source system, and a server-side $gte against a
        # string field silently matches nothing (see from_mongo's docstring) - diffing
        # by ticket_no here is what's actually correct regardless of that.
        candidates = extract_tickets(source="mongo", limit=limit)
        new_tickets = [t for t in candidates if t["ticket_no"] not in known_nos]

        if not new_tickets:
            if verbose:
                print("      No new tickets found - nothing to do.")
            return None

        if verbose:
            print(f"      {len(new_tickets)} new ticket(s) found.")
            print("[2/5] Embedding just the new tickets (local model)...")
        new_texts = [build_embedding_text(t) for t in new_tickets]
        new_vectors = embed_texts(new_texts)
        store.append([t["ticket_no"] for t in new_tickets], new_vectors)

        tickets = known_tickets + new_tickets
        vectors = store.vectors
    else:
        if verbose:
            print(f"[1/5] Extracting tickets (source={source})...")
        tickets = extract_tickets(source=source, limit=limit, file_path=file_path)
        if not tickets:
            raise RuntimeError("No tickets extracted - check your MONGO_URI/MONGO_DB or the sample dataset.")

        if verbose:
            print(f"      {len(tickets)} tickets read.")
            print("[2/5] Cleaning text and generating embeddings (local model, first run downloads it)...")

        texts = [build_embedding_text(t) for t in tickets]
        vectors = embed_texts(texts)
        store.replace_all([t["ticket_no"] for t in tickets], vectors)

    return _finalize(tickets, vectors, source=source, verbose=verbose)


def add_manual_ticket(ticket: dict, verbose: bool = True) -> dict:
    """Phase 5, manual path: a single ticket submitted through the web form
    (`/submit` in app.py) instead of pulled from MongoDB. Embeds just that
    one ticket and appends it to the existing store/tickets, then re-runs
    the same clustering + report-data pass an incremental Mongo poll would -
    so a manually submitted ticket can join an existing recurring-issue
    cluster exactly like a real new ticket would."""
    store = VectorStore(settings.vector_store_dir)
    known_tickets = _load_known_tickets(store)

    if verbose:
        print(f"[1/3] Embedding submitted ticket {ticket['ticket_no']}...")
    vector = embed_texts([build_embedding_text(ticket)])
    store.append([ticket["ticket_no"]], vector)

    tickets = known_tickets + [ticket]
    vectors = store.vectors
    return _finalize(tickets, vectors, source="manual", verbose=verbose)


def _finalize(tickets: list[dict], vectors, source: str, verbose: bool) -> dict:
    """Clustering onward: everything that turns an already-extracted,
    already-embedded (tickets, vectors) pair into data/clusters.json +
    data/tickets.json. Shared by both run_pipeline branches above so a full
    run and an incremental run compute results the exact same way."""
    if verbose:
        print(f"[3/5] Clustering at similarity threshold {settings.similarity_threshold}...")
    clusters = cluster_tickets(vectors)

    now = max((t["created_at"] for t in tickets if t["created_at"]), default=datetime.now(timezone.utc))

    if verbose:
        print(f"[4/5] Labeling {sum(1 for c in clusters if len(c.ticket_indices) >= 2)} recurring clusters...")

    cluster_records = []
    for c in clusters:
        members = [tickets[i] for i in c.ticket_indices]
        size = len(members)
        subjects = [m["subject"] or m["description"][:60] for m in members]
        rep = tickets[c.representative_index]

        dated = [m["created_at"] for m in members if m["created_at"]]
        first_seen = min(dated) if dated else None
        last_seen = max(dated) if dated else None

        recent_30 = sum(1 for d in dated if (now - d) <= timedelta(days=30))
        prev_30 = sum(1 for d in dated if timedelta(days=30) < (now - d) <= timedelta(days=60))
        if prev_30 == 0:
            trend_pct = 100 if recent_30 > 0 else 0
        else:
            trend_pct = round((recent_30 - prev_30) / prev_30 * 100)

        mix = Counter(map_priority(m["priority"]) for m in members)
        dominant = mix.most_common(1)[0][0] if mix else "med"
        category = Counter(m["category"] for m in members).most_common(1)[0][0]

        cluster_records.append(
            {
                "issue": label_cluster(subjects) if size >= 2 else subjects[0],
                "category": category,
                "count": size,
                "trend_pct": trend_pct,
                "priority_mix": {k: mix.get(k, 0) for k in ("crit", "high", "med", "low")},
                "dominant_priority": dominant,
                "first_seen": first_seen.strftime("%Y-%m-%d") if first_seen else None,
                "last_seen": last_seen.strftime("%Y-%m-%d") if last_seen else None,
                "sample_ticket": rep["ticket_no"],
                "member_ticket_nos": [m["ticket_no"] for m in members],
            }
        )

    recurring = sorted(
        [c for c in cluster_records if c["count"] >= 2], key=lambda c: c["count"], reverse=True
    )

    if verbose:
        print("[5/5] Computing insights and writing data/clusters.json...")

    weeks_back = 8
    week_labels = []
    for w in range(weeks_back - 1, -1, -1):
        label_date = now - timedelta(days=w * 7)
        week_labels.append(f"{label_date.strftime('%b')} {label_date.day}")

    ticket_by_no = {t["ticket_no"]: t for t in tickets}
    trend_series = []
    for c in recurring[:4]:
        counts = [0] * weeks_back
        for ticket_no in c["member_ticket_nos"]:
            dt = ticket_by_no[ticket_no]["created_at"]
            if not dt:
                continue
            bucket = _week_bucket(dt, now)
            if 0 <= bucket < weeks_back:
                counts[weeks_back - 1 - bucket] += 1
        trend_series.append({"name": c["issue"], "data": counts})

    total = len(tickets)
    redundant = sum(c["count"] - 1 for c in recurring)
    repeat_rate = round(sum(c["count"] for c in recurring) / total * 100) if total else 0

    # A cluster's members ordered by when they were reported: the earliest is
    # the original report, everything after it is a redundant repeat of the
    # same issue. This "redundant" set is what every breakdown below tallies.
    redundant_ids: set[str] = set()
    for c in recurring:
        members_dated = sorted(
            c["member_ticket_nos"],
            key=lambda no: ticket_by_no[no]["created_at"] or datetime.max.replace(tzinfo=timezone.utc),
        )
        redundant_ids.update(members_dated[1:])

    redundant_still_open = sum(
        1 for t in tickets if t["ticket_no"] in redundant_ids and t["status"] in OPEN_STATUSES
    )

    # Per-ticket record for the RAG Q&A layer (phase 4): the vector store only
    # holds ids + vectors, so this is what turns a retrieved vector back into
    # readable text + its cluster context for grounding an LLM answer.
    cluster_meta_by_ticket: dict[str, dict] = {}
    for c in cluster_records:
        for ticket_no in c["member_ticket_nos"]:
            cluster_meta_by_ticket[ticket_no] = {
                "cluster_issue": c["issue"] if c["count"] >= 2 else None,
                "cluster_size": c["count"],
            }
    tickets_index = [
        {
            "ticket_no": t["ticket_no"],
            "subject": t["subject"],
            "description": t["description"][:500],
            "category": t["category"],
            "department": t["department"],
            "domain": t["domain"],
            "priority": t["priority"],
            "status": t["status"],
            "created_at": t["created_at"].strftime("%Y-%m-%d") if t["created_at"] else None,
            "closed_at": t["closed_at"].isoformat() if t.get("closed_at") else None,
            "is_redundant": t["ticket_no"] in redundant_ids,
            **cluster_meta_by_ticket.get(t["ticket_no"], {"cluster_issue": None, "cluster_size": 1}),
        }
        for t in tickets
    ]
    settings.tickets_path.write_text(json.dumps(tickets_index), encoding="utf-8")

    # Tickets that formed a cluster all on their own (no repeat has shown up
    # yet) - the "new / not-repeated" bucket a just-submitted ticket falls
    # into when it doesn't match anything already on record.
    non_repeating_tickets = sorted(
        (t for t in tickets_index if t["cluster_size"] == 1),
        key=lambda t: t["created_at"] or "",
        reverse=True,
    )

    # Resolution-time insights: only tickets with both a created_at and a
    # closed_at (real MongoDB tickets with a closed/reopened status, or a
    # sample-data ticket generated the same way) contribute here - a
    # just-submitted or still-open ticket has nothing to measure yet.
    resolved = [
        (t, (t["closed_at"] - t["created_at"]).total_seconds() / 3600)
        for t in tickets
        if t["created_at"] and t.get("closed_at") and t["closed_at"] >= t["created_at"]
    ]
    avg_resolution_hours = round(sum(h for _, h in resolved) / len(resolved), 1) if resolved else None
    reopened_count = sum(1 for t in tickets if "reopen" in (t["status"] or ""))
    reopened_rate_pct = round(reopened_count / total * 100) if total else 0

    def _resolution_row(t: dict, hours: float) -> dict:
        return {
            "ticket_no": t["ticket_no"],
            "subject": t["subject"],
            "category": t["category"],
            "priority": t["priority"],
            "priority_bucket": map_priority(t["priority"]),
            "resolution_hours": round(hours, 1),
            "created_at": t["created_at"].strftime("%Y-%m-%d") if t["created_at"] else None,
            "closed_at": t["closed_at"].strftime("%Y-%m-%d") if t["closed_at"] else None,
        }

    slowest_by_time = sorted(resolved, key=lambda pair: pair[1], reverse=True)[:8]
    fastest_by_time = sorted(resolved, key=lambda pair: pair[1])[:8]
    slowest_to_resolve = [_resolution_row(t, h) for t, h in slowest_by_time]
    fastest_to_resolve = [_resolution_row(t, h) for t, h in fastest_by_time]

    def _member_row(t: dict) -> dict:
        return {
            "ticket_no": t["ticket_no"],
            "subject": t["subject"],
            "description": (t["description"] or "")[:300],
            "domain": t["domain"],
            "category": t["category"],
            "priority": t["priority"],
            "priority_bucket": map_priority(t["priority"]),
            "status": t["status"],
            "department": t["department"],
            "created_at": t["created_at"].strftime("%Y-%m-%d") if t["created_at"] else None,
        }

    category_breakdown = _dimension_breakdown(tickets, redundant_ids, lambda t: t["category"])
    department_breakdown = _dimension_breakdown(tickets, redundant_ids, lambda t: t["department"])
    domain_breakdown = _dimension_breakdown(tickets, redundant_ids, lambda t: t["domain"], top_n=6)
    status_breakdown = _dimension_breakdown(tickets, redundant_ids, lambda t: t["status"], top_n=8)
    priority_breakdown = _dimension_breakdown(tickets, redundant_ids, lambda t: map_priority(t["priority"]), top_n=4)
    priority_breakdown.sort(key=lambda r: PRIORITY_ORDER.get(r["name"], 9))

    size_buckets = [("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, None)]
    cluster_size_distribution = []
    for label_, lo, hi in size_buckets:
        matched = [c for c in recurring if c["count"] >= lo and (hi is None or c["count"] <= hi)]
        if matched:
            cluster_size_distribution.append(
                {"bucket": label_, "clusters": len(matched), "tickets": sum(c["count"] for c in matched)}
            )

    overall_weeks_back = 12
    overall_week_labels = []
    for w in range(overall_weeks_back - 1, -1, -1):
        label_date = now - timedelta(days=w * 7)
        overall_week_labels.append(f"{label_date.strftime('%b')} {label_date.day}")

    total_by_week = [0] * overall_weeks_back
    redundant_by_week = [0] * overall_weeks_back
    for t in tickets:
        dt = t["created_at"]
        if not dt:
            continue
        bucket = _week_bucket(dt, now)
        if 0 <= bucket < overall_weeks_back:
            idx = overall_weeks_back - 1 - bucket
            total_by_week[idx] += 1
            if t["ticket_no"] in redundant_ids:
                redundant_by_week[idx] += 1
    repeat_rate_by_week = [round(r / tt * 100) if tt else 0 for r, tt in zip(redundant_by_week, total_by_week)]

    crit_high_redundant = sum(
        1 for t in tickets if t["ticket_no"] in redundant_ids and map_priority(t["priority"]) in ("crit", "high")
    )
    recommended_view, recommendation_reason = _recommend_view(
        total_by_week, redundant_by_week, overall_weeks_back, redundant, crit_high_redundant, category_breakdown, total
    )

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_start": min((t["created_at"] for t in tickets if t["created_at"]), default=now).strftime(
                "%Y-%m-%d"
            ),
            "window_end": now.strftime("%Y-%m-%d"),
            "source": source,
            "total_read": total,
        },
        "kpis": {
            "tickets_analyzed": total,
            "recurring_clusters": len(recurring),
            "repeat_rate_pct": repeat_rate,
            "redundant_tickets": redundant,
            "redundant_still_open": redundant_still_open,
            "non_repeating_tickets": len(non_repeating_tickets),
            "avg_resolution_hours": avg_resolution_hours,
            "reopened_rate_pct": reopened_rate_pct,
        },
        "top_issues": [
            {"issue": c["issue"], "category": c["category"], "count": c["count"]} for c in recurring[:8]
        ],
        "trend": {"weeks": week_labels, "series": trend_series},
        "overall_trend": {
            "weeks": overall_week_labels,
            "total": total_by_week,
            "redundant": redundant_by_week,
            "repeat_rate_pct": repeat_rate_by_week,
        },
        "breakdowns": {
            "category": category_breakdown,
            "department": department_breakdown,
            "domain": domain_breakdown,
            "priority": priority_breakdown,
            "status": status_breakdown,
        },
        "cluster_size_distribution": cluster_size_distribution,
        "resolution": {
            "slowest": slowest_to_resolve,
            "fastest": fastest_to_resolve,
        },
        "insights": {
            "recommended_view": recommended_view,
            "recommendation_reason": recommendation_reason,
        },
        "clusters": [
            {
                **{k: v for k, v in c.items() if k != "member_ticket_nos"},
                "members": sorted(
                    (_member_row(ticket_by_no[no]) for no in c["member_ticket_nos"]),
                    key=lambda m: m["created_at"] or "",
                    reverse=True,
                ),
            }
            for c in recurring
        ],
        "new_tickets": [
            {
                "ticket_no": t["ticket_no"],
                "subject": t["subject"],
                "domain": t["domain"],
                "category": t["category"],
                "priority": t["priority"],
                "priority_bucket": map_priority(t["priority"]),
                "department": t["department"],
                "created_at": t["created_at"],
            }
            for t in non_repeating_tickets
        ],
    }

    settings.clusters_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if verbose:
        print(f"      Done. {len(recurring)} recurring issues found across {total} tickets.")
        print(f"      Recommended view: {recommended_view} ({recommendation_reason})")
        print(f"      -> {settings.clusters_path}")
    return result
