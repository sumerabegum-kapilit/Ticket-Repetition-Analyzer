"""Pull tickets from MongoDB, or fall back to the bundled sample dataset.

Only the fields the dedup/insights pipeline actually needs are projected out
of your (much wider) tickets collection. Domain-specific fields live under
`domain_fields.*` and vary by ticket domain, so category is resolved by
trying several candidate paths in order.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from .config import settings

MONGO_PROJECTION = {
    "ticket_no": 1,
    "ticket_id": 1,
    "domain": 1,
    "category": 1,
    "service": 1,
    "subject": 1,
    "description": 1,
    "priority": 1,
    "status": 1,
    "reporter.department": 1,
    "reporter.name": 1,
    "domain_fields.category_type": 1,
    "domain_fields.category": 1,
    "domain_fields.issue_type": 1,
    "domain_fields.company": 1,
    "domain_fields.client_company": 1,
    "domain_fields.company_client": 1,
    "domain_fields.branch": 1,
    "domain_fields.product_name": 1,
    "created_at": 1,
    "closed_at": 1,
    "team_id": 1,
}


def _get_path(doc: dict, path: str) -> Any:
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first(doc: dict, *paths: str) -> Optional[Any]:
    for p in paths:
        v = _get_path(doc, p)
        if v:
            return v
    return None


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            v = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(v)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _clean(value: Any, default: str) -> str:
    """Collapse a possibly-messy free-text field (stray whitespace from CSV
    exports especially) down to a comparable label, falling back to `default`."""
    text = " ".join(str(value).split()) if value else ""
    return text or default


def _normalize(doc: dict) -> dict:
    category_raw = _first(
        doc,
        "domain_fields.category_type",
        "domain_fields.category",
        "domain_fields.issue_type",
        "category",
        "service",
    )
    company_raw = _first(
        doc, "domain_fields.company", "domain_fields.client_company", "domain_fields.company_client"
    )
    return {
        "ticket_no": doc.get("ticket_no") or str(doc.get("ticket_id") or doc.get("_id") or ""),
        "subject": (doc.get("subject") or "").strip(),
        "description": (doc.get("description") or "").strip(),
        "domain": _clean(doc.get("domain"), "unspecified"),
        "category": _clean(category_raw, "uncategorized"),
        "priority": str(doc.get("priority") or "medium").lower(),
        "status": _clean(doc.get("status"), "unknown").lower(),
        "department": _clean(_first(doc, "reporter.department"), "unspecified"),
        "company": _clean(company_raw, "") or None,
        "created_at": parse_dt(doc.get("created_at")),
        "closed_at": parse_dt(doc.get("closed_at")),
    }


def from_mongo(limit: Optional[int] = None) -> list[dict]:
    """Always reads the full collection (deliberately no server-side
    `created_at`-based filter) - the incremental pipeline run (phase 5) needs
    to work regardless of whether the source system stores `created_at` as a
    native BSON Date or as a string (both are common; a real export from this
    same field mixed a plain ISO string and a `Z`-suffixed one). A `$gte`
    query against a Python datetime silently matches zero documents when the
    field is a string, since MongoDB's BSON comparison order ranks every
    string below every date - that would make new tickets never get
    detected, silently, forever. So this stays correct-by-construction: read
    everything (cheap - a dozen small projected fields, no embeddings), and
    let the caller diff against tickets it already knows about by ticket_no,
    which works no matter how the date field is stored."""
    from pymongo import MongoClient

    if not settings.mongo_uri or not settings.mongo_db:
        raise RuntimeError(
            "MONGO_URI and MONGO_DB must be set (in .env) to extract from MongoDB. "
            "Leave them blank to run against the bundled sample dataset instead."
        )
    # Short timeout: phase 5's background poller and "check now" button call
    # this every AUTO_REFRESH_INTERVAL_SECONDS / on demand - if Mongo is
    # briefly unreachable it should fail fast, not hang on pymongo's ~30s
    # default and make the UI look stuck.
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=8000)
    try:
        coll = client[settings.mongo_db][settings.mongo_collection]
        cursor = coll.find({}, MONGO_PROJECTION)
        if limit:
            cursor = cursor.limit(limit)
        tickets = [_normalize(doc) for doc in cursor]
    finally:
        client.close()
    return [t for t in tickets if t["subject"] or t["description"]]


def from_sample(limit: Optional[int] = None) -> list[dict]:
    with open(settings.sample_data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if limit:
        raw = raw[:limit]
    return [_normalize(doc) for doc in raw]


def _unflatten(row: dict) -> dict:
    """Turn CSV-style dotted keys ("reporter.department") into a nested dict
    so `_normalize`'s `_get_path` lookups work the same as for a real Mongo doc.
    Keys like "assigned_to[]" / "attachments[].filename" (repeated/array
    sub-fields) are dropped - the pipeline doesn't need them."""
    out: dict = {}
    for key, value in row.items():
        if "[]" in key:
            continue
        parts = key.split(".")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                break
        else:
            cur[parts[-1]] = value if value != "" else None
    return out


def _read_csv(p) -> list[dict]:
    import csv

    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return [_unflatten(row) for row in csv.DictReader(f)]


def from_file(path: str, limit: Optional[int] = None) -> list[dict]:
    """Read tickets from a file exported from MongoDB.

    Supports:
    - .json  : a JSON array of ticket documents (mongoexport --jsonArray), or a single document
    - .jsonl : one JSON document per line (mongoexport default)
    - .bson  : a raw BSON collection dump (mongodump)
    - .csv   : a flattened CSV export (mongoexport --type=csv), dotted column
               names like "reporter.department" / "domain_fields.category_type"
    """
    from pathlib import Path

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".bson":
        import bson

        raw = bson.decode_all(p.read_bytes())
    elif suffix == ".jsonl":
        raw = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".csv":
        raw = _read_csv(p)
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]

    if limit:
        raw = raw[:limit]
    tickets = [_normalize(doc) for doc in raw]
    return [t for t in tickets if t["subject"] or t["description"]]


def extract_tickets(
    source: str = "auto",
    limit: Optional[int] = None,
    file_path: Optional[str] = None,
) -> list[dict]:
    """source: 'auto' (mongo if configured, else sample), 'mongo', 'sample', or 'file'."""
    if source == "file":
        if not file_path:
            raise RuntimeError("file_path is required when source='file'")
        return from_file(file_path, limit)
    if source == "mongo" or (source == "auto" and settings.mongo_uri and settings.mongo_db):
        return from_mongo(limit)
    return from_sample(limit)
