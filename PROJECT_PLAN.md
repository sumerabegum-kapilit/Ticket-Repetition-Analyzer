# Ticket Repetition Analyzer — RAG Application Plan

> **Status:** Phases 1–5 are implemented (`src/`, `run.py`, `app.py`) — see
> the commands below to run it. Point it at your real MongoDB collection,
> or upload an export, by filling in `.env`. Phase 4 (Ask AI Q&A) additionally
> needs `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` set. Phase 5
> (automation) needs MongoDB configured — it doesn't apply to file uploads.

## 0. How to Run (start here)

One-time setup, then two ways to run it. All commands below are PowerShell,
run from the project root.

### Setup (once)

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Leave `.env` as-is to use the bundled sample dataset. To connect to real
data, fill in `MONGO_URI` / `MONGO_DB` / `MONGO_COLLECTION`, or just upload
an export file from the web UI (Option A below) — no `.env` changes needed
for that.

### Option A — Web app (recommended for a demo)

```powershell
.venv\Scripts\python app.py
```

Open **http://127.0.0.1:5000**, upload a file exported from your MongoDB
`tickets` collection, click **Analyze**. The dashboard opens automatically.
From there, click **Ask AI about this data** to ask natural-language
questions (needs the LLM setup below).

To export a file to upload:
```powershell
mongoexport --db=<db> --collection=tickets --jsonArray --out=tickets.json
```

### Option B — Command line

```powershell
.venv\Scripts\python run.py                  # auto: Mongo if configured in .env, else sample data
.venv\Scripts\python run.py --source sample  # force the bundled sample dataset
.venv\Scripts\python run.py --limit 500      # quick test on the first 500 tickets
```

Writes `data/clusters.json` and `reports/report_<timestamp>.html` — open the
HTML file in any browser to view the dashboard.

### Enabling Ask AI (Phase 4, optional)

Add to `.env`, then restart `app.py`:
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Automation (Phase 5) — keep the dashboard current as new tickets arrive

Requires `MONGO_URI` / `MONGO_DB` set in `.env` (it has nothing to poll for a
file upload or the sample dataset). Two ways to run it, pick one:

- **Keep `app.py` running** — it polls MongoDB every
  `AUTO_REFRESH_INTERVAL_SECONDS` (default 60s, set in `.env`) for tickets
  newer than the last check, and refreshes the dashboard automatically the
  moment any show up. There's also a **Check for new tickets now** button on
  the upload page for an on-demand check instead of waiting for the timer.
- **External scheduler** (Task Scheduler / cron), if you'd rather not keep
  `app.py` running all the time:
  ```powershell
  .venv\Scripts\python run.py --incremental
  ```
  Point a scheduled task at this command to run every minute (or whatever
  cadence you want) — it's a no-op (prints "no new tickets", doesn't touch
  the report) when there's nothing new.

Either way, only the new tickets get embedded — phases 1–3 (extract, embed,
cluster) still run, but re-use everything already processed instead of
starting over, so the update is fast even against a large existing dataset.

### Troubleshooting

- **Nothing runs / import errors:** make sure you activated the venv
  (`.venv\Scripts\python ...`, not a system `python`).
- **First run is slow:** the local embedding model downloads once and is
  cached after that.
- **"Failed to process file" / DLL errors:** delete `.venv` and reinstall
  (`py -m venv .venv` + `pip install -r requirements.txt`) — this project
  has no dependency on `scikit-learn` or other compiled tree-search
  libraries, so a clean install should not hit this.

See [README.md](README.md) for more detail (project layout, tuning the
similarity threshold, etc).

## 1. Problem Statement

You have a MongoDB collection of support tickets (subject, description, category,
domain, priority, reporter info, timestamps, etc.). Many incoming tickets are
**not truly new problems** — they are the same underlying issue reported again
and again in different words ("VPN not connecting", "unable to access VPN",
"VPN down since morning"). Today there's no way to see this at a glance.

**Goal:** Build a RAG-based application that reads `subject` + `description`
(plus supporting metadata) from the tickets table, semantically groups tickets
that represent the *same recurring issue* (even when worded differently), and
visually shows:
- How many distinct recurring issues exist
- How many tickets belong to each recurring issue ("repeat count")
- Which issues are trending / growing over time
- Ability to ask natural-language questions like *"What are the top 5 issues
  repeating this month?"* or *"How many network-related tickets repeated in
  the Finance branch?"*

## 2. Why RAG (not just keyword search)

Keyword/exact-match grouping fails because tickets use different phrasing for
the same problem. RAG solves this in two ways:
1. **Embeddings + similarity** turn subject/description into vectors so
   "printer not working" and "unable to print documents" land close together
   in vector space — this is what powers the *duplicate detection*.
2. **Retrieval + LLM** lets a human ask free-form questions and get answers
   grounded in the actual ticket data, not hallucinated — this is the
   *chat/Q&A layer* on top of the same vector index.

These are two related but distinct features of the same app. Duplicate
detection runs automatically over all tickets; the Q&A layer is used on
demand.

## 3. Decisions Made So Far

| Decision | Choice | Why |
|---|---|---|
| Source DB | MongoDB | Field names (`domain_fields.*`, `attachments[].*`) confirm this |
| Dashboard delivery | Static HTML report | No server to maintain; easy to email/share; regenerate on a schedule |
| Scale | Unknown / assume tens of thousands for now | Architecture below scales up without a rewrite |
| AI backend | **Hybrid** (recommended below) | Balances cost, privacy, and quality |

### AI backend recommendation (you were unsure — here's the reasoning)

Split the two features, because they have very different call volumes:

- **Duplicate detection runs over every ticket** (could be 50k+ embeddings).
  Use a **local, free, offline embedding model**
  (`sentence-transformers/all-MiniLM-L6-v2` or `bge-small-en`). No API cost,
  no data leaves your machine, and embedding quality is more than sufficient
  for grouping similar support tickets. This is the workhorse of the app.

- **The natural-language Q&A layer is used occasionally** (a handful of
  questions per session, not per ticket). Here quality matters more than
  cost, so a **cloud LLM API call** (e.g. Claude) is worth it — it only reads
  the small set of tickets retrieved for that specific question, not the
  whole database, so cost stays low.

You can start **100% local** (skip the Q&A/LLM layer, keep only the
clustering + dashboard) and add the cloud LLM later with no rework — it's an
additive layer on top of the same vector index.

## 4. High-Level Architecture

```
MongoDB (tickets collection)
        │
        ▼
┌───────────────────────┐
│ 1. Extract & Clean     │  pull subject + description + metadata
│    (incremental: only  │  strip HTML/signatures, normalize whitespace
│     new/updated rows)  │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. Embed               │  local model → vector per ticket
│    (sentence-          │  (subject weighted more than description,
│     transformers)      │   since subject is the concise signal)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. Vector Store        │  Chroma (embedded, local, free)
│    (persisted on disk) │  one entry per ticket_id + vector + metadata
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 4. Cluster / Dedup     │  group tickets whose vectors are within a
│    engine              │  similarity threshold → "recurring issue" clusters
└───────────┬───────────┘
            ▼
┌───────────────────────┐        ┌───────────────────────────┐
│ 5. Cluster store       │───────▶│ 6. Static HTML dashboard   │
│    (Mongo collection   │        │    (charts + tables,       │
│    "ticket_clusters")  │        │    generated by a script)  │
└───────────┬───────────┘        └───────────────────────────┘
            │
            ▼
┌───────────────────────┐
│ 7. (Optional) RAG Q&A  │  retrieve relevant tickets/clusters for a
│    layer, cloud LLM    │  question → LLM composes grounded answer
└───────────────────────┘
```

## 5. Duplicate / Repeat Detection Algorithm

This is the core logic — worth getting right:

1. **Text to embed:** `subject` (weighted higher, e.g. repeated or
   concatenated twice) + `description` (cleaned of signatures/greetings).
   Optionally include `category` / `domain_fields.category_type` as a soft
   filter so unrelated domains never get merged (e.g. don't cluster an HR
   ticket with a network ticket even if wording is vaguely similar).
2. **Vectorize** each ticket with the local embedding model.
3. **Group into clusters** — two practical options, pick based on results:
   - **Threshold + Union-Find:** compute cosine similarity between each new
     ticket and existing cluster centroids; if similarity ≥ threshold
     (start around 0.80–0.85, tune empirically), add to that cluster,
     else start a new cluster. Cheap, incremental, works well for
     streaming/new tickets.
   - **HDBSCAN / Agglomerative clustering:** run periodically over all
     vectors for a cleaner global clustering (good for a "re-cluster
     everything nightly" batch job, catches cases threshold-based
     incremental grouping missed).
   Start with option A (simpler, incremental-friendly), add option B as a
   nightly re-balancing job once the app is working.
4. **Cluster record** stored per group: `cluster_id`, `representative_subject`
   (the clearest/shortest ticket in the group, or an LLM-generated summary if
   the Q&A layer is enabled), `ticket_count`, `ticket_ids[]`, `first_seen`,
   `last_seen`, `category`, `domain`, `priority breakdown`.
5. **"Repeating" metric definitions** shown on the dashboard:
   - **Repeat count** = tickets in a cluster minus 1 (the first occurrence
     isn't a "repeat", the rest are).
   - **Repeat rate** = tickets in clusters of size ≥2 ÷ total tickets.
   - **Trend** = tickets per cluster per week/month, to spot issues that are
     *newly* spiking vs. steady background noise.

## 6. Dashboard (Static HTML Report)

Generated by a Python script (`generate_report.py`) that reads the
`ticket_clusters` collection and renders a single self-contained HTML file
(charts via Plotly's offline mode, or a hand-built artifact) with:

- **Headline numbers:** total tickets analyzed, number of distinct recurring
  issues, overall repeat rate %.
- **Top-N repeating issues bar chart** — cluster representative subject vs.
  ticket count.
- **Trend chart** — top clusters' ticket counts over time (weekly/monthly).
- **Breakdown by category/domain/priority** — stacked bar or treemap.
- **Sortable/filterable table** — every cluster with its count, category,
  date range, and a "view tickets" expandable list (client-side JS, no
  server needed, so it still counts as a static file).

Regeneration cadence: run the script daily/weekly (manually or via a
scheduled task) to refresh the report with the latest tickets.

## 7. Project Structure (proposed)

```
RAG_1/
├── config/
│   └── settings.yaml          # Mongo URI, thresholds, model names
├── src/
│   ├── extract.py             # pull tickets from MongoDB
│   ├── preprocess.py          # clean subject/description text
│   ├── embed.py                # local embedding generation
│   ├── vector_store.py        # Chroma wrapper
│   ├── cluster.py             # threshold + Union-Find clustering
│   ├── rebalance_job.py       # nightly HDBSCAN re-clustering
│   ├── rag_qa.py               # optional: retrieval + cloud LLM Q&A
│   └── generate_report.py     # builds the static HTML dashboard
├── data/
│   └── chroma/                # persisted vector store (gitignored)
├── reports/
│   └── report_YYYY-MM-DD.html # generated dashboards
└── requirements.txt
```

## 8. Implementation Phases

1. **Phase 1 — Data pipeline:** connect to MongoDB, extract & clean
   subject/description for a sample of tickets, confirm field mapping.
2. **Phase 2 — Embedding + clustering:** generate embeddings, implement
   threshold-based clustering, validate cluster quality manually on real
   data (tune the similarity threshold).
3. **Phase 3 — Dashboard:** build `generate_report.py`, produce the first
   real static HTML report from actual clusters.
4. **Phase 4 (optional, implemented) — RAG Q&A layer:** `src/rag_qa.py` +
   the `/ask` page in `app.py`. Retrieves the most relevant tickets from the
   existing vector store, pairs them with the dataset-wide stats already in
   `data/clusters.json`, and asks Claude to answer grounded only in that
   context. Needs `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` in `.env`.
5. **Phase 5 (implemented) — Automation:** `run_pipeline(incremental=True)`
   in `src/pipeline.py` fetches only tickets created after the newest one
   already in `data/tickets.json`, embeds just those, appends them to the
   vector store, then re-clusters and regenerates the report over the full
   (old + new) set - re-clustering is a cheap global pass, but embedding
   (the expensive step) never repeats work. Two ways to trigger it: `app.py`
   polls automatically on a timer while it's running (plus a manual "Check
   for new tickets now" button), or `python run.py --incremental` for an
   external scheduler (Task Scheduler / cron). See section 0 above.

## 9. Open Questions for You

- MongoDB connection details (host/URI, DB name, collection name) — needed
  to start Phase 1.
- Similarity threshold tuning needs a look at real examples of tickets you
  consider "the same issue" vs. "different issue" — a few real examples
  would help calibrate this fast.
- Do you want the Q&A layer (Phase 4) at all, or is the dashboard the whole
  scope for now?
