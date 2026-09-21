# Ticket Repetition Analyzer — Guide for Team Lead

This tool reads support tickets, automatically groups the ones that are really
the *same underlying problem* reported in different words, and shows how much
of the ticket load is repeat/duplicate work. No manual tagging required — it
uses AI (text embeddings) to detect that "VPN not connecting" and "unable to
access VPN" are the same issue.

## Latest results (workflowdesk production tickets export)

Report file: `reports/report_2026-09-15_121318.html` — open it directly in
any browser (Chrome, Edge, etc.), no server needed.

| Metric | Value | What it means |
|---|---|---|
| Tickets analyzed | 11,692 | Total tickets in the export, 2026-07-09 to 2026-09-08 |
| Distinct recurring issues | 947 | Number of *different* problems that came up more than once |
| Repeat rate | 34% | Of all tickets, 1 in 3 was actually a report of a problem that already existed |
| Redundant tickets | 2,985 | Tickets that could've been merged into an existing issue instead of handled as new |
| Still open | 191 | Of those redundant tickets, 191 are still open/in-progress — fixing the original issue could close these too |

**Bottom line:** roughly a third of the ticket volume is duplicate effort, and
191 tickets right now are open repeats of something already being worked on.

The report itself has 5 views (tabs): **Overview**, **Trends** (repeat volume
over time), **Categories & segments** (by category/department/domain),
**Priority & status**, and **All clusters** (full searchable table). It opens
automatically on whichever view best fits the data, with a note explaining why.

## How to run it yourself

### One-time setup (already done on this machine, for reference only)

```powershell
cd C:\RAG_1
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

### Option A — Upload a file from the browser (recommended)

1. Open a terminal in `C:\RAG_1` and run:
   ```powershell
   .venv\Scripts\python app.py
   ```
   Keep this window open — closing it stops the server.
2. Open **http://127.0.0.1:5000** in a browser.
3. Click the upload box and choose a file exported from MongoDB:
   - `mongoexport --db=<db> --collection=tickets --type=csv --fields=<...> --out=tickets.csv` (`.csv`)
   - `mongoexport --db=<db> --collection=tickets --jsonArray --out=tickets.json` (`.json`)
   - `mongoexport --db=<db> --collection=tickets --out=tickets.jsonl` (`.jsonl`)
   - `mongodump --db=<db> --collection=tickets --out=dump\` (`.bson`)
4. Click **Analyze**. The dashboard opens in the same tab when it's done.
   - First run ever downloads the embedding model (a few minutes, one-time).
   - ~11,700 tickets takes about 1 minute to process.

### Option B — Command line (live MongoDB connection, or the bundled demo data)

```powershell
.venv\Scripts\python run.py                  # uses MONGO_URI/MONGO_DB from .env if set, else demo data
.venv\Scripts\python run.py --source mongo    # force a live MongoDB connection
.venv\Scripts\python run.py --limit 500       # quick test on the first 500 tickets only
```

This writes a new file to `reports\report_<timestamp>.html` — open it in a browser.

## Where things are

- `reports\` — every generated dashboard (one HTML file per run)
- `data\clusters.json` — the raw clustering results behind the current report
- `.env` — connection settings (Mongo URI, similarity threshold, optional Claude API key for nicer issue names)

## Tuning notes

- `SIMILARITY_THRESHOLD` in `.env` (default `0.75`) controls how similar two
  tickets must be to count as "the same issue." Lower it if obvious
  duplicates are landing in separate groups; raise it if unrelated tickets
  are getting merged together.
- The uploaded file is deleted from disk immediately after processing — it
  is never stored or sent anywhere external. Embeddings run locally.
